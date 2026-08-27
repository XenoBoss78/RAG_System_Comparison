"""Generate grounded answers from saved benchmark retrieval details.

This module never calls a retrieval backend.  It reads the ranked documents
already saved in one or more ``*_details.jsonl`` files, reconstructs their
evidence from the corpus, and sends only that evidence to an Ollama answer
model.  This makes answer-quality experiments reproducible without rerunning
Chroma, Engram, query decomposition, embeddings, or graph traversal.

Most Chroma detail records include a saved ``chunk_id`` and can be reproduced
at the chunk level.  Engram records commonly preserve an ``episode_id`` but
not the text or source chunk index.  For those records, the configurable
fallback uses the saved source document and marks that fact in the output.

Example (from ``Temporal-Testing``)::

    python answer_saved_benchmark_runs.py \
        --details-file ..\\Fin-RATE\\retrieval_benchmarks\\ltqa_subset_nsq_details.jsonl \
        --systems chroma_metadata_nsq,engram_temporal_nsq \
        --sample-size 20 \
        --answer-model qwen3:4b-instruct \
        --output-prefix ltqa_nsq_20_answers
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
FIN_RATE_DIR = WORKSPACE_ROOT / "Fin-RATE"
DEFAULT_CORPUS_PATH = FIN_RATE_DIR / "corpus" / "corpus" / "corpus.jsonl"
DEFAULT_OUTPUT_DIR = FIN_RATE_DIR / "retrieval_benchmarks"
DEFAULT_SEED = 42
DEFAULT_CONTEXT_CHARS = 16_000
DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")
CHUNK_ID_PATTERN = re.compile(r"::chunk_(\d+)$")

ANSWER_SYSTEM_PROMPT = (
    "Answer the financial QA question using only the retrieved evidence below. "
    "Do not use outside knowledge or assume facts that are absent from the evidence. "
    "If the evidence is insufficient or conflicting, say so clearly. "
    "Cite the supporting retrieved excerpts with their bracketed source labels, such as [1] or [2]."
)


@dataclass(frozen=True)
class SavedDetailRecord:
    """One completed benchmark row, retained with its source-file provenance."""

    source_path: Path
    line_number: int
    payload: dict[str, Any]

    @property
    def system(self) -> str:
        return str(self.payload.get("system") or "").strip()

    @property
    def qid(self) -> str:
        value = self.payload.get("qid") or self.payload.get("q_id")
        return str(value or "").strip()

    @property
    def run_id(self) -> str:
        return str(self.payload.get("run_id") or 1)

    @property
    def record_key(self) -> str:
        return f"{self.source_path.resolve()}::{self.line_number}::{self.system}::{self.run_id}"


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    title: str
    text: str


class OllamaAnswerClient:
    """Small dependency-free Ollama generation client for answer evaluation."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
        temperature: float = 0.0,
        seed: int = DEFAULT_SEED,
        num_predict: int = 768,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.seed = seed
        self.num_predict = num_predict

    @property
    def generate_url(self) -> str:
        return self.base_url if self.base_url.endswith("/api/generate") else f"{self.base_url}/api/generate"

    def complete(self, prompt: str, *, system: str = ANSWER_SYSTEM_PROMPT) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.num_predict,
            },
        }
        request = urllib.request.Request(
            self.generate_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {body or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and check the URL."
            ) from exc
        try:
            response_payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc
        answer = str(response_payload.get("response") or "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty answer")
        return answer


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_detail_records(paths: Iterable[Path]) -> list[SavedDetailRecord]:
    records: list[SavedDetailRecord] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Benchmark details file not found: {path}")
        with path.open("r", encoding="utf-8") as details_file:
            for line_number, line in enumerate(details_file, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(payload, dict):
                    raise ValueError(f"Detail record at {path}:{line_number} is not a JSON object")
                record = SavedDetailRecord(path, line_number, payload)
                if not record.system:
                    raise ValueError(f"Detail record at {path}:{line_number} has no system")
                if not record.qid:
                    raise ValueError(f"Detail record at {path}:{line_number} has no qid")
                records.append(record)
    if not records:
        raise ValueError("No benchmark records were found in the supplied details files")
    return records


def _parse_csv(raw: str | None) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _select_records(
    records: Sequence[SavedDetailRecord],
    *,
    systems: set[str] | None,
    qids: set[str] | None,
    run_id: int | None,
    sample_size: int | None,
    seed: int,
) -> list[SavedDetailRecord]:
    selected = [
        record
        for record in records
        if (systems is None or record.system in systems)
        and (qids is None or record.qid in qids)
        and (run_id is None or record.run_id == str(run_id))
    ]
    if not selected:
        raise ValueError("No detail records match the requested systems, qids, and run ID")

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("--sample-size must be positive")
        available_qids = sorted({record.qid for record in selected})
        if sample_size > len(available_qids):
            raise ValueError(
                f"--sample-size={sample_size} exceeds the {len(available_qids)} matching questions"
            )
        chosen_qids = set(random.Random(seed).sample(available_qids, sample_size))
        selected = [record for record in selected if record.qid in chosen_qids]

    # Preserve details-file order and benchmark ranking order, rather than sorting
    # by score or qid again.
    return selected


def _retrieved_items(payload: dict[str, Any], *, max_documents: int) -> list[dict[str, Any]]:
    if max_documents <= 0:
        raise ValueError("max_documents must be positive")
    retrieved = payload.get("retrieved")
    items: list[dict[str, Any]] = []
    if isinstance(retrieved, list):
        for item in retrieved:
            if isinstance(item, dict) and str(item.get("doc_id") or "").strip():
                items.append(item)
    if not items:
        for doc_id in payload.get("retrieved_doc_ids") or []:
            text = str(doc_id).strip()
            if text:
                items.append({"doc_id": text, "detail": {}})

    deduplicated: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    for item in items:
        doc_id = str(item.get("doc_id") or "").strip()
        if not doc_id or doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        deduplicated.append(item)
        if len(deduplicated) >= max_documents:
            break
    return deduplicated


def _needed_doc_ids(records: Iterable[SavedDetailRecord], *, max_documents: int) -> set[str]:
    return {
        str(item.get("doc_id") or "").strip()
        for record in records
        for item in _retrieved_items(record.payload, max_documents=max_documents)
        if str(item.get("doc_id") or "").strip()
    }


def _record_text(record: dict[str, Any]) -> str:
    value = record.get("text", record.get("content", record.get("contents", "")))
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _load_corpus_documents(corpus_path: Path, wanted_ids: set[str]) -> dict[str, CorpusDocument]:
    found: dict[str, CorpusDocument] = {}
    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {corpus_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Corpus record at {corpus_path}:{line_number} is not a JSON object")
            doc_id = str(record.get("_id", record.get("id", f"line_{line_number}"))).strip()
            if doc_id not in wanted_ids:
                continue
            title = record.get("title", "")
            found[doc_id] = CorpusDocument(
                doc_id=doc_id,
                title=title if isinstance(title, str) else str(title),
                text=_record_text(record),
            )
            if len(found) == len(wanted_ids):
                break
    return found


def _chunk_text(text: str, *, index: int, chunk_size_tokens: int, chunk_overlap_tokens: int) -> str | None:
    if index < 0:
        return None
    if chunk_size_tokens <= 0 or chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
    matches = list(TOKEN_PATTERN.finditer(text))
    if not matches:
        return None
    step = chunk_size_tokens - chunk_overlap_tokens
    start_token = index * step
    if start_token >= len(matches):
        return None
    end_token = min(len(matches), start_token + chunk_size_tokens)
    return text[matches[start_token].start() : matches[end_token - 1].end()]


def _saved_chunk_index(item: dict[str, Any]) -> int | None:
    detail = item.get("detail")
    detail = detail if isinstance(detail, dict) else {}
    chunk_id = str(detail.get("chunk_id") or item.get("chunk_id") or "")
    match = CHUNK_ID_PATTERN.search(chunk_id)
    if match:
        return int(match.group(1))
    for candidate in (detail.get("chunk_index"), item.get("chunk_index")):
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _format_evidence_block(
    *,
    rank: int,
    document: CorpusDocument,
    text: str,
    source_kind: str,
) -> str:
    heading = f"[{rank}] RETRIEVED {source_kind.upper()} | doc_id={document.doc_id}"
    if document.title:
        heading += f" | title={document.title}"
    return f"{heading}\n{text.strip()}"


def _assemble_context(
    record: SavedDetailRecord,
    corpus_docs: dict[str, CorpusDocument],
    *,
    context_mode: str,
    missing_chunk_policy: str,
    max_documents: int,
    max_context_chars: int,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> tuple[str, list[dict[str, Any]], dict[str, int]]:
    """Reconstruct ordered evidence without querying an index or graph."""
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")
    blocks: list[str] = []
    evidence: list[dict[str, Any]] = []
    remaining = max_context_chars
    source_kinds: Counter[str] = Counter()

    for rank, item in enumerate(_retrieved_items(record.payload, max_documents=max_documents), start=1):
        doc_id = str(item.get("doc_id") or "").strip()
        document = corpus_docs.get(doc_id)
        if document is None:
            evidence.append({"rank": rank, "doc_id": doc_id, "status": "missing_from_corpus"})
            source_kinds["missing_from_corpus"] += 1
            continue

        saved_chunk_index = _saved_chunk_index(item)
        text = ""
        source_kind = ""
        if context_mode == "source-document":
            text = document.text
            source_kind = "source_document"
        elif saved_chunk_index is not None:
            chunk = _chunk_text(
                document.text,
                index=saved_chunk_index,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
            if chunk:
                text = chunk
                source_kind = "saved_chunk_reconstructed"
            elif missing_chunk_policy == "source-document":
                text = document.text
                source_kind = "source_document_fallback_invalid_chunk"
            elif missing_chunk_policy == "first-chunk":
                text = _chunk_text(
                    document.text,
                    index=0,
                    chunk_size_tokens=chunk_size_tokens,
                    chunk_overlap_tokens=chunk_overlap_tokens,
                ) or ""
                source_kind = "first_chunk_fallback_invalid_chunk"
        elif missing_chunk_policy == "source-document":
            text = document.text
            source_kind = "source_document_fallback_no_chunk_id"
        elif missing_chunk_policy == "first-chunk":
            text = _chunk_text(
                document.text,
                index=0,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            ) or ""
            source_kind = "first_chunk_fallback_no_chunk_id"

        if not text.strip():
            evidence.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "status": "no_reconstructable_evidence",
                    "saved_chunk_index": saved_chunk_index,
                }
            )
            source_kinds["no_reconstructable_evidence"] += 1
            continue

        block = _format_evidence_block(
            rank=rank,
            document=document,
            text=text,
            source_kind=source_kind,
        )
        included_chars = min(len(block), remaining)
        included = block[:remaining]
        if len(block) > remaining:
            included = included.rsplit(" ", 1)[0].rstrip()
        if included.strip():
            blocks.append(included)
        evidence.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "status": "included" if included.strip() else "excluded_context_budget",
                "saved_chunk_index": saved_chunk_index,
                "context_source": source_kind,
                "available_context_chars": len(block),
                "included_context_chars": len(included),
            }
        )
        source_kinds[source_kind] += 1
        remaining -= included_chars + 7
        if remaining <= 0:
            break

    return "\n\n---\n\n".join(blocks), evidence, dict(source_kinds)


def _answer_prompt(question: str, context: str) -> str:
    return f"Question:\n{question}\n\nRetrieved evidence:\n{context}\n\nAnswer:"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _successful_record_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    complete: set[str] = set()
    with path.open("r", encoding="utf-8") as output_file:
        for line in output_file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(payload, dict)
                and str(payload.get("generated_answer") or "").strip()
                and not str(payload.get("answer_error") or "").strip()
            ):
                key = str(payload.get("source_record_key") or "")
                if key:
                    complete.add(key)
    return complete


def generate_answers_from_saved_details(
    *,
    details_files: Sequence[str | Path],
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    answer_model: str = "qwen3:4b-instruct",
    answer_llm: Callable[[str], str] | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_timeout: float = 300.0,
    temperature: float = 0.0,
    num_predict: int = 768,
    systems: Iterable[str] | None = None,
    qids: Iterable[str] | None = None,
    run_id: int | None = 1,
    sample_size: int | None = None,
    seed: int = DEFAULT_SEED,
    output_prefix: str | None = None,
    max_documents: int = 15,
    max_context_chars: int = DEFAULT_CONTEXT_CHARS,
    context_mode: str = "saved-chunk",
    missing_chunk_policy: str = "source-document",
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    save_context: bool = True,
    resume: bool = False,
) -> dict[str, Path]:
    """Generate one answer per saved benchmark row without rerunning retrieval.

    ``details_files`` may contain one multi-system benchmark run or several
    runs.  ``systems`` and ``qids`` filter those saved rows.  A seeded
    ``sample_size`` samples question IDs and retains every selected system's
    rows for those questions.

    ``context_mode='saved-chunk'`` reconstructs a saved Chroma chunk whenever
    its ``chunk_id``/``chunk_index`` was recorded.  For saved rows without that
    information (normally Engram), ``missing_chunk_policy`` controls whether
    to use the source document, its first chunk, or skip it.  The output marks
    every choice so answer comparisons remain interpretable.
    """
    if context_mode not in {"saved-chunk", "source-document"}:
        raise ValueError("context_mode must be 'saved-chunk' or 'source-document'")
    if missing_chunk_policy not in {"source-document", "first-chunk", "skip"}:
        raise ValueError("missing_chunk_policy must be 'source-document', 'first-chunk', or 'skip'")
    if max_documents <= 0:
        raise ValueError("max_documents must be positive")
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive")
    if ollama_timeout <= 0:
        raise ValueError("ollama_timeout must be positive")
    if num_predict <= 0:
        raise ValueError("num_predict must be positive")

    source_paths = [Path(path) for path in details_files]
    records = _read_detail_records(source_paths)
    selected = _select_records(
        records,
        systems={str(value).strip() for value in systems or [] if str(value).strip()} or None,
        qids={str(value).strip() for value in qids or [] if str(value).strip()} or None,
        run_id=run_id,
        sample_size=sample_size,
        seed=seed,
    )

    resolved_corpus = Path(corpus_path).expanduser().resolve()
    if not resolved_corpus.exists():
        raise FileNotFoundError(f"Corpus file not found: {resolved_corpus}")
    corpus_docs = _load_corpus_documents(
        resolved_corpus,
        _needed_doc_ids(selected, max_documents=max_documents),
    )

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or f"saved_benchmark_answers_{_utc_slug()}"
    answers_path = resolved_output_dir / f"{prefix}_answers.jsonl"
    summary_path = resolved_output_dir / f"{prefix}_summary.json"
    if answers_path.exists() and not resume:
        raise FileExistsError(
            f"Output already exists: {answers_path}. Choose another --output-prefix or use --resume."
        )

    completed_keys = _successful_record_keys(answers_path) if resume else set()
    llm = answer_llm or OllamaAnswerClient(
        answer_model,
        base_url=ollama_url,
        timeout=ollama_timeout,
        temperature=temperature,
        seed=seed,
        num_predict=num_predict,
    ).complete
    started_at = _utc_now()
    generated = 0
    skipped_existing = 0
    answer_errors = 0
    context_source_counts: Counter[str] = Counter()
    write_mode = "a" if resume else "w"

    with answers_path.open(write_mode, encoding="utf-8") as answers_file:
        for index, record in enumerate(selected, start=1):
            if record.record_key in completed_keys:
                skipped_existing += 1
                continue
            payload = record.payload
            question = str(payload.get("question") or "").strip()
            context, evidence, source_counts = _assemble_context(
                record,
                corpus_docs,
                context_mode=context_mode,
                missing_chunk_policy=missing_chunk_policy,
                max_documents=max_documents,
                max_context_chars=max_context_chars,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
            context_source_counts.update(source_counts)
            answer = ""
            answer_error = ""
            answer_latency_ms = 0.0
            if not question:
                answer_error = "Saved benchmark row has no question."
            elif not context.strip():
                answer_error = "Answer skipped because no reconstructable retrieved evidence was available."
            else:
                print(
                    f"[Saved-answer generation] {index}/{len(selected)} | {record.system} | {record.qid}",
                    flush=True,
                )
                answer_started = time.perf_counter()
                try:
                    answer = str(llm(_answer_prompt(question, context))).strip()
                except Exception as exc:  # noqa: BLE001 - preserve context for a failed LLM call
                    answer_error = f"{type(exc).__name__}: {exc}"
                answer_latency_ms = (time.perf_counter() - answer_started) * 1000.0
            if answer_error:
                answer_errors += 1
            elif answer:
                generated += 1

            result = {
                "source_record_key": record.record_key,
                "details_file": str(record.source_path),
                "details_line_number": record.line_number,
                "run_id": record.run_id,
                "system": record.system,
                "qid": record.qid,
                "question": question,
                "gold_answer": str(payload.get("gold_answer") or payload.get("answer") or ""),
                "gold_key_points": payload.get("gold_key_points") or payload.get("key_points") or [],
                "gold_doc_ids": [str(value) for value in payload.get("gold_doc_ids") or payload.get("doc_ids") or []],
                "retrieval_error": str(payload.get("error") or ""),
                "retrieved_doc_ids": [
                    str(value) for value in payload.get("retrieved_doc_ids") or [] if str(value).strip()
                ],
                "retrieved": payload.get("retrieved") or [],
                "context_mode": context_mode,
                "missing_chunk_policy": missing_chunk_policy,
                "evidence": evidence,
                "context_source_counts": source_counts,
                "context_chars": len(context),
                "answer_model": answer_model,
                "answer_system_prompt": ANSWER_SYSTEM_PROMPT,
                "answer_generation_latency_ms": round(answer_latency_ms, 3),
                "answer_error": answer_error,
                "generated_answer": answer,
            }
            if save_context:
                result["context"] = context
            answers_file.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            answers_file.flush()

    summary = {
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "details_files": [str(path.expanduser().resolve()) for path in source_paths],
        "corpus_path": str(resolved_corpus),
        "answer_model": answer_model,
        "systems": sorted({record.system for record in selected}),
        "questions": len({record.qid for record in selected}),
        "benchmark_rows_selected": len(selected),
        "answers_generated": generated,
        "answer_errors": answer_errors,
        "answers_skipped_on_resume": skipped_existing,
        "max_documents": max_documents,
        "max_context_chars": max_context_chars,
        "context_mode": context_mode,
        "missing_chunk_policy": missing_chunk_policy,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "context_source_counts": dict(context_source_counts),
        "answers_jsonl": str(answers_path),
    }
    _write_json(summary_path, summary)
    return {"answers_jsonl": answers_path, "summary_json": summary_path}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grounded answers from saved benchmark details without rerunning retrieval."
    )
    parser.add_argument(
        "--details-file",
        type=Path,
        action="append",
        required=True,
        help="A *_details.jsonl benchmark file. Repeat this option to compare separate runs.",
    )
    parser.add_argument("--systems", help="Optional comma-separated saved system names to answer.")
    parser.add_argument("--qids", help="Optional comma-separated LT-QA IDs to answer.")
    parser.add_argument(
        "--run-id",
        type=int,
        default=1,
        help="Saved benchmark repetition to use. Pass --all-runs to keep every repetition.",
    )
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Seeded random number of question IDs to use; all selected systems are retained per question.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix")
    parser.add_argument("--answer-model", default="qwen3:4b-instruct")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=768)
    parser.add_argument("--max-documents", type=int, default=15)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_CONTEXT_CHARS)
    parser.add_argument(
        "--context-mode",
        choices=("saved-chunk", "source-document"),
        default="saved-chunk",
    )
    parser.add_argument(
        "--missing-chunk-policy",
        choices=("source-document", "first-chunk", "skip"),
        default="source-document",
        help="How to reconstruct saved rows that lack a Chroma chunk ID (normally Engram rows).",
    )
    parser.add_argument("--chunk-size-tokens", type=int, default=DEFAULT_CHUNK_SIZE_TOKENS)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--no-save-context", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only unfinished rows to an existing answers JSONL with the same output prefix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report the selected saved rows without calling Ollama or writing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ollama_timeout <= 0:
        raise ValueError("--ollama-timeout must be positive")
    if args.num_predict <= 0:
        raise ValueError("--num-predict must be positive")
    if args.chunk_size_tokens <= 0 or args.chunk_overlap_tokens < 0:
        raise ValueError("chunk size must be positive and overlap must be non-negative")
    if args.chunk_overlap_tokens >= args.chunk_size_tokens:
        raise ValueError("--chunk-overlap-tokens must be smaller than --chunk-size-tokens")

    systems = _parse_csv(args.systems)
    qids = _parse_csv(args.qids)
    if args.dry_run:
        records = _select_records(
            _read_detail_records(args.details_file),
            systems=set(systems) or None,
            qids=set(qids) or None,
            run_id=None if args.all_runs else args.run_id,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        payload = {
            "details_files": [str(path.expanduser().resolve()) for path in args.details_file],
            "systems": sorted({record.system for record in records}),
            "questions": len({record.qid for record in records}),
            "benchmark_rows_selected": len(records),
            "retrieved_document_ids": len(_needed_doc_ids(records, max_documents=args.max_documents)),
            "answer_model": args.answer_model,
            "retrieval_rerun": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    client = OllamaAnswerClient(
        args.answer_model,
        base_url=args.ollama_url,
        timeout=args.ollama_timeout,
        temperature=args.temperature,
        seed=args.seed,
        num_predict=args.num_predict,
    )
    paths = generate_answers_from_saved_details(
        details_files=args.details_file,
        corpus_path=args.corpus,
        output_dir=args.output_dir,
        answer_model=args.answer_model,
        answer_llm=client.complete,
        ollama_url=args.ollama_url,
        ollama_timeout=args.ollama_timeout,
        temperature=args.temperature,
        num_predict=args.num_predict,
        systems=systems or None,
        qids=qids or None,
        run_id=None if args.all_runs else args.run_id,
        sample_size=args.sample_size,
        seed=args.seed,
        output_prefix=args.output_prefix,
        max_documents=args.max_documents,
        max_context_chars=args.max_context_chars,
        context_mode=args.context_mode,
        missing_chunk_policy=args.missing_chunk_policy,
        chunk_size_tokens=args.chunk_size_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        save_context=not args.no_save_context,
        resume=args.resume,
    )
    print(f"answers JSONL: {paths['answers_jsonl']}")
    print(f"summary JSON: {paths['summary_json']}")


if __name__ == "__main__":
    main()
