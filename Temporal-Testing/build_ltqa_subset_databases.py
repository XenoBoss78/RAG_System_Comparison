"""Create an LT-QA subset and build the selected retrieval databases.

The selected corpus documents are the union of the gold ``doc_ids`` for a
seeded LT-QA sample, plus an optional seeded sample of non-gold corpus
documents.  The script writes the questions, selected document IDs, and a
manifest before it starts database construction.

Examples (run from ``Temporal-Testing``)::

    # Build every current physical database for 150 LT-QA questions.
    python build_ltqa_subset_databases.py --questions 150 --systems all

    # Add 75 distractor documents and build only the systems of interest.
    python build_ltqa_subset_databases.py --questions 150 --extra-documents 75 \
        --systems chroma_metadata,engram_temporal --subset-name ltqa_150_plus_75

    # Build Temporal Engram with the expensive LLM-enriched ingestion stages.
    python build_ltqa_subset_databases.py --questions 100 --systems engram_temporal \
        --temporal-detail full

The query-only variants intentionally share stores with their base systems:

* ``chroma_metadata_multilayer``, ``chroma_metadata_verified``,
  ``chroma_metadata_yearly``, and ``chroma_metadata_nsq`` use the same metadata
  Chroma collection as ``chroma_metadata``.
* ``engram_temporal_nsq`` and ``engram_temporal_graph`` use the same Temporal
  Engram store as ``engram_temporal``.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent
FIN_RATE_DIR = WORKSPACE_ROOT / "Fin-RATE"
DEFAULT_QA_FILE = FIN_RATE_DIR / "qa" / "LT-QA.json"
DEFAULT_CORPUS_FILE = FIN_RATE_DIR / "corpus" / "corpus" / "corpus.jsonl"
DEFAULT_SUBSET_OUTPUT_DIR = SCRIPT_DIR / "subsets"
DEFAULT_STORE_ROOT = FIN_RATE_DIR / "ltqa_subset_stores"
DEFAULT_SEED = 42

SYSTEM_ALIASES = {"engram": "engram_base"}
ALL_SYSTEMS = (
    "chroma",
    "chroma_metadata",
    "chroma_metadata_multilayer",
    "chroma_metadata_verified",
    "chroma_metadata_yearly",
    "chroma_metadata_nsq",
    "engram_base",
    "engram_temporal",
    "engram_temporal_nsq",
    "engram_temporal_graph",
)


@dataclass(frozen=True)
class BuildTarget:
    """One physical store and the benchmark systems it can serve."""

    key: str
    systems: tuple[str, ...]


BUILD_TARGETS = (
    BuildTarget("chroma", ("chroma",)),
    BuildTarget(
        "chroma_metadata",
        (
            "chroma_metadata",
            "chroma_metadata_multilayer",
            "chroma_metadata_verified",
            "chroma_metadata_yearly",
            "chroma_metadata_nsq",
        ),
    ),
    BuildTarget("engram_base", ("engram_base",)),
    BuildTarget(
        "engram_temporal",
        ("engram_temporal", "engram_temporal_nsq", "engram_temporal_graph"),
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    """Return a filesystem-safe identifier while retaining readable names."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    if not normalized:
        raise ValueError("--subset-name must contain at least one letter or number")
    return normalized


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _load_ltqa_records(path: Path) -> list[dict[str, Any]]:
    """Load LT-QA lists and the supported wrapper formats used by this workspace."""
    payload = _read_json(path)
    if isinstance(payload, dict):
        for key in ("questions", "records", "data"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                payload = candidate
                break
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of LT-QA records")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"LT-QA record {index} in {path} is not a JSON object")
        question = str(item.get("question") or "").strip()
        doc_ids = item.get("doc_ids")
        if not question:
            raise ValueError(f"LT-QA record {index} has no question")
        if not isinstance(doc_ids, list) or not doc_ids:
            qid = item.get("q_id", f"index {index}")
            raise ValueError(f"LT-QA record {qid!r} has no non-empty doc_ids list")
        records.append(item)
    return records


def _corpus_document_ids(path: Path) -> list[str]:
    """Read IDs in corpus order so seeded sampling is stable across runs."""
    ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Corpus record at {path}:{line_number} is not a JSON object")
            doc_id = str(record.get("_id", record.get("id", f"line_{line_number}"))).strip()
            if doc_id and doc_id not in seen:
                ids.append(doc_id)
                seen.add(doc_id)
    if not ids:
        raise ValueError(f"No document IDs were found in {path}")
    return ids


def _select_subset(
    records: list[dict[str, Any]],
    corpus_ids: Iterable[str],
    *,
    question_count: int,
    extra_documents: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    """Select QA records, their gold documents, and optional non-gold extras."""
    if question_count <= 0:
        raise ValueError("--questions must be positive")
    if question_count > len(records):
        raise ValueError(
            f"--questions={question_count} exceeds the {len(records)} records available in the QA file"
        )
    if extra_documents < 0:
        raise ValueError("--extra-documents must be zero or positive")

    sampler = random.Random(seed)
    selected_questions = sampler.sample(records, question_count)
    gold_ids = {
        str(doc_id).strip()
        for record in selected_questions
        for doc_id in record["doc_ids"]
        if str(doc_id).strip()
    }
    if not gold_ids:
        raise ValueError("The selected LT-QA records did not contain usable gold document IDs")

    ordered_corpus_ids = list(corpus_ids)
    corpus_id_set = set(ordered_corpus_ids)
    missing_gold = sorted(gold_ids - corpus_id_set)
    if missing_gold:
        preview = ", ".join(missing_gold[:10])
        suffix = " ..." if len(missing_gold) > 10 else ""
        raise ValueError(
            f"{len(missing_gold)} gold document IDs are absent from the corpus: {preview}{suffix}"
        )

    extra_candidates = [doc_id for doc_id in ordered_corpus_ids if doc_id not in gold_ids]
    if extra_documents > len(extra_candidates):
        raise ValueError(
            f"--extra-documents={extra_documents} exceeds the {len(extra_candidates)} non-gold corpus documents"
        )
    extras = sampler.sample(extra_candidates, extra_documents)
    selected_ids = sorted(gold_ids | set(extras))
    return selected_questions, sorted(gold_ids), sorted(extras), selected_ids


def _normalise_systems(raw: str) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--systems must name at least one system or 'all'")
    if "all" in values:
        if len(values) != 1:
            raise ValueError("Use either --systems all or an explicit comma-separated system list")
        return list(ALL_SYSTEMS)

    systems: list[str] = []
    for value in values:
        value = SYSTEM_ALIASES.get(value, value)
        if value not in ALL_SYSTEMS:
            supported = ", ".join((*ALL_SYSTEMS, "all"))
            raise ValueError(f"Unknown system {value!r}. Supported values: {supported}")
        if value not in systems:
            systems.append(value)
    return systems


def _targets_for_systems(systems: Iterable[str]) -> list[BuildTarget]:
    requested = set(systems)
    return [target for target in BUILD_TARGETS if requested.intersection(target.systems)]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _command_for_target(
    target: BuildTarget,
    *,
    args: argparse.Namespace,
    document_ids_file: Path,
    store_root: Path,
    collection_name: str,
    namespace_prefix: str,
) -> tuple[list[str], Path]:
    """Build the existing scripts' command line for one physical store."""
    python = sys.executable
    common_chroma = [
        "--build",
        "--corpus",
        str(args.corpus.resolve()),
        "--document-ids-file",
        str(document_ids_file.resolve()),
        "--collection",
        collection_name,
        "--embedding-backend",
        args.embedding_backend,
        "--embedding-model",
        args.embedding_model,
        "--device",
        args.device,
        "--ollama-embed-url",
        args.ollama_url.rstrip("/") + "/api/embed",
        "--ollama-embedding-batch-size",
        str(args.ollama_embedding_batch_size),
        "--progress-every",
        str(args.chroma_progress_every),
        "--seed",
        str(args.seed),
    ]

    if target.key == "chroma":
        output = store_root / "chroma"
        command = [
            python,
            str(SCRIPT_DIR / "ChromaSetup.py"),
            *common_chroma,
            "--db-dir",
            str(output),
            "--batch-size",
            str(args.chroma_batch_size),
        ]
        return command, output

    if target.key == "chroma_metadata":
        # All metadata query variants read this schema-compatible collection.
        output = store_root / "chroma_metadata"
        command = [
            python,
            str(SCRIPT_DIR / "ChromaSetupMetaData.py"),
            *common_chroma,
            "--db-dir",
            str(output),
        ]
        return command, output

    if target.key == "engram_base":
        output = store_root / "engram_base"
        command = [
            python,
            str(SCRIPT_DIR / "Engram_Base.py"),
            "--build",
            "--corpus",
            str(args.corpus.resolve()),
            "--document-ids-file",
            str(document_ids_file.resolve()),
            "--store-dir",
            str(output),
            "--namespace",
            f"{namespace_prefix}-base",
            "--ollama-url",
            args.ollama_url,
            "--ollama-embed-model",
            args.engram_embed_model,
            "--seed",
            str(args.seed),
        ]
        if args.overwrite:
            command.append("--reset")
        return command, output

    if target.key == "engram_temporal":
        output = store_root / "engram_temporal"
        command = [
            python,
            str(SCRIPT_DIR / "Engram_Temporal.py"),
            "--build",
            "--corpus",
            str(args.corpus.resolve()),
            "--document-ids-file",
            str(document_ids_file.resolve()),
            "--store-dir",
            str(output),
            "--namespace",
            f"{namespace_prefix}-temporal",
            "--ollama-url",
            args.ollama_url,
            "--ollama-embed-model",
            args.engram_embed_model,
            "--progress-every",
            str(args.temporal_progress_every),
            "--seed",
            str(args.seed),
        ]
        if args.temporal_detail == "full":
            command.extend(
                [
                    "--ollama-model",
                    args.temporal_llm_model,
                    "--llm-document-metadata",
                    "--llm-summaries",
                    "--llm-filing-summaries",
                    "--metadata-llm-text-chars",
                    str(args.temporal_metadata_llm_text_chars),
                    "--filing-summary-input-chars",
                    str(args.temporal_filing_summary_input_chars),
                ]
            )
        if args.temporal_extract_text_facts:
            if args.temporal_detail != "full":
                command.extend(["--ollama-model", args.temporal_llm_model])
            command.append("--extract-text-facts")
        if args.overwrite:
            command.append("--reset")
        return command, output

    raise RuntimeError(f"No command builder for target {target.key!r}")


def _existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample LT-QA questions, add optional non-gold documents, and build the selected "
            "Chroma/Engram stores."
        )
    )
    parser.add_argument("--questions", type=int, required=True, help="Number of LT-QA records to sample.")
    parser.add_argument(
        "--systems",
        default="all",
        help=(
            "Comma-separated systems to support, or 'all'. Query-only variants reuse their base store. "
            f"Choices: {', '.join((*ALL_SYSTEMS, 'all'))}."
        ),
    )
    parser.add_argument(
        "--extra-documents",
        type=int,
        default=0,
        help="Seeded random corpus documents to add after excluding selected questions' gold IDs.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_FILE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_FILE)
    parser.add_argument(
        "--subset-name",
        help="Output label. Defaults to ltqa-q<questions>-extra<documents>-seed<seed>.",
    )
    parser.add_argument("--subset-output-dir", type=Path, default=DEFAULT_SUBSET_OUTPUT_DIR)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument(
        "--collection",
        help="Chroma collection name. Defaults to fin_rate_<subset-name>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow reuse of the named output. Chroma collections and selected Engram stores are reset."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the deterministic question/document selection and manifest without building stores.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selection and build plan without writing files or running builders.",
    )

    parser.add_argument(
        "--embedding-backend",
        choices=("default", "sentence-transformers", "bge", "ollama"),
        default="ollama",
        help="Embedding backend for the Chroma stores.",
    )
    parser.add_argument(
        "--embedding-model",
        default="embeddinggemma:latest",
        help="Embedding model for the selected Chroma backend.",
    )
    parser.add_argument("--device", default="cpu", help="Device for Sentence Transformers / BGE embeddings.")
    parser.add_argument("--engram-embed-model", default="embeddinggemma:latest")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-embedding-batch-size", type=int, default=32)
    parser.add_argument("--chroma-batch-size", type=int, default=256)
    parser.add_argument(
        "--chroma-progress-every",
        type=int,
        default=25,
        help="Print and flush Chroma build progress after this many documents; use 0 to disable.",
    )

    parser.add_argument(
        "--temporal-detail",
        choices=("standard", "full"),
        default="standard",
        help=(
            "'standard' uses deterministic document metadata, offline chunk summaries, and filing "
            "digests. 'full' adds LLM document metadata, chunk summaries, and filing summaries."
        ),
    )
    parser.add_argument("--temporal-llm-model", default="qwen3:4b")
    parser.add_argument("--temporal-metadata-llm-text-chars", type=int, default=6000)
    parser.add_argument("--temporal-filing-summary-input-chars", type=int, default=24000)
    parser.add_argument("--temporal-progress-every", type=int, default=25)
    parser.add_argument(
        "--temporal-extract-text-facts",
        action="store_true",
        help="Also extract LLM facts during the Temporal Engram build.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ollama_embedding_batch_size <= 0:
        raise ValueError("--ollama-embedding-batch-size must be positive")
    if args.chroma_batch_size <= 0:
        raise ValueError("--chroma-batch-size must be positive")
    if args.chroma_progress_every < 0:
        raise ValueError("--chroma-progress-every must be zero or positive")
    if args.temporal_progress_every < 0:
        raise ValueError("--temporal-progress-every must be zero or positive")
    if args.temporal_metadata_llm_text_chars <= 0:
        raise ValueError("--temporal-metadata-llm-text-chars must be positive")
    if args.temporal_filing_summary_input_chars <= 0:
        raise ValueError("--temporal-filing-summary-input-chars must be positive")

    args.qa_file = args.qa_file.expanduser().resolve()
    args.corpus = args.corpus.expanduser().resolve()
    args.subset_output_dir = args.subset_output_dir.expanduser().resolve()
    args.store_root = args.store_root.expanduser().resolve()
    if not args.qa_file.exists():
        raise FileNotFoundError(f"LT-QA file not found: {args.qa_file}")
    if not args.corpus.exists():
        raise FileNotFoundError(f"Corpus file not found: {args.corpus}")

    systems = _normalise_systems(args.systems)
    targets = _targets_for_systems(systems)
    default_name = f"ltqa-q{args.questions}-extra{args.extra_documents}-seed{args.seed}"
    subset_name = _safe_name(args.subset_name or default_name)
    collection_name = args.collection or f"fin_rate_{subset_name.replace('-', '_')}"
    namespace_prefix = f"fin-rate-{subset_name}"

    records = _load_ltqa_records(args.qa_file)
    corpus_ids = _corpus_document_ids(args.corpus)
    questions, gold_ids, extra_ids, selected_ids = _select_subset(
        records,
        corpus_ids,
        question_count=args.questions,
        extra_documents=args.extra_documents,
        seed=args.seed,
    )

    question_path = args.subset_output_dir / f"{subset_name}_questions.json"
    document_ids_path = args.subset_output_dir / f"{subset_name}_doc_ids.json"
    selection_path = args.subset_output_dir / f"{subset_name}_selection.json"
    store_dir = args.store_root / subset_name
    manifest_path = store_dir / "build_manifest.json"

    plans: list[dict[str, Any]] = []
    for target in targets:
        command, output_path = _command_for_target(
            target,
            args=args,
            document_ids_file=document_ids_path,
            store_root=store_dir,
            collection_name=collection_name,
            namespace_prefix=namespace_prefix,
        )
        plans.append(
            {
                "target": target.key,
                "requested_systems": [system for system in systems if system in target.systems],
                "all_compatible_systems": list(target.systems),
                "output_path": str(output_path),
                "command": command,
                "status": "pending",
            }
        )

    selection_summary = {
        "created_at_utc": _utc_now(),
        "subset_name": subset_name,
        "seed": args.seed,
        "qa_file": str(args.qa_file),
        "corpus_file": str(args.corpus),
        "question_count": len(questions),
        "gold_document_count": len(gold_ids),
        "extra_document_count": len(extra_ids),
        "selected_document_count": len(selected_ids),
        "question_ids": [str(record.get("q_id", "")) for record in questions],
        "gold_document_ids": gold_ids,
        "extra_document_ids": extra_ids,
        "selected_document_ids": selected_ids,
        "questions_file": str(question_path),
        "document_ids_file": str(document_ids_path),
    }
    manifest = {
        "created_at_utc": _utc_now(),
        "subset": selection_summary,
        "requested_systems": systems,
        "collection_name": collection_name,
        "engram_base_namespace": f"{namespace_prefix}-base",
        "engram_temporal_namespace": f"{namespace_prefix}-temporal",
        "temporal_detail": args.temporal_detail,
        "plans": plans,
    }

    print(
        "[LT-QA subset] "
        f"questions={len(questions)}, gold_documents={len(gold_ids)}, "
        f"extra_documents={len(extra_ids)}, selected_documents={len(selected_ids)}",
        flush=True,
    )
    print(f"[LT-QA subset] systems={', '.join(systems)}", flush=True)
    for plan in plans:
        print(
            f"[LT-QA build] {plan['target']} -> {plan['output_path']} "
            f"(for {', '.join(plan['requested_systems'])})",
            flush=True,
        )

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    existing = _existing_paths([question_path, document_ids_path, selection_path, manifest_path])
    existing.extend(_existing_paths(Path(plan["output_path"]) for plan in plans))
    if existing and not args.overwrite:
        paths = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to mix a new selection with existing output. Choose a new --subset-name or "
            f"pass --overwrite to reset the selected stores:\n  {paths}"
        )

    _write_json(question_path, questions)
    _write_json(document_ids_path, selected_ids)
    _write_json(selection_path, selection_summary)
    _write_json(manifest_path, manifest)
    print(f"[LT-QA subset] saved questions: {question_path}", flush=True)
    print(f"[LT-QA subset] saved document IDs: {document_ids_path}", flush=True)

    if args.prepare_only:
        print(f"[LT-QA subset] manifest saved without building: {manifest_path}", flush=True)
        return

    for plan in plans:
        plan["status"] = "running"
        plan["started_at_utc"] = _utc_now()
        _write_json(manifest_path, manifest)
        print(f"[LT-QA build] starting {plan['target']}", flush=True)
        started = time.perf_counter()
        try:
            subprocess.run(plan["command"], check=True)
        except subprocess.CalledProcessError as exc:
            plan["status"] = "failed"
            plan["return_code"] = exc.returncode
            plan["finished_at_utc"] = _utc_now()
            plan["elapsed_seconds"] = time.perf_counter() - started
            _write_json(manifest_path, manifest)
            raise RuntimeError(
                f"Build target {plan['target']!r} failed with exit code {exc.returncode}. "
                f"See the command output above and {manifest_path}."
            ) from exc
        plan["status"] = "completed"
        plan["return_code"] = 0
        plan["finished_at_utc"] = _utc_now()
        plan["elapsed_seconds"] = time.perf_counter() - started
        _write_json(manifest_path, manifest)
        print(
            f"[LT-QA build] completed {plan['target']} in {plan['elapsed_seconds']:.1f}s",
            flush=True,
        )

    manifest["completed_at_utc"] = _utc_now()
    _write_json(manifest_path, manifest)
    print(f"[LT-QA build] completed. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
