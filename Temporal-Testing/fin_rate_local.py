"""Build and query a local Engram memory store from the Fin-RATE corpus.

This mirrors the local corpus.jsonl workflow used by the vector/Chroma scripts in
the parent workspace, but writes into Engram instead:

  * raw filing chunks are stored as dated episodes for retrieval
  * deterministic metadata facts seed Engram's semantic graph locally
  * corpus-derived edges add longitudinal and comparison relationships
  * the result is persisted as JSONL under Fin-RATE/engram_data

Start small from the Temporal-Testing directory:

    python fin_rate_local.py --build --limit 200 --reset
    python fin_rate_local.py --query "Valero financial results" --ollama-model llama3.1:8b
    python fin_rate_local.py --graph-query VLO --export-graph graph.json

Then remove --limit for a full local build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .reproducibility import DEFAULT_SEED, seed_everything
except ImportError:
    from reproducibility import DEFAULT_SEED, seed_everything


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
ENGRAM_DIR = WORKSPACE_DIR / "engram"
DEFAULT_CORPUS_PATH = WORKSPACE_DIR / "Fin-RATE" / "corpus" / "corpus" / "corpus.jsonl"
DEFAULT_DATA_DIR = WORKSPACE_DIR / "Fin-RATE" / "engram_data"
DEFAULT_GRAPH_EXPORT = WORKSPACE_DIR / "Fin-RATE" / "engram_graph.json"
DEFAULT_NAMESPACE = "fin-rate"

sys.path.insert(0, str(ENGRAM_DIR))

from engram import Memory  # noqa: E402
from engram.consolidate.classify import classify_fact  # noqa: E402
from engram.embed.base import Embedder  # noqa: E402
from engram.llm.base import LLM  # noqa: E402
from engram.types import Fact  # noqa: E402


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

FIN_RATE_SUMMARY_SYSTEM = (
    "You summarize one SEC filing chunk for a financial retrieval memory index. Preserve exact ticker, "
    "company names, filing form, filing date, fiscal periods, section/item names, financial metrics, "
    "amounts, units, percentages, risks, exhibits, named executives, auditors, accounting standards, and "
    "cross-references. Be concise but do not drop numbers or period labels. No preamble."
)

FIN_RATE_EXTRACT_SYSTEM = (
    "You are an SEC filing information-extraction engine for a graph memory store. Extract only facts "
    "explicitly stated in the filing chunk. Output ONLY a JSON array of objects with keys "
    "\"subject\", \"predicate\", \"object\", and \"text\". Every value must be a plain string; never use "
    "nested objects, arrays, null, {}, or [] inside a fact. Use concise snake_case predicates. Good "
    "predicates include reports_metric, has_revenue, has_net_income, has_segment, discusses_risk, "
    "has_exhibit, references_filing, signed_by, audited_by, has_accounting_policy, has_subsidiary, "
    "announced_results_for_period, incorporated_by_reference, and defines_term. Include exact periods, "
    "units, and amounts in the object when present. Prefer subjects like the ticker/company, the filing, "
    "the document id, the section, or a named entity from the chunk. Do not infer, calculate, normalize, "
    "or invent values. Skip any fact whose object would be empty or vague. Limit output to the 12 most useful durable facts. If there are no useful facts, "
    "output []."
)

FIN_RATE_ANSWER_SYSTEM = (
    "You answer financial-analysis questions using only the provided Engram retrieval context. Be concise, "
    "cite document ids, filing dates, or section names when available, and say when the context is "
    "insufficient. Do not use outside knowledge."
)


@dataclass(frozen=True)
class FilingMeta:
    doc_id: str
    title: str
    ticker: str
    filing_date: str
    year: str
    form_type: str
    section: str
    section_family: str
    comparison_group: str
    filing_node: str
    section_node: str
    event_time: float


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and check the URL."
            ) from exc
        return json.loads(raw) if raw else {}

    def models(self) -> list[str]:
        return [m.get("name", "") for m in self.request("/api/tags").get("models", [])]


class OllamaLLM(LLM):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        num_predict: int = 768,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.model = model
        self.client = OllamaClient(base_url, timeout)
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.seed = seed

    def complete(self, prompt: str, system: str | None = None, **kwargs) -> str:
        options = {
            "temperature": kwargs.pop("temperature", self.temperature),
            "num_ctx": kwargs.pop("num_ctx", self.num_ctx),
            "num_predict": kwargs.pop("num_predict", self.num_predict),
            "seed": kwargs.pop("seed", self.seed),
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        payload.update(kwargs)
        return str(self.client.request("/api/generate", payload).get("response", "")).strip()


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
    ) -> None:
        self.model_name = f"ollama:{model}"
        self.model = model
        self.client = OllamaClient(base_url, timeout)
        self.dim = 0

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        try:
            response = self.client.request("/api/embed", {"model": self.model, "input": items})
            vectors = response.get("embeddings")
            if isinstance(vectors, list) and vectors:
                self.dim = len(vectors[0])
                return vectors
        except RuntimeError:
            raise
        except Exception:
            pass

        vectors = []
        for text in items:
            response = self.client.request("/api/embeddings", {"model": self.model, "prompt": text})
            vector = response.get("embedding")
            if not isinstance(vector, list):
                raise RuntimeError(
                    f"Ollama model {self.model!r} did not return embeddings. "
                    "Use an embedding model such as nomic-embed-text."
                )
            vectors.append(vector)
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _service_namespace_dir(data_dir: Path, namespace: str) -> Path:
    """Match MemoryService._safe_user so the same store can be served by /ui."""
    raw = str(namespace)
    prefix = re.sub(r"[^A-Za-z0-9]+", "-", raw).strip("-").lower()[:48]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return data_dir / f"{prefix or 'namespace'}--{digest}"


def _tokens(text: str) -> list[re.Match[str]]:
    return list(TOKEN_PATTERN.finditer(text))


def _token_windows(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> Iterable[tuple[str, int, int]]:
    if overlap_tokens >= max_tokens:
        raise ValueError("chunk overlap must be smaller than chunk size")

    matches = _tokens(text)
    if not matches:
        stripped = text.strip()
        if stripped:
            yield stripped, 0, 0
        return

    step = max(1, max_tokens - overlap_tokens)
    for token_start in range(0, len(matches), step):
        token_end = min(len(matches), token_start + max_tokens)
        char_start = matches[token_start].start()
        char_end = matches[token_end - 1].end()
        chunk = text[char_start:char_end].strip()
        if chunk:
            yield chunk, token_start, token_end
        if token_end >= len(matches):
            break


def _record_text(record: dict[str, Any]) -> str:
    text = record.get("text", record.get("content", record.get("contents", "")))
    return text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)


def _doc_id(record: dict[str, Any], line_number: int) -> str:
    return str(record.get("_id", record.get("id", f"line_{line_number}")))


def _title(record: dict[str, Any]) -> str:
    title = record.get("title", "")
    return title if isinstance(title, str) else str(title)


def _humanize_slug(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip()


def _item_code(value: str) -> str:
    if value.isdigit() and len(value) == 3:
        return f"{value[0]}.{value[1:]}"
    return value


def _section_from_title(title: str) -> str:
    marker = "_item_"
    lower = title.lower()
    if marker not in lower:
        return "unknown"

    # Preserve the original title casing where possible while using the lower-case
    # marker position for robust matching.
    tail = title[lower.index(marker) + len(marker):]
    parts = [part for part in tail.split("_") if part]
    if not parts:
        return "unknown"

    code = _item_code(parts[0])
    label = _humanize_slug("_".join(parts[1:])) if len(parts) > 1 else ""
    if code == "unknown":
        return label or "unknown"
    return f"Item {code}: {label}" if label else f"Item {code}"


def _section_family_from_title(title: str) -> str:
    marker = "_item_"
    lower = title.lower()
    if marker not in lower:
        return "unknown"
    tail = title[lower.index(marker) + len(marker):]
    parts = [part for part in tail.split("_") if part]
    if not parts:
        return "unknown"

    code = _item_code(parts[0])
    if code != "unknown":
        return f"Item {code}"

    label = _humanize_slug("_".join(parts[1:]))
    if not label:
        return "unknown"
    return "unknown: " + " ".join(label.split()[:6])


def _event_time(date_text: str) -> float:
    if DATE_RE.match(date_text):
        dt = datetime.strptime(date_text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return time.time()


def parse_filing_meta(record: dict[str, Any], line_number: int) -> FilingMeta:
    doc_id = _doc_id(record, line_number)
    title = _title(record)
    parts = title.split("_")
    ticker = parts[0] if len(parts) >= 1 and parts[0] else "UNKNOWN"
    filing_date = parts[1] if len(parts) >= 2 and DATE_RE.match(parts[1]) else "unknown-date"
    year = filing_date[:4] if DATE_RE.match(filing_date) else "unknown-year"
    form_type = parts[2] if len(parts) >= 3 and parts[2] else "unknown-form"
    section = _section_from_title(title)
    section_family = _section_family_from_title(title)

    filing_node = f"{ticker} {form_type} {filing_date}"
    section_node = f"{filing_node} | {section}"
    comparison_group = f"{form_type} {year} {section_family}"
    return FilingMeta(
        doc_id=doc_id,
        title=title,
        ticker=ticker,
        filing_date=filing_date,
        year=year,
        form_type=form_type,
        section=section,
        section_family=section_family,
        comparison_group=comparison_group,
        filing_node=filing_node,
        section_node=section_node,
        event_time=_event_time(filing_date),
    )


def add_metadata_fact(
    mem: Memory,
    seen: set[tuple[str, str, str]],
    *,
    user_id: str,
    subject: str,
    predicate: str,
    object_: str,
    valid_at: float,
    provenance: list[str] | None = None,
) -> bool:
    """Insert deterministic corpus metadata without conflict-resolution churn.

    The predicates intentionally use the multi-valued `has_*` pattern so filings
    can accumulate many sections/documents without older edges being retired.
    """
    subject = subject.strip()
    predicate = predicate.strip()
    object_ = object_.strip()
    if not subject or not predicate or not object_:
        return False

    key = (subject.lower(), predicate.lower(), object_.lower())
    if key in seen:
        return False
    seen.add(key)

    fact = Fact(
        subject=subject,
        predicate=predicate,
        object=object_,
        user_id=user_id,
        source="metadata",
        valid_at=valid_at,
        created_at=time.time(),
        provenance=provenance or [],
    )
    fact.embedding = mem.embedder.embed(fact.text)
    classify_fact(fact)
    mem.fact_store.upsert(fact.id, fact.embedding, fact)
    mem.engine.graph_builder.add_fact(fact)
    return True


def _existing_fact_keys(mem: Memory) -> set[tuple[str, str, str]]:
    return {
        (f.subject.lower(), f.predicate.lower(), f.object.lower())
        for f in mem.fact_store.values() + mem.cold_store.values()
    }


def strip_user_profile_block(context: str) -> str:
    if not context.startswith("USER PROFILE:"):
        return context
    markers = (
        "\n\nFACTS",
        "\n\nCURRENT STATE",
        "\n\nFACT HISTORY",
        "\n\nFACT EVOLUTION",
        "\n\nSESSION SUMMARIES",
        "\n\nRELEVANT CONVERSATIONS",
        "\n\nPROVENANCE RAW EVIDENCE",
    )
    starts = [context.find(marker) for marker in markers if context.find(marker) > 0]
    if not starts:
        return context
    return context[min(starts) + 2 :]


def open_memory(
    store_dir: Path,
    *,
    llm: LLM | None = None,
    embedder: Embedder | None = None,
    fin_rate_prompts: bool = False,
) -> Memory:
    kwargs: dict[str, Any] = {}
    if llm is not None:
        kwargs["llm"] = llm
    if embedder is not None:
        kwargs["embedder"] = embedder
    mem = Memory.open(str(store_dir), **kwargs)
    if fin_rate_prompts and llm is not None:
        mem.set_policy(
            extract_system=FIN_RATE_EXTRACT_SYSTEM,
            summary_system=FIN_RATE_SUMMARY_SYSTEM,
        )
    return mem


def _unique_metas(metas: Iterable[FilingMeta], key) -> list[FilingMeta]:
    by_key: dict[str, FilingMeta] = {}
    for meta in metas:
        by_key.setdefault(key(meta), meta)
    return list(by_key.values())


def _ordered_unique_metas(metas: Iterable[FilingMeta], key) -> list[FilingMeta]:
    return sorted(
        _unique_metas(metas, key),
        key=lambda meta: (meta.event_time, meta.filing_node, meta.section_family, meta.doc_id),
    )


def add_deep_relationships(
    mem: Memory,
    seen_facts: set[tuple[str, str, str]],
    *,
    metas: list[FilingMeta],
    doc_provenance: dict[str, list[str]],
    namespace: str,
) -> int:
    """Add bounded document relationships derived from the whole corpus index.

    Shared concepts are represented as hub nodes, and temporal relationships
    connect adjacent records only. That gives useful graph traversal without
    creating all-to-all same-company or same-section cliques.
    """
    added = 0

    def add(subject: str, predicate: str, object_: str, meta: FilingMeta) -> None:
        nonlocal added
        provenance = doc_provenance.get(meta.doc_id, [])
        if add_metadata_fact(
            mem,
            seen_facts,
            user_id=namespace,
            subject=subject,
            predicate=predicate,
            object_=object_,
            valid_at=meta.event_time,
            provenance=provenance[:1],
        ):
            added += 1

    for meta in metas:
        # Make documents first-class graph nodes instead of only leaf objects.
        add(meta.doc_id, "belongs_to_company", meta.ticker, meta)
        add(meta.doc_id, "belongs_to_filing", meta.filing_node, meta)
        add(meta.doc_id, "has_form_type", meta.form_type, meta)
        add(meta.doc_id, "has_filing_date", meta.filing_date, meta)
        add(meta.doc_id, "has_year", meta.year, meta)
        add(meta.doc_id, "has_section", meta.section_node, meta)
        add(meta.doc_id, "has_section_family", meta.section_family, meta)
        add(meta.doc_id, "has_comparison_group", meta.comparison_group, meta)
        add(meta.comparison_group, "has_member_document", meta.doc_id, meta)

    by_filing: dict[str, list[FilingMeta]] = defaultdict(list)
    by_company_section: dict[tuple[str, str, str], list[FilingMeta]] = defaultdict(list)
    by_company_form: dict[tuple[str, str], list[FilingMeta]] = defaultdict(list)

    for meta in metas:
        by_filing[meta.filing_node].append(meta)
        by_company_section[(meta.ticker, meta.form_type, meta.section_family)].append(meta)
        by_company_form[(meta.ticker, meta.form_type)].append(meta)

    for group in by_filing.values():
        ordered = _ordered_unique_metas(group, lambda meta: meta.doc_id)
        for previous, current in zip(ordered, ordered[1:]):
            add(previous.doc_id, "has_next_document_in_filing", current.doc_id, previous)
            add(current.doc_id, "has_previous_document_in_filing", previous.doc_id, current)

    for group in by_company_section.values():
        ordered = _ordered_unique_metas(group, lambda meta: meta.doc_id)
        for previous, current in zip(ordered, ordered[1:]):
            if previous.doc_id == current.doc_id:
                continue
            add(previous.doc_id, "has_next_same_company_section_doc", current.doc_id, previous)
            add(current.doc_id, "has_previous_same_company_section_doc", previous.doc_id, current)

    for group in by_company_form.values():
        ordered = _ordered_unique_metas(group, lambda meta: meta.filing_node)
        for previous, current in zip(ordered, ordered[1:]):
            if previous.filing_node == current.filing_node:
                continue
            add(previous.filing_node, "has_next_same_company_form_filing", current.filing_node, previous)
            add(current.filing_node, "has_previous_same_company_form_filing", previous.filing_node, current)

    return added


def build_store(
    *,
    corpus_path: Path,
    store_dir: Path,
    namespace: str,
    limit: int | None,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    reset: bool,
    summarize: bool,
    extract_text_facts: bool,
    deep_relationships: bool,
    llm: LLM | None = None,
    embedder: Embedder | None = None,
    llm_summaries: bool = False,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    seed_everything(seed)
    if isinstance(llm, OllamaLLM):
        llm.seed = seed
    corpus_path = corpus_path.resolve()
    store_dir = store_dir.resolve()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    if reset and store_dir.exists():
        shutil.rmtree(store_dir)

    mem = open_memory(
        store_dir,
        llm=llm,
        embedder=embedder,
        fin_rate_prompts=llm is not None,
    )
    seen_facts = _existing_fact_keys(mem)
    started_at = _utc_now_iso()
    start = time.perf_counter()

    documents_seen = 0
    chunks_added = 0
    metadata_facts_added = 0
    deep_relationship_facts_added = 0
    new_episodes = []
    metas: list[FilingMeta] = []
    doc_provenance: dict[str, list[str]] = {}

    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            if limit is not None and documents_seen >= limit:
                break

            record = json.loads(line)
            meta = parse_filing_meta(record, line_number)
            text = _record_text(record)
            documents_seen += 1
            metas.append(meta)
            doc_episode_ids: list[str] = []

            for chunk_index, (chunk_text, token_start, token_end) in enumerate(
                _token_windows(
                    text,
                    max_tokens=chunk_size_tokens,
                    overlap_tokens=chunk_overlap_tokens,
                )
            ):
                content = f"{meta.title}\n\n{chunk_text}" if meta.title else chunk_text
                ep = mem.add(
                    content,
                    user_id=namespace,
                    session_id=meta.filing_node,
                    speaker="filing",
                    event_time=meta.event_time,
                )
                ep.metadata.update(
                    {
                        "doc_id": meta.doc_id,
                        "title": meta.title,
                        "ticker": meta.ticker,
                        "filing_date": meta.filing_date,
                        "form_type": meta.form_type,
                        "section": meta.section,
                        "section_family": meta.section_family,
                        "comparison_group": meta.comparison_group,
                        "chunk_index": chunk_index,
                        "token_start": token_start,
                        "token_end": token_end,
                    }
                )
                if not extract_text_facts:
                    ep.consolidated = True
                new_episodes.append(ep)
                doc_episode_ids.append(ep.id)
                chunks_added += 1

            doc_provenance[meta.doc_id] = doc_episode_ids
            provenance = doc_episode_ids[:1]
            metadata_edges = [
                (meta.ticker, "has_filing", meta.filing_node),
                (meta.filing_node, "has_form_type", meta.form_type),
                (meta.filing_node, "has_filing_date", meta.filing_date),
                (meta.filing_node, "has_year", meta.year),
                (meta.filing_node, "has_company", meta.ticker),
                (meta.filing_node, "has_section", meta.section_node),
                (meta.section_node, "has_section_family", meta.section_family),
                (meta.section_node, "has_document", meta.doc_id),
            ]
            for subject, predicate, object_ in metadata_edges:
                if add_metadata_fact(
                    mem,
                    seen_facts,
                    user_id=namespace,
                    subject=subject,
                    predicate=predicate,
                    object_=object_,
                    valid_at=meta.event_time,
                    provenance=provenance,
                ):
                    metadata_facts_added += 1

            if documents_seen % 500 == 0:
                print(
                    f"indexed {documents_seen} docs / {chunks_added} chunks / "
                    f"{metadata_facts_added} metadata facts",
                    file=sys.stderr,
                )

    if deep_relationships and metas:
        deep_relationship_facts_added = add_deep_relationships(
            mem,
            seen_facts,
            metas=metas,
            doc_provenance=doc_provenance,
            namespace=namespace,
        )

    extracted_facts = 0
    if extract_text_facts and new_episodes:
        extracted_facts = mem.consolidate(new_episodes).get("facts_added", 0)

    summary_llm = getattr(mem.summarizer, "llm", None)
    if not llm_summaries:
        mem.summarizer.llm = None
    summaries = mem.summarize_episodes(new_episodes) if summarize else 0
    mem.summarizer.llm = summary_llm
    mem.save(str(store_dir))

    stats = {
        "corpus_path": str(corpus_path),
        "store_dir": str(store_dir),
        "namespace": namespace,
        "documents_seen": documents_seen,
        "chunks_added": chunks_added,
        "metadata_facts_added": metadata_facts_added,
        "deep_relationship_facts_added": deep_relationship_facts_added,
        "text_facts_extracted": extracted_facts,
        "summaries_added": summaries,
        "llm_configured": llm is not None,
        "llm_summaries": bool(llm is not None and llm_summaries),
        "seed": seed,
        "embedder": getattr(embedder, "model_name", embedder.__class__.__name__ if embedder else "hashing"),
        "graph_nodes": len(mem.graph.entities),
        "graph_edges": len(mem.graph.relations()),
        "build_started_at_utc": started_at,
        "build_finished_at_utc": _utc_now_iso(),
        "build_seconds": time.perf_counter() - start,
        "local_only": True,
    }
    (store_dir / "fin_rate_build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def query_store(
    *,
    store_dir: Path,
    namespace: str,
    query: str,
    top_k: int,
    context_chars: int,
    answer_context_chars: int,
    include_profile_context: bool,
    llm: LLM | None = None,
    embedder: Embedder | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    # Keep query retrieval deterministic and fast. The LLM is used only after
    # retrieval to answer over the assembled context; passing it into Memory here
    # would also trigger profile/persona synthesis, which is not useful for SEC filings.
    seed_everything(seed)
    if isinstance(llm, OllamaLLM):
        llm.seed = seed
    mem = open_memory(store_dir, embedder=embedder)
    context = mem.lean_context(query, user_id=namespace, n_chunks=top_k)
    if not include_profile_context:
        context = strip_user_profile_block(context)
    result = mem.search(query, user_id=namespace)
    context_out = context if context_chars <= 0 else context[:context_chars]
    truncated = context_chars > 0 and len(context) > context_chars
    payload = {
        "query": query,
        "answer": result.answer(),
        "via": result.via,
        "note": "Use `llm_answer` when an Ollama model is supplied; `answer` is Engram's top fact value.",
        "facts": [
            {
                "text": fact.text,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
            }
            for fact in result.facts[:top_k]
        ],
        "context_chars": len(context),
        "context_truncated": truncated,
        "context": context_out,
    }
    if llm is not None:
        answer_context = context if answer_context_chars <= 0 else context[:answer_context_chars]
        payload["answer_context_chars"] = len(answer_context)
        payload["answer_context_truncated"] = len(answer_context) < len(context)
        answer_prompt = f"Question:\n{query}\n\nEngram retrieval context:\n{answer_context}\n\nAnswer:"
        payload["llm_answer"] = llm.complete(
            answer_prompt,
            system=FIN_RATE_ANSWER_SYSTEM,
            num_predict=512,
        )
    return payload


def graph_store(
    *,
    store_dir: Path,
    namespace: str,
    graph_query: str,
    include_sensitive: bool,
    live_only: bool,
    limit: int | None,
    embedder: Embedder | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    seed_everything(seed)
    mem = open_memory(store_dir, embedder=embedder)
    return mem.graph_data(
        user_id=namespace,
        q=graph_query,
        include_sensitive=include_sensitive,
        live_only=live_only,
        limit=limit,
    )


def make_ollama_llm(args: argparse.Namespace) -> OllamaLLM | None:
    if not args.ollama_model:
        return None
    return OllamaLLM(
        args.ollama_model,
        base_url=args.ollama_url,
        timeout=args.ollama_timeout,
        num_ctx=args.ollama_num_ctx,
        num_predict=args.ollama_num_predict,
        seed=args.seed,
    )


def make_ollama_embedder(args: argparse.Namespace) -> OllamaEmbedder | None:
    if not args.ollama_embed_model:
        return None
    return OllamaEmbedder(
        args.ollama_embed_model,
        base_url=args.ollama_url,
        timeout=args.ollama_timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/query a fully local Engram store from Fin-RATE corpus.jsonl."
    )
    parser.add_argument("--build", action="store_true", help="Build or extend the local Engram store.")
    parser.add_argument("--query", help="Retrieve an Engram lean context for a question.")
    parser.add_argument("--graph-query", default="", help="Filter graph nodes/edges by text.")
    parser.add_argument("--export-graph", type=Path, help="Write graph JSON to this path.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Engram service data root. Default: Fin-RATE/engram_data.",
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        help="Exact Memory.open store path. Defaults to the service-compatible namespace path.",
    )
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--limit", type=int, help="Only ingest the first N corpus records.")
    parser.add_argument("--chunk-size-tokens", type=int, default=512)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--context-chars",
        type=int,
        default=5000,
        help="Characters of lean context to print for --query. Use 0 for full context.",
    )
    parser.add_argument(
        "--answer-context-chars",
        type=int,
        default=8000,
        help="Characters of retrieved context sent to Ollama for --query. Use 0 for full context.",
    )
    parser.add_argument(
        "--include-profile-context",
        action="store_true",
        help="Keep Engram's USER PROFILE block in query context. Usually noisy for SEC corpora.",
    )
    parser.add_argument("--graph-limit", type=int, default=50, help="Use 0 for the full graph.")
    parser.add_argument("--reset", action="store_true", help="Delete the target Engram store before build.")
    parser.add_argument("--no-summaries", action="store_true", help="Skip offline summary indexing.")
    parser.add_argument(
        "--llm-summaries",
        action="store_true",
        help="Use --ollama-model for build-time summaries. Slow on the full corpus.",
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Skip deeper corpus-derived document relationship edges.",
    )
    parser.add_argument(
        "--extract-text-facts",
        action="store_true",
        help=(
            "Also extract facts from chunk text. With --ollama-model this uses a finance SEC prompt; "
            "without it, offline rules are limited for SEC text."
        ),
    )
    parser.add_argument("--ollama-list", action="store_true", help="List local Ollama models and exit.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", help="Ollama completion model for answers/extraction/summaries.")
    parser.add_argument(
        "--ollama-embed-model",
        help="Optional Ollama embedding model, e.g. nomic-embed-text. Rebuild the store when changing it.",
    )
    parser.add_argument("--ollama-timeout", type=float, default=300.0)
    parser.add_argument("--ollama-workers", type=int, default=1, help="Local LLM extraction/summary workers.")
    parser.add_argument("--ollama-num-ctx", type=int, default=8192)
    parser.add_argument("--ollama-num-predict", type=int, default=768)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--include-sensitive",
        action="store_true",
        help="Include facts marked sensitive in graph output.",
    )
    args = parser.parse_args()
    seed_everything(args.seed)
    store_dir = args.store_dir or _service_namespace_dir(args.data_dir, args.namespace)

    if args.ollama_list:
        client = OllamaClient(args.ollama_url, args.ollama_timeout)
        print(json.dumps({"models": client.models()}, ensure_ascii=False, indent=2))
        return

    if args.ollama_workers <= 0:
        raise ValueError("--ollama-workers must be positive")
    os.environ["ENGRAM_EXTRACT_WORKERS"] = str(args.ollama_workers)
    os.environ["ENGRAM_SUMMARIZE_WORKERS"] = str(args.ollama_workers)
    llm = make_ollama_llm(args)
    embedder = make_ollama_embedder(args)

    if args.build:
        stats = build_store(
            corpus_path=args.corpus,
            store_dir=store_dir,
            namespace=args.namespace,
            limit=args.limit,
            chunk_size_tokens=args.chunk_size_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            reset=args.reset,
            summarize=not args.no_summaries,
            extract_text_facts=args.extract_text_facts,
            deep_relationships=not args.basic_only,
            llm=llm if args.extract_text_facts or args.llm_summaries else None,
            embedder=embedder,
            llm_summaries=args.llm_summaries,
            seed=args.seed,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.query:
        result = query_store(
            store_dir=store_dir,
            namespace=args.namespace,
            query=args.query,
            top_k=args.top_k,
            context_chars=args.context_chars,
            answer_context_chars=args.answer_context_chars,
            include_profile_context=args.include_profile_context,
            llm=llm,
            embedder=embedder,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.graph_query or args.export_graph:
        graph_limit = None if args.graph_limit <= 0 else args.graph_limit
        graph = graph_store(
            store_dir=store_dir,
            namespace=args.namespace,
            graph_query=args.graph_query,
            include_sensitive=args.include_sensitive,
            live_only=True,
            limit=graph_limit,
            embedder=embedder,
            seed=args.seed,
        )
        if args.export_graph:
            args.export_graph.parent.mkdir(parents=True, exist_ok=True)
            args.export_graph.write_text(
                json.dumps(graph, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"wrote graph: {args.export_graph}")
        else:
            print(json.dumps(graph, ensure_ascii=False, indent=2))

    if not (args.build or args.query or args.graph_query or args.export_graph):
        parser.print_help()


if __name__ == "__main__":
    main()
