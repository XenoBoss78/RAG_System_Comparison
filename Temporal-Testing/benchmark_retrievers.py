from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from .reproducibility import DEFAULT_SEED, seed_everything
except ImportError:
    from reproducibility import DEFAULT_SEED, seed_everything


ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT_DIR.parent
FIN_RATE_DIR = WORKSPACE_ROOT / "Fin-RATE"
ENGRAM_DIR = WORKSPACE_ROOT / "engram"

DEFAULT_QA_PATH = FIN_RATE_DIR / "qa" / "LT-QA.json"
DEFAULT_OUTPUT_DIR = FIN_RATE_DIR / "retrieval_benchmarks"
DEFAULT_CHROMA_DB_DIR = FIN_RATE_DIR / "chroma_db"
DEFAULT_CHROMA_METADATA_DB_DIR = FIN_RATE_DIR / "chroma_db"
DEFAULT_CHROMA_COLLECTION = "fin_rate"
DEFAULT_ENGRAM_DATA_DIR = FIN_RATE_DIR / "engram_data"
DEFAULT_ENGRAM_NAMESPACE = "fin-rate"
DEFAULT_ENGRAM_BASE_STORE_DIR = FIN_RATE_DIR / "engram_data_regular_ltqa_subset"
DEFAULT_ENGRAM_BASE_NAMESPACE = "fin-rate-regular-ltqa-subset"
DEFAULT_ENGRAM_TEMPORAL_STORE_DIR = (
    FIN_RATE_DIR / "engram_data_ltqa_subset" / "fin-rate-ltqa-subset--9caea23b52b2bd9b"
)
DEFAULT_ENGRAM_TEMPORAL_NAMESPACE = "fin-rate-ltqa-subset"
DEFAULT_TOP_KS = (5, 10, 15)

DOC_ID_RE = re.compile(r"\bdoc_\d{6}\b")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")


@dataclass
class RetrievedDoc:
    doc_id: str
    score: float | None = None
    source: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResult:
    docs: list[RetrievedDoc]
    latency_ms: float
    error: str = ""


class RetrievalBackend:
    name: str

    def retrieve(self, query: str, *, n_results: int) -> list[RetrievedDoc]:
        raise NotImplementedError


def _utc_now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_qa(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    for key in ("data", "items", "qa", "examples", "qa_pairs"):
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Could not find a QA list in {path}")


def _select_examples(
    examples: list[dict[str, Any]],
    *,
    limit: int | None,
    offset: int,
    shuffle: bool,
    seed: int,
) -> list[dict[str, Any]]:
    selected = examples[offset:] if offset else list(examples)
    if shuffle:
        import random

        rng = random.Random(seed)
        selected = list(selected)
        rng.shuffle(selected)
    return selected[:limit] if limit is not None else selected


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}


def _dedupe_docs(items: Iterable[RetrievedDoc]) -> list[RetrievedDoc]:
    seen: set[str] = set()
    out: list[RetrievedDoc] = []
    for item in items:
        if not item.doc_id or item.doc_id in seen:
            continue
        seen.add(item.doc_id)
        out.append(item)
    return out


def _doc_ids_from_text(*values: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for match in DOC_ID_RE.findall(value or ""):
            if match not in seen:
                seen.add(match)
                out.append(match)
    return out


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class ChromaBackend(RetrievalBackend):
    def __init__(
        self,
        *,
        name: str,
        module_name: str,
        db_dir: Path,
        collection_name: str,
        embedding_backend: str,
        embedding_model: str,
        device: str,
        seed: int,
        pool_multiplier: int,
    ) -> None:
        self.name = name
        self.pool_multiplier = max(1, pool_multiplier)
        self.module = importlib.import_module(module_name)
        self.db_dir = Path(db_dir)
        self.collection_name = collection_name
        self.seed = seed
        self.embedding_function = self.module.create_embedding_function(
            backend=embedding_backend,
            model_name=embedding_model,
            device=device,
            seed=seed,
        )
        self.collection = self.module.get_chroma_collection(
            db_dir=self.db_dir,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )

    def _query_chunks(self, query: str, *, n_results: int) -> list[dict[str, Any]]:
        count = self.collection.count()
        if not query.strip() or count <= 0:
            return []
        pool = min(max(n_results * self.pool_multiplier, n_results), count)
        result = self.collection.query(
            query_texts=[query],
            n_results=pool,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out: list[dict[str, Any]] = []
        for i, chunk_id in enumerate(ids):
            metadata = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            distance = distances[i] if i < len(distances) else None
            score = None if distance is None else 1.0 - float(distance)
            out.append(
                {
                    "id": chunk_id,
                    "text": docs[i] if i < len(docs) else "",
                    "metadata": metadata,
                    "distance": distance,
                    "score": score,
                }
            )
        return out

    def retrieve(self, query: str, *, n_results: int) -> list[RetrievedDoc]:
        chunks = self._query_chunks(query, n_results=n_results)
        docs = [
            RetrievedDoc(
                doc_id=str((chunk.get("metadata") or {}).get("doc_id") or ""),
                score=_safe_float(chunk.get("score")),
                source=self.name,
                detail={
                    "chunk_id": chunk.get("id"),
                    "title": (chunk.get("metadata") or {}).get("title"),
                    "chunk_index": (chunk.get("metadata") or {}).get("chunk_index"),
                },
            )
            for chunk in chunks
        ]
        return _dedupe_docs(docs)[:n_results]


class ChromaMetadataBackend(ChromaBackend):
    def __init__(
        self,
        *,
        metadata_boost: float,
        auto_metadata_filter: bool,
        metadata_filter_mode: str,
        metadata_llm_model: str,
        metadata_temperature: float,
        ollama_url: str,
        strict_metadata_filter: bool,
        corpus_path: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.metadata_boost = metadata_boost
        self.auto_metadata_filter = auto_metadata_filter
        self.metadata_filter_mode = metadata_filter_mode
        self.metadata_llm_model = metadata_llm_model
        self.metadata_temperature = metadata_temperature
        self.ollama_url = ollama_url
        self.strict_metadata_filter = strict_metadata_filter
        self.corpus_path = corpus_path

    def retrieve(self, query: str, *, n_results: int) -> list[RetrievedDoc]:
        q_tokens = _tokens(query)
        count = self.collection.count()
        if not query.strip() or count <= 0:
            return []
        pool = min(max(n_results * self.pool_multiplier, n_results), count)
        chunks = self.module.retrieve_relevant_chunks(
            query,
            db_dir=self.db_dir,
            collection_name=self.collection_name,
            n_results=pool,
            embedding_function=self.embedding_function,
            auto_metadata_filter=self.auto_metadata_filter,
            metadata_filter_mode=self.metadata_filter_mode,
            metadata_llm_model=self.metadata_llm_model,
            metadata_temperature=self.metadata_temperature,
            ollama_url=self.ollama_url,
            strict_metadata_filter=self.strict_metadata_filter,
            corpus_path=self.corpus_path,
            seed=self.seed,
        )
        reranked = []
        for rank, chunk in enumerate(chunks):
            metadata = chunk.get("metadata") or {}
            title = str(metadata.get("title") or "")
            title_tokens = _tokens(title.replace("_", " "))
            overlap = len(q_tokens & title_tokens)
            base = _safe_float(chunk.get("score")) or 0.0
            adjusted = base + self.metadata_boost * overlap
            reranked.append((adjusted, -rank, overlap, chunk))
        reranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        docs = []
        for adjusted, _, overlap, chunk in reranked:
            metadata = chunk.get("metadata") or {}
            docs.append(
                RetrievedDoc(
                    doc_id=str(metadata.get("doc_id") or ""),
                    score=adjusted,
                    source=self.name,
                    detail={
                        "chunk_id": chunk.get("id"),
                        "title": metadata.get("title"),
                        "chunk_index": metadata.get("chunk_index"),
                        "metadata_overlap": overlap,
                        "vector_score": chunk.get("score"),
                        "auto_metadata_filter": self.auto_metadata_filter,
                        "metadata_filter_mode": self.metadata_filter_mode,
                        "strict_metadata_filter": self.strict_metadata_filter,
                    },
                )
            )
        return _dedupe_docs(docs)[:n_results]


class EngramBackend(RetrievalBackend):
    def __init__(
        self,
        *,
        name: str,
        module_name: str,
        store_dir: Path,
        namespace: str,
        mode: str,
        seed: int,
        ollama_embed_model: str | None = None,
        ollama_url: str = "http://127.0.0.1:11434",
    ) -> None:
        self.name = name
        self.namespace = namespace
        self.mode = mode
        self.seed = seed
        sys.path.insert(0, str(ROOT_DIR))
        sys.path.insert(0, str(ENGRAM_DIR))
        self.module = importlib.import_module(module_name)
        embedder = None
        if ollama_embed_model:
            embedder = self.module.OllamaEmbedder(ollama_embed_model, base_url=ollama_url)
        self.mem = self.module.open_memory(store_dir, embedder=embedder)

    def _doc_from_episode(self, ep_id: str) -> str:
        ep = self.mem.episodes_doc.get(ep_id)
        if ep is None:
            return ""
        metadata = getattr(ep, "metadata", {}) or {}
        return str(metadata.get("doc_id") or "")

    def _docs_from_fact(self, fact: Any) -> list[RetrievedDoc]:
        docs: list[RetrievedDoc] = []
        for doc_id in _doc_ids_from_text(
            getattr(fact, "subject", ""),
            getattr(fact, "object", ""),
            getattr(fact, "text", ""),
            getattr(fact, "display", ""),
        ):
            docs.append(
                RetrievedDoc(
                    doc_id=doc_id,
                    score=None,
                    source="engram_fact_text",
                    detail={"fact_id": getattr(fact, "id", ""), "fact": getattr(fact, "text", "")},
                )
            )
        for ep_id in getattr(fact, "provenance", []) or []:
            doc_id = self._doc_from_episode(ep_id)
            if doc_id:
                docs.append(
                    RetrievedDoc(
                        doc_id=doc_id,
                        score=None,
                        source="engram_fact_provenance",
                        detail={"fact_id": getattr(fact, "id", ""), "episode_id": ep_id},
                    )
                )
        return docs

    def retrieve(self, query: str, *, n_results: int) -> list[RetrievedDoc]:
        seed_everything(self.seed)
        out: list[RetrievedDoc] = []
        if self.mode in {"facts", "combined"}:
            result = self.mem.search(query, user_id=self.namespace, top_k=max(n_results, 1))
            for fact in getattr(result, "facts", []) or []:
                out.extend(self._docs_from_fact(fact))
        if self.mode in {"episodes", "combined"}:
            for ep in self.mem.retrieve_episodes(query, self.namespace, k=max(n_results, 1)):
                metadata = getattr(ep, "metadata", {}) or {}
                doc_id = str(metadata.get("doc_id") or "")
                if doc_id:
                    out.append(
                        RetrievedDoc(
                            doc_id=doc_id,
                            score=None,
                            source="engram_episode",
                            detail={"episode_id": getattr(ep, "id", ""), "title": metadata.get("title")},
                        )
                    )
        if self.mode in {"summaries", "combined"}:
            for ep in self.mem.retrieve_summaries(query, self.namespace, k=max(n_results, 1)):
                metadata = getattr(ep, "metadata", {}) or {}
                doc_id = str(metadata.get("doc_id") or "")
                if doc_id:
                    out.append(
                        RetrievedDoc(
                            doc_id=doc_id,
                            score=None,
                            source="engram_summary",
                            detail={"episode_id": getattr(ep, "id", ""), "title": metadata.get("title")},
                        )
                    )
        return _dedupe_docs(out)[:n_results]


class TemporalEngramBackend(RetrievalBackend):
    """Benchmark the temporal query workflow, including metadata filtering and filing summaries."""

    def __init__(
        self,
        *,
        store_dir: Path,
        namespace: str,
        seed: int,
        ollama_embed_model: str | None,
        ollama_url: str,
        llm_query_filters: bool,
        filing_summary_top_k: int,
        evidence_per_filing_summary: int,
    ) -> None:
        self.name = "engram_temporal"
        self.store_dir = store_dir
        self.namespace = namespace
        self.seed = seed
        self.llm_query_filters = llm_query_filters
        self.filing_summary_top_k = filing_summary_top_k
        self.evidence_per_filing_summary = evidence_per_filing_summary
        sys.path.insert(0, str(ROOT_DIR))
        sys.path.insert(0, str(ENGRAM_DIR))
        self.module = importlib.import_module("Engram_Temporal")
        self.embedder = (
            self.module.OllamaEmbedder(ollama_embed_model, base_url=ollama_url)
            if ollama_embed_model
            else None
        )

    def retrieve(self, query: str, *, n_results: int) -> list[RetrievedDoc]:
        seed_everything(self.seed)
        result = self.module.query_store(
            store_dir=self.store_dir,
            namespace=self.namespace,
            query=query,
            top_k=max(n_results, 1),
            context_chars=0,
            answer_context_chars=0,
            include_profile_context=False,
            llm_query_filters=self.llm_query_filters,
            use_filing_summaries=True,
            filing_summary_top_k=self.filing_summary_top_k,
            evidence_per_filing_summary=self.evidence_per_filing_summary,
            llm=None,
            embedder=self.embedder,
            seed=self.seed,
        )
        filters = result.get("filters") or {}
        return _dedupe_docs(
            RetrievedDoc(
                doc_id=str(item.get("doc_id") or ""),
                score=_safe_float(item.get("score")),
                source="engram_temporal",
                detail={
                    "episode_id": item.get("episode_id"),
                    "ticker": item.get("ticker"),
                    "filing_date": item.get("filing_date"),
                    "query_scope": result.get("query_scope"),
                    "retrieval_method": result.get("retrieval_method"),
                    "filters": filters,
                },
            )
            for item in result.get("retrieved_documents") or []
            if isinstance(item, dict)
        )[:n_results]


def _service_namespace_dir(data_dir: Path, namespace: str) -> Path:
    import hashlib

    prefix = re.sub(r"[^A-Za-z0-9]+", "-", namespace).strip("-").lower()[:48]
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:16]
    return data_dir / f"{prefix or 'namespace'}--{digest}"


def _build_backends(args: argparse.Namespace) -> dict[str, RetrievalBackend]:
    requested = [name.strip() for name in args.systems.split(",") if name.strip()]
    backends: dict[str, RetrievalBackend] = {}
    for name in requested:
        if name == "chroma":
            backends[name] = ChromaBackend(
                name="chroma",
                module_name="ChromaSetup",
                db_dir=args.chroma_db_dir,
                collection_name=args.chroma_collection,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                device=args.device,
                seed=args.seed,
                pool_multiplier=args.pool_multiplier,
            )
        elif name == "chroma_metadata":
            backends[name] = ChromaMetadataBackend(
                name="chroma_metadata",
                module_name="ChromaSetupMetaData",
                db_dir=args.chroma_metadata_db_dir,
                collection_name=args.chroma_collection,
                embedding_backend=args.embedding_backend,
                embedding_model=args.embedding_model,
                device=args.device,
                seed=args.seed,
                pool_multiplier=args.pool_multiplier,
                metadata_boost=args.metadata_boost,
                auto_metadata_filter=not args.no_auto_metadata_filter,
                metadata_filter_mode=args.metadata_filter_mode,
                metadata_llm_model=args.metadata_llm_model,
                metadata_temperature=args.metadata_temperature,
                ollama_url=args.ollama_url,
                strict_metadata_filter=args.strict_metadata_filter,
                corpus_path=args.corpus,
            )
        elif name in {"engram", "engram_base"}:
            store_dir = args.engram_base_store_dir or args.engram_store_dir or _service_namespace_dir(
                args.engram_data_dir,
                args.engram_namespace,
            )
            backends[name] = EngramBackend(
                name=name,
                module_name="Engram_Base",
                store_dir=store_dir,
                namespace=args.engram_base_namespace,
                mode=args.engram_mode,
                seed=args.seed,
                ollama_embed_model=args.engram_ollama_embed_model,
                ollama_url=args.ollama_url,
            )
        elif name == "engram_temporal":
            backends[name] = TemporalEngramBackend(
                store_dir=args.engram_temporal_store_dir,
                namespace=args.engram_temporal_namespace,
                seed=args.seed,
                ollama_embed_model=args.engram_ollama_embed_model,
                ollama_url=args.ollama_url,
                llm_query_filters=False,
                filing_summary_top_k=args.temporal_filing_summary_top_k,
                evidence_per_filing_summary=args.temporal_evidence_per_filing_summary,
            )
        else:
            raise ValueError(
                "Unknown system "
                f"{name!r}. Use chroma, chroma_metadata, engram_base, and/or engram_temporal."
            )
    return backends


def _run_backend(
    backend: RetrievalBackend,
    query: str,
    *,
    n_results: int,
) -> BackendResult:
    start = time.perf_counter()
    try:
        docs = backend.retrieve(query, n_results=n_results)
        error = ""
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures and continue.
        docs = []
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - start) * 1000.0
    return BackendResult(docs=docs, latency_ms=latency_ms, error=error)


def _metrics_for_ranking(ranked_doc_ids: list[str], gold_doc_ids: list[str], top_ks: tuple[int, ...]) -> dict[str, Any]:
    gold = set(gold_doc_ids)
    metrics: dict[str, Any] = {
        "gold_count": len(gold),
        "retrieved_count": len(ranked_doc_ids),
    }
    first_hit_rank = 0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold:
            first_hit_rank = rank
            break
    metrics["mrr"] = 0.0 if first_hit_rank == 0 else 1.0 / first_hit_rank
    metrics["first_hit_rank"] = first_hit_rank
    for k in top_ks:
        top = ranked_doc_ids[:k]
        hits = len(set(top) & gold)
        metrics[f"hits@{k}"] = hits
        metrics[f"recall@{k}"] = 0.0 if not gold else hits / len(gold)
        metrics[f"precision@{k}"] = 0.0 if not top else hits / len(top)
        metrics[f"hit@{k}"] = 1.0 if hits else 0.0
    return metrics


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _summarize(rows: list[dict[str, Any]], top_ks: tuple[int, ...]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_system.setdefault(row["system"], []).append(row)
    summary: dict[str, Any] = {}
    for system, items in by_system.items():
        latencies = [float(item["latency_ms"]) for item in items]
        errors = [item for item in items if item.get("error")]
        payload: dict[str, Any] = {
            "queries": len(items),
            "errors": len(errors),
            "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "p95_latency_ms": _percentile(latencies, 0.95),
            "mrr": statistics.fmean(item["metrics"]["mrr"] for item in items) if items else 0.0,
        }
        for k in top_ks:
            payload[f"recall@{k}"] = statistics.fmean(
                item["metrics"][f"recall@{k}"] for item in items
            ) if items else 0.0
            payload[f"hit@{k}"] = statistics.fmean(
                item["metrics"][f"hit@{k}"] for item in items
            ) if items else 0.0
            payload[f"precision@{k}"] = statistics.fmean(
                item["metrics"][f"precision@{k}"] for item in items
            ) if items else 0.0
        summary[system] = payload
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _print_summary(summary: dict[str, Any], top_ks: tuple[int, ...]) -> None:
    cols = ["system", "queries", "errors", "mean_ms", "p95_ms", "mrr"]
    cols.extend(f"recall@{k}" for k in top_ks)
    print("\t".join(cols))
    for system, item in summary.items():
        row = [
            system,
            str(item["queries"]),
            str(item["errors"]),
            f"{item['mean_latency_ms']:.1f}",
            f"{item['p95_latency_ms']:.1f}",
            f"{item['mrr']:.4f}",
        ]
        row.extend(f"{item[f'recall@{k}']:.4f}" for k in top_ks)
        print("\t".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Fin-RATE LT-QA retrieval speed and document-level recall."
    )
    parser.add_argument(
        "--systems",
        default="chroma,chroma_metadata,engram_base,engram_temporal",
        help=(
            "Comma-separated systems: chroma, chroma_metadata, engram_base, "
            "and engram_temporal. The legacy name engram aliases engram_base."
        ),
    )
    parser.add_argument("--qa-file", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--limit", type=int, help="Number of QA examples to evaluate.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--runs", type=int, default=1, help="Repeat the same selected QA set N times.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup queries per backend, not recorded.")
    parser.add_argument("--top-ks", default=",".join(str(k) for k in DEFAULT_TOP_KS))
    parser.add_argument("--n-results", type=int, help="Retrieved unique doc_ids per query. Defaults to max top-k.")
    parser.add_argument("--pool-multiplier", type=int, default=4, help="Chunk pool multiplier before doc dedupe.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", help="Filename prefix. Defaults to a UTC timestamp.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--chroma-db-dir", type=Path, default=DEFAULT_CHROMA_DB_DIR)
    parser.add_argument("--chroma-metadata-db-dir", type=Path, default=DEFAULT_CHROMA_METADATA_DB_DIR)
    parser.add_argument("--chroma-collection", default=DEFAULT_CHROMA_COLLECTION)
    parser.add_argument("--corpus", type=Path, default=FIN_RATE_DIR / "corpus" / "corpus" / "corpus.jsonl")
    parser.add_argument("--embedding-backend", default="default")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--metadata-boost",
        type=float,
        default=0.05,
        help="Post-filter score boost per query token found in Chroma title metadata.",
    )
    parser.add_argument(
        "--no-auto-metadata-filter",
        action="store_true",
        help="Disable ChromaSetupMetaData's company/year candidate filter during benchmarking.",
    )
    parser.add_argument(
        "--metadata-filter-mode",
        choices=("heuristic", "ollama", "none"),
        default="heuristic",
        help="How the metadata backend extracts company/year filters before vector retrieval.",
    )
    parser.add_argument(
        "--metadata-llm-model",
        default="qwen3:4b",
        help="Ollama model used only when --metadata-filter-mode ollama is selected.",
    )
    parser.add_argument(
        "--metadata-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for --metadata-filter-mode ollama.",
    )
    parser.add_argument(
        "--strict-metadata-filter",
        action="store_true",
        help="Do not fall back to unfiltered Chroma retrieval when a company/year filter finds no chunks.",
    )

    parser.add_argument("--engram-data-dir", type=Path, default=DEFAULT_ENGRAM_DATA_DIR)
    parser.add_argument("--engram-store-dir", type=Path)
    parser.add_argument("--engram-namespace", default=DEFAULT_ENGRAM_NAMESPACE)
    parser.add_argument(
        "--engram-base-store-dir",
        type=Path,
        default=DEFAULT_ENGRAM_BASE_STORE_DIR,
        help="Exact regular Engram Base store directory.",
    )
    parser.add_argument(
        "--engram-base-namespace",
        default=DEFAULT_ENGRAM_BASE_NAMESPACE,
        help="Namespace used by the regular Engram Base subset store.",
    )
    parser.add_argument(
        "--engram-temporal-store-dir",
        type=Path,
        default=DEFAULT_ENGRAM_TEMPORAL_STORE_DIR,
        help="Exact temporal Engram store directory.",
    )
    parser.add_argument(
        "--engram-temporal-namespace",
        default=DEFAULT_ENGRAM_TEMPORAL_NAMESPACE,
        help="Namespace used by the temporal Engram subset store.",
    )
    parser.add_argument(
        "--engram-mode",
        choices=("facts", "episodes", "summaries", "combined"),
        default="combined",
    )
    parser.add_argument("--engram-ollama-embed-model")
    parser.add_argument(
        "--temporal-filing-summary-top-k",
        type=int,
        default=4,
        help="Temporal filing summaries considered for broad queries.",
    )
    parser.add_argument(
        "--temporal-evidence-per-filing-summary",
        type=int,
        default=2,
        help="Temporal raw-evidence chunks expanded from each filing summary.",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.temporal_filing_summary_top_k <= 0:
        raise ValueError("--temporal-filing-summary-top-k must be positive")
    if args.temporal_evidence_per_filing_summary <= 0:
        raise ValueError("--temporal-evidence-per-filing-summary must be positive")
    seed_everything(args.seed)
    top_ks = tuple(sorted({int(k.strip()) for k in args.top_ks.split(",") if k.strip()}))
    if not top_ks or any(k <= 0 for k in top_ks):
        raise ValueError("--top-ks must contain positive integers")
    n_results = args.n_results or max(top_ks)
    if n_results < max(top_ks):
        raise ValueError("--n-results must be >= max(--top-ks)")

    examples = _select_examples(
        _load_qa(args.qa_file),
        limit=args.limit,
        offset=args.offset,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    backends = _build_backends(args)

    if args.warmup and examples:
        warmup_examples = examples[: args.warmup]
        for backend in backends.values():
            for example in warmup_examples:
                _run_backend(backend, str(example.get("question", "")), n_results=n_results)

    rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for run_id in range(1, args.runs + 1):
        for index, example in enumerate(examples):
            question = str(example.get("question", ""))
            gold_doc_ids = [str(doc_id) for doc_id in example.get("doc_ids", [])]
            qid = str(example.get("q_id") or example.get("qid") or index)
            for system, backend in backends.items():
                result = _run_backend(backend, question, n_results=n_results)
                ranked_doc_ids = [item.doc_id for item in result.docs]
                rows.append(
                    {
                        "run_id": run_id,
                        "example_index": index,
                        "qid": qid,
                        "system": system,
                        "question": question,
                        "gold_doc_ids": gold_doc_ids,
                        "retrieved_doc_ids": ranked_doc_ids,
                        "retrieved": [
                            {
                                "doc_id": item.doc_id,
                                "score": item.score,
                                "source": item.source,
                                "detail": item.detail,
                            }
                            for item in result.docs
                        ],
                        "latency_ms": result.latency_ms,
                        "error": result.error,
                        "metrics": _metrics_for_ranking(ranked_doc_ids, gold_doc_ids, top_ks),
                    }
                )

    summary = _summarize(rows, top_ks)
    finished_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "qa_file": str(args.qa_file),
        "examples": len(examples),
        "runs": args.runs,
        "systems": list(backends),
        "top_ks": top_ks,
        "n_results": n_results,
        "settings": {
            "pool_multiplier": args.pool_multiplier,
            "embedding_backend": args.embedding_backend,
            "embedding_model": args.embedding_model,
            "device": args.device,
            "metadata_boost": args.metadata_boost,
            "auto_metadata_filter": not args.no_auto_metadata_filter,
            "metadata_filter_mode": args.metadata_filter_mode,
            "metadata_llm_model": args.metadata_llm_model,
            "metadata_temperature": args.metadata_temperature,
            "strict_metadata_filter": args.strict_metadata_filter,
            "corpus_path": str(args.corpus),
            "engram_mode": args.engram_mode,
            "engram_base_store_dir": str(args.engram_base_store_dir),
            "engram_base_namespace": args.engram_base_namespace,
            "engram_temporal_store_dir": str(args.engram_temporal_store_dir),
            "engram_temporal_namespace": args.engram_temporal_namespace,
            "temporal_filing_summary_top_k": args.temporal_filing_summary_top_k,
            "temporal_evidence_per_filing_summary": args.temporal_evidence_per_filing_summary,
        },
        "summary": summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"retrieval_benchmark_{_utc_now_slug()}"
    summary_path = args.output_dir / f"{prefix}_summary.json"
    details_path = args.output_dir / f"{prefix}_details.jsonl"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_jsonl(details_path, rows)
    _print_summary(summary, top_ks)
    print(f"\nsummary: {summary_path}")
    print(f"details: {details_path}")


if __name__ == "__main__":
    main()
