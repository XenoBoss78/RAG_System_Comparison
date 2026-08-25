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
LEGAL_COMPANY_NAME_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&'’.-]*\s+){1,7}"
    r"(?:Inc(?:orporated)?|Corp(?:oration)?|Company|Co|Ltd|Limited|LLC|L\.L\.C\.|PLC)\.?\b"
)


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
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RetrievalBackend:
    name: str

    def retrieve(self, query: str, *, n_results: int) -> list[RetrievedDoc]:
        raise NotImplementedError

    def retrieve_with_plan(
        self,
        query: str,
        *,
        n_results: int,
        subqueries: Iterable[str] | None = None,
    ) -> list[RetrievedDoc]:
        """Retrieve with optional precomputed NSQ branches.

        Existing systems deliberately ignore NSQ branches. Dedicated NSQ
        systems override this method, so their baseline counterparts remain
        byte-for-byte equivalent at the public retrieval interface.
        """
        del subqueries
        return self.retrieve(query, n_results=n_results)

    def answer_context(self, docs: list[RetrievedDoc], *, max_chars: int) -> str:
        """Render the retrieved evidence that an answer model is allowed to use."""
        raise NotImplementedError

    def benchmark_diagnostics(self) -> dict[str, Any]:
        """Return lightweight diagnostics for the most recent retrieval, if any."""
        return {}


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

    # Companion query files deliberately contain only q_id plus a query variant
    # (an expanded question or NSQ subqueries). They remain benchmarkable
    # without copying answers or evidence into generated input data.
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        reference_by_qid: dict[str, dict[str, Any]] = {}
        source_name = str(data.get("source_questions_file") or "").strip()
        if source_name:
            source_path = (path.parent / source_name).resolve()
            if source_path == path.resolve():
                raise ValueError("A question-expansion file cannot reference itself as its source.")
            if not source_path.exists():
                raise FileNotFoundError(
                    "Question-expansion source file was not found: "
                    f"{source_path}"
                )
            reference_by_qid = {
                str(item.get("q_id") or item.get("qid") or ""): item
                for item in _load_qa(source_path)
                if str(item.get("q_id") or item.get("qid") or "")
            }

        companion_examples: list[dict[str, Any]] = []
        for item in data["records"]:
            if not isinstance(item, dict):
                continue
            expanded_question = str(item.get("expanded_question") or "").strip()
            raw_subqueries = item.get("subqueries")
            subqueries = [
                str(value).strip()
                for value in raw_subqueries
                if isinstance(value, str) and value.strip()
            ] if isinstance(raw_subqueries, list) else []
            if not expanded_question and not subqueries:
                raise ValueError(
                    "Each companion-query record must contain expanded_question and/or subqueries."
                )
            record = dict(item)
            qid = str(record.get("q_id") or record.get("qid") or "")
            reference = reference_by_qid.get(qid)
            if expanded_question:
                record["question"] = expanded_question
                record["query_variant"] = "expanded_question"
            elif reference is not None:
                record["question"] = str(reference.get("question") or "").strip()
                record["query_variant"] = "nsq_subqueries"
            else:
                raise ValueError(
                    f"NSQ record {qid or '<missing q_id>'} requires a source question file."
                )
            if subqueries:
                record["subqueries"] = subqueries
            if reference is not None:
                # The IDs are read only to compute the same retrieval metrics
                # as the source subset. They never influence the generated
                # query wording or retrieval request.
                record["doc_ids"] = list(reference.get("doc_ids") or [])
            companion_examples.append(record)
        return companion_examples
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


def _normalise_nsq_branches(query: str, subqueries: Iterable[str] | None) -> list[str]:
    """Retain the original question as an anchor and append unique NSQ branches."""
    branches = [query.strip()]
    seen = {query.strip().lower()}
    for value in subqueries or ():
        branch = str(value).strip()
        if not branch or branch.lower() in seen:
            continue
        seen.add(branch.lower())
        branches.append(branch)
    return branches


def _fuse_nsq_documents(
    branches: Iterable[tuple[str, list[RetrievedDoc]]],
    *,
    source: str,
    n_results: int,
    rrf_k: int,
) -> tuple[list[RetrievedDoc], dict[str, Any]]:
    """Fuse parent and NSQ branch rankings with weighted reciprocal-rank fusion."""
    candidates: dict[str, dict[str, Any]] = {}
    branch_diagnostics: list[dict[str, Any]] = []
    for branch_index, (branch, branch_docs) in enumerate(branches):
        # Retaining the parent question prevents a synthetic branch from
        # becoming the only interpretation of the user's request.
        branch_weight = 1.25 if branch_index == 0 else 1.0
        branch_diagnostics.append(
            {
                "branch_index": branch_index,
                "query": branch,
                "returned_documents": len(branch_docs),
            }
        )
        for rank, doc in enumerate(branch_docs, start=1):
            if not doc.doc_id:
                continue
            entry = candidates.get(doc.doc_id)
            if entry is None:
                entry = {
                    "representative": doc,
                    "fusion_score": 0.0,
                    "branches": [],
                    "best_source_score": doc.score,
                }
                candidates[doc.doc_id] = entry
            entry["fusion_score"] += branch_weight / (rrf_k + rank)
            entry["branches"].append({"branch_index": branch_index, "rank": rank})
            current_score = _safe_float(doc.score)
            best_score = _safe_float(entry["best_source_score"])
            if current_score is not None and (best_score is None or current_score > best_score):
                entry["representative"] = doc
                entry["best_source_score"] = doc.score

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -float(item["fusion_score"]),
            -(_safe_float(item["best_source_score"]) or float("-inf")),
            item["representative"].doc_id,
        ),
    )
    fused: list[RetrievedDoc] = []
    for item in ranked[:n_results]:
        representative = item["representative"]
        fused.append(
            RetrievedDoc(
                doc_id=representative.doc_id,
                score=float(item["fusion_score"]),
                source=source,
                detail={
                    **(representative.detail or {}),
                    "nsq": {
                        "branches": item["branches"],
                        "best_source_score": item["best_source_score"],
                        "fusion_method": "weighted_reciprocal_rank_fusion",
                        "rrf_k": rrf_k,
                    },
                },
            )
        )
    return fused, {
        "branches": branch_diagnostics,
        "unique_candidate_documents": len(candidates),
        "fusion_method": "weighted_reciprocal_rank_fusion",
        "rrf_k": rrf_k,
    }


def _assemble_answer_context(blocks: Iterable[str], *, max_chars: int) -> str:
    """Join labelled evidence blocks without exceeding the answer model's prompt budget."""
    if max_chars <= 0:
        return "\n\n---\n\n".join(block for block in blocks if block.strip())
    selected: list[str] = []
    remaining = max_chars
    for block in blocks:
        if not block.strip() or remaining <= 0:
            break
        if len(block) > remaining:
            selected.append(block[:remaining].rsplit(" ", 1)[0].rstrip())
            break
        selected.append(block)
        remaining -= len(block) + 7
    return "\n\n---\n\n".join(selected)


def _evidence_block(rank: int, doc: RetrievedDoc, text: str, *, label: str = "RETRIEVED CHUNK") -> str:
    metadata = doc.detail or {}
    title = str(metadata.get("title") or "")
    header = f"[{rank}] {label} | doc_id={doc.doc_id}"
    if title:
        header += f" | title={title}"
    if doc.score is not None:
        header += f" | score={doc.score:.4f}"
    return f"{header}\n{text.strip()}"


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

    def answer_context(self, docs: list[RetrievedDoc], *, max_chars: int) -> str:
        chunk_ids = [str(doc.detail.get("chunk_id") or "") for doc in docs]
        chunk_ids = [chunk_id for chunk_id in chunk_ids if chunk_id]
        if not chunk_ids:
            return ""
        stored = self.collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        stored_ids = stored.get("ids") or []
        stored_docs = stored.get("documents") or []
        stored_metadata = stored.get("metadatas") or []
        by_chunk_id = {
            str(chunk_id): (
                str(stored_docs[index] or "") if index < len(stored_docs) else "",
                stored_metadata[index]
                if index < len(stored_metadata) and isinstance(stored_metadata[index], dict)
                else {},
            )
            for index, chunk_id in enumerate(stored_ids)
        }
        blocks: list[str] = []
        for rank, doc in enumerate(docs, start=1):
            chunk_id = str(doc.detail.get("chunk_id") or "")
            text, metadata = by_chunk_id.get(chunk_id, ("", {}))
            if not text.strip():
                continue
            context_doc = RetrievedDoc(
                doc_id=doc.doc_id,
                score=doc.score,
                source=doc.source,
                detail={**doc.detail, **metadata},
            )
            blocks.append(_evidence_block(rank, context_doc, text))
        return _assemble_answer_context(blocks, max_chars=max_chars)


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
        verification_enabled: bool = False,
        verification_recovery_results: int = 3,
        verification_max_keypoints: int = 8,
        verification_llm_model: str = "qwen3:4b",
        verification_llm_context_chars: int = 16_000,
        year_branching_enabled: bool = False,
        year_branch_candidates: int = 20,
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
        self.verification_enabled = verification_enabled
        self.verification_recovery_results = verification_recovery_results
        self.verification_max_keypoints = verification_max_keypoints
        self.verification_llm_model = verification_llm_model
        self.verification_llm_context_chars = verification_llm_context_chars
        self.year_branching_enabled = year_branching_enabled
        self.year_branch_candidates = year_branch_candidates
        self._last_diagnostics: dict[str, Any] = {}

    def retrieve(self, query: str, *, n_results: int) -> list[RetrievedDoc]:
        self._last_diagnostics = {}
        q_tokens = _tokens(query)
        count = self.collection.count()
        if not query.strip() or count <= 0:
            return []
        pool = min(max(n_results * self.pool_multiplier, n_results), count)
        # The verified module owns final top-k selection for the final-context
        # verifier and for per-year branching. Regular Chroma Metadata keeps
        # the benchmark's historical expanded-pool behavior before doc dedupe.
        module_n_results = (
            n_results
            if self.verification_enabled or self.year_branching_enabled
            else pool
        )
        retrieval_kwargs: dict[str, Any] = {
            "db_dir": self.db_dir,
            "collection_name": self.collection_name,
            "n_results": module_n_results,
            "embedding_function": self.embedding_function,
            "auto_metadata_filter": self.auto_metadata_filter,
            "metadata_filter_mode": self.metadata_filter_mode,
            "metadata_llm_model": self.metadata_llm_model,
            "metadata_temperature": self.metadata_temperature,
            "ollama_url": self.ollama_url,
            "strict_metadata_filter": self.strict_metadata_filter,
            "corpus_path": self.corpus_path,
            "seed": self.seed,
        }
        if self.verification_enabled:
            # Preserve the same candidate depth as regular Chroma Metadata,
            # while requiring the module to choose and verify the actual final
            # top-k chunks rather than the whole candidate pool.
            retrieval_kwargs.update(
                {
                    "verification_enabled": True,
                    "verification_pool_multiplier": self.pool_multiplier,
                    "verification_recovery_results": self.verification_recovery_results,
                    "verification_max_keypoints": self.verification_max_keypoints,
                    "verification_llm_model": self.verification_llm_model,
                    "verification_llm_context_chars": self.verification_llm_context_chars,
                }
            )
        if self.year_branching_enabled:
            retrieval_kwargs.update(
                {
                    "year_branching_enabled": True,
                    "year_branch_candidates": self.year_branch_candidates,
                }
            )
        chunks = self.module.retrieve_relevant_chunks(
            query,
            **retrieval_kwargs,
        )
        verification = chunks[0].get("verification") if chunks else None
        self._last_diagnostics = (
            {"retrieval_verification": verification} if isinstance(verification, dict) else {}
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
                        "verification_enabled": self.verification_enabled,
                        "year_branching_enabled": self.year_branching_enabled,
                        "year_branch_candidates": self.year_branch_candidates,
                    },
                )
            )
        return _dedupe_docs(docs)[:n_results]

    def benchmark_diagnostics(self) -> dict[str, Any]:
        return self._last_diagnostics


class ChromaMetadataNSQBackend(ChromaMetadataBackend):
    """Precomputed NSQ branches over the unchanged Chroma Metadata database."""

    def __init__(self, *, nsq_branch_candidates: int, nsq_rrf_k: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = "chroma_metadata_nsq"
        self.nsq_branch_candidates = nsq_branch_candidates
        self.nsq_rrf_k = nsq_rrf_k

    def retrieve_with_plan(
        self,
        query: str,
        *,
        n_results: int,
        subqueries: Iterable[str] | None = None,
    ) -> list[RetrievedDoc]:
        branches = _normalise_nsq_branches(query, subqueries)
        per_branch = max(n_results, self.nsq_branch_candidates)
        branch_results: list[tuple[str, list[RetrievedDoc]]] = []
        branch_backend_diagnostics: list[dict[str, Any]] = []
        for branch in branches:
            docs = super().retrieve(branch, n_results=per_branch)
            branch_results.append((branch, docs))
            branch_backend_diagnostics.append(dict(self._last_diagnostics))
        fused, diagnostics = _fuse_nsq_documents(
            branch_results,
            source=self.name,
            n_results=n_results,
            rrf_k=self.nsq_rrf_k,
        )
        self._last_diagnostics = {
            "nsq": {
                **diagnostics,
                "precomputed_subqueries_used": len(branches) - 1,
                "per_branch_candidates": per_branch,
            },
            "branch_retrieval_diagnostics": branch_backend_diagnostics,
        }
        return fused


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

    def answer_context(self, docs: list[RetrievedDoc], *, max_chars: int) -> str:
        blocks: list[str] = []
        for rank, doc in enumerate(docs, start=1):
            detail = doc.detail or {}
            episode_id = str(detail.get("episode_id") or "")
            episode = self.mem.episodes_doc.get(episode_id) if episode_id else None
            if episode is not None:
                text = str(getattr(episode, "content", "") or "")
                if text.strip():
                    metadata = getattr(episode, "metadata", {}) or {}
                    context_doc = RetrievedDoc(
                        doc_id=doc.doc_id,
                        score=doc.score,
                        source=doc.source,
                        detail={**detail, **metadata},
                    )
                    blocks.append(_evidence_block(rank, context_doc, text))
                    continue
            fact_text = str(detail.get("fact") or "")
            if fact_text:
                blocks.append(_evidence_block(rank, doc, fact_text, label="RETRIEVED GRAPH FACT"))
        return _assemble_answer_context(blocks, max_chars=max_chars)


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
        llm_query_time_company_resolver: bool,
        llm_model: str,
        llm_temperature: float,
        filing_summary_top_k: int,
        evidence_per_filing_summary: int,
    ) -> None:
        self.name = "engram_temporal"
        self.store_dir = store_dir
        self.namespace = namespace
        self.seed = seed
        self.llm_query_time_company_resolver = llm_query_time_company_resolver
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
        self.resolver_llm = (
            self.module.OllamaLLM(
                llm_model,
                base_url=ollama_url,
                temperature=llm_temperature,
                seed=seed,
            )
            if llm_query_time_company_resolver
            else None
        )
        self._context_memory: Any | None = None

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
            llm_query_filters=False,
            llm_query_time_company_resolver=self.llm_query_time_company_resolver,
            use_filing_summaries=True,
            filing_summary_top_k=self.filing_summary_top_k,
            evidence_per_filing_summary=self.evidence_per_filing_summary,
            llm=self.resolver_llm,
            embedder=self.embedder,
            generate_answer=False,
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

    def answer_context(self, docs: list[RetrievedDoc], *, max_chars: int) -> str:
        if self._context_memory is None:
            self._context_memory = self.module.open_memory(self.store_dir, embedder=self.embedder)
        blocks: list[str] = []
        for rank, doc in enumerate(docs, start=1):
            episode_id = str(doc.detail.get("episode_id") or "")
            episode = self._context_memory.episodes_doc.get(episode_id) if episode_id else None
            if episode is None:
                continue
            text = str(getattr(episode, "content", "") or "")
            if not text.strip():
                continue
            metadata = getattr(episode, "metadata", {}) or {}
            context_doc = RetrievedDoc(
                doc_id=doc.doc_id,
                score=doc.score,
                source=doc.source,
                detail={**doc.detail, **metadata},
            )
            blocks.append(_evidence_block(rank, context_doc, text))
        return _assemble_answer_context(blocks, max_chars=max_chars)


class TemporalEngramNSQBackend(TemporalEngramBackend):
    """Precomputed NSQ branches over the unchanged Temporal Engram store."""

    def __init__(
        self,
        *,
        nsq_branch_candidates: int,
        nsq_rrf_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.name = "engram_temporal_nsq"
        self.nsq_branch_candidates = nsq_branch_candidates
        self.nsq_rrf_k = nsq_rrf_k

    def _parent_filters(
        self,
        query: str,
        branches: Iterable[str],
    ) -> Any:
        """Resolve parent constraints once, then reuse them for all NSQ branches."""
        if self._context_memory is None:
            self._context_memory = self.module.open_memory(self.store_dir, embedder=self.embedder)
        episodes = list(self._context_memory.episodes_doc.values())
        filters = self.module.extract_query_filters(
            query,
            episodes,
            self.namespace,
            llm=self.resolver_llm,
            llm_query_filters=False,
            llm_query_time_company_resolver=True,
        )
        if filters.company_tickers:
            return filters

        # The precomputed branches explicitly repeat the legal company name.
        # When the model returns an empty list, validate those names directly
        # against the existing store instead of guessing a ticker.
        legal_names: list[str] = []
        for text in branches:
            for match in LEGAL_COMPANY_NAME_RE.finditer(text):
                candidate = match.group(0).rstrip(". ")
                if candidate and candidate not in legal_names:
                    legal_names.append(candidate)
        resolution = self.module.resolve_llm_company_names_in_existing_chunks(
            legal_names,
            episodes,
            self.namespace,
        )
        if resolution.company_tickers:
            return self.module.QueryFilters(
                company_tickers=resolution.company_tickers,
                company_names=resolution.company_names,
                years=filters.years,
                source=f"{filters.source}+nsq_legal_name_content_fallback",
                start_date=filters.start_date,
                end_date=filters.end_date,
                relative_time_expression=filters.relative_time_expression,
                relative_time_reference=filters.relative_time_reference,
                company_resolution="nsq_legal_name_content_exact_fallback",
                company_aliases_matched=resolution.matched_company_names,
            )
        return filters

    def retrieve_with_plan(
        self,
        query: str,
        *,
        n_results: int,
        subqueries: Iterable[str] | None = None,
    ) -> list[RetrievedDoc]:
        seed_everything(self.seed)
        branches = _normalise_nsq_branches(query, subqueries)
        parent_filters = self._parent_filters(query, branches)
        per_branch = max(n_results, self.nsq_branch_candidates)
        branch_results: list[tuple[str, list[RetrievedDoc]]] = []
        branch_diagnostics: list[dict[str, Any]] = []

        for branch_index, branch in enumerate(branches):
            branch_years = tuple(sorted(set(self.module.YEAR_RE.findall(branch))))
            branch_filters = self.module.QueryFilters(
                company_tickers=parent_filters.company_tickers,
                company_names=parent_filters.company_names,
                years=branch_years or parent_filters.years,
                source=f"{parent_filters.source}+nsq_branch_{branch_index}",
                start_date=parent_filters.start_date if not branch_years else "",
                end_date=parent_filters.end_date if not branch_years else "",
                relative_time_expression=parent_filters.relative_time_expression if not branch_years else "",
                relative_time_reference=parent_filters.relative_time_reference if not branch_years else "",
                company_resolution=parent_filters.company_resolution,
                company_aliases_matched=parent_filters.company_aliases_matched,
            )
            metadata_scope = "company_and_time"
            raw_candidates = self.module.filter_episodes_by_query_metadata(
                self._context_memory.episodes_doc.values(),
                self.namespace,
                branch_filters,
                artifact_type="filing_chunk",
            )
            if not raw_candidates and branch_filters.company_tickers and branch_filters.years:
                # Some legacy filings represent a requested reporting year only
                # in their text and later filing date. Preserve the validated
                # company constraint and relax time only after the exact branch
                # filter is empty; this is never applied to ordinary hits.
                branch_filters = self.module.QueryFilters(
                    company_tickers=branch_filters.company_tickers,
                    company_names=branch_filters.company_names,
                    years=(),
                    source=f"{branch_filters.source}+nsq_company_only_temporal_fallback",
                    company_resolution=branch_filters.company_resolution,
                    company_aliases_matched=branch_filters.company_aliases_matched,
                )
                metadata_scope = "company_only_after_empty_time_filter"
            result = self.module.query_store(
                store_dir=self.store_dir,
                namespace=self.namespace,
                query=branch,
                top_k=per_branch,
                context_chars=0,
                answer_context_chars=0,
                include_profile_context=False,
                query_timings_path=self.store_dir / "nsq_query_timings.jsonl",
                query_filters_override=branch_filters,
                llm_query_filters=False,
                llm_query_time_company_resolver=False,
                use_filing_summaries=True,
                filing_summary_top_k=self.filing_summary_top_k,
                evidence_per_filing_summary=self.evidence_per_filing_summary,
                llm=None,
                embedder=self.embedder,
                generate_answer=False,
                seed=self.seed,
            )
            filters = result.get("filters") or {}
            docs = _dedupe_docs(
                RetrievedDoc(
                    doc_id=str(item.get("doc_id") or ""),
                    score=_safe_float(item.get("score")),
                    source=self.name,
                    detail={
                        "episode_id": item.get("episode_id"),
                        "ticker": item.get("ticker"),
                        "filing_date": item.get("filing_date"),
                        "query_scope": result.get("query_scope"),
                        "retrieval_method": result.get("retrieval_method"),
                        "filters": filters,
                        "nsq_branch_index": branch_index,
                    },
                )
                for item in result.get("retrieved_documents") or []
                if isinstance(item, dict)
            )[:per_branch]
            branch_results.append((branch, docs))
            branch_diagnostics.append(
                {
                    "branch_index": branch_index,
                    "filters": filters,
                    "metadata_scope": metadata_scope,
                    "documents_after_filter": result.get("documents_after_filter"),
                    "documents_retrieved": len(docs),
                }
            )

        fused, diagnostics = _fuse_nsq_documents(
            branch_results,
            source=self.name,
            n_results=n_results,
            rrf_k=self.nsq_rrf_k,
        )
        self._last_diagnostics = {
            "nsq": {
                **diagnostics,
                "precomputed_subqueries_used": len(branches) - 1,
                "per_branch_candidates": per_branch,
                "parent_filters": parent_filters.as_dict(),
            },
            "branch_retrieval_diagnostics": branch_diagnostics,
        }
        return fused

    def benchmark_diagnostics(self) -> dict[str, Any]:
        return self._last_diagnostics


class TemporalGraphEngramBackend(TemporalEngramBackend):
    """Benchmark the graph-guided temporal query path over the same Engram store."""

    def __init__(
        self,
        *,
        store_dir: Path,
        namespace: str,
        seed: int,
        ollama_embed_model: str | None,
        ollama_url: str,
        llm_model: str,
        llm_temperature: float,
        filing_summary_top_k: int,
        evidence_per_filing_summary: int,
        llm_query_decomposition: bool,
        llm_query_time_company_resolver: bool,
        graph_max_hops: int,
        graph_seed_k: int,
        graph_candidate_k: int,
        graph_max_neighbors: int,
    ) -> None:
        self.name = "engram_temporal_graph"
        self.store_dir = store_dir
        self.namespace = namespace
        self.seed = seed
        self.llm_query_decomposition = llm_query_decomposition
        self.llm_query_time_company_resolver = llm_query_time_company_resolver
        self.filing_summary_top_k = filing_summary_top_k
        self.evidence_per_filing_summary = evidence_per_filing_summary
        self.graph_max_hops = graph_max_hops
        self.graph_seed_k = graph_seed_k
        self.graph_candidate_k = graph_candidate_k
        self.graph_max_neighbors = graph_max_neighbors
        sys.path.insert(0, str(ROOT_DIR))
        sys.path.insert(0, str(ENGRAM_DIR))
        self.module = importlib.import_module("Temporal_Engram_Graph")
        self.embedder = (
            self.module.OllamaEmbedder(ollama_embed_model, base_url=ollama_url)
            if ollama_embed_model
            else None
        )
        # qwen3:4b is intentionally available to both the question-only graph
        # planner and the legal-name company resolver.  It never sees gold data.
        self.graph_llm = self.module.OllamaLLM(
            llm_model,
            base_url=ollama_url,
            temperature=llm_temperature,
            seed=seed,
        )
        self._context_memory: Any | None = None
        self._last_diagnostics: dict[str, Any] = {}

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
            llm_query_filters=False,
            llm_query_time_company_resolver=self.llm_query_time_company_resolver,
            llm_query_decomposition=self.llm_query_decomposition,
            use_filing_summaries=True,
            filing_summary_top_k=self.filing_summary_top_k,
            evidence_per_filing_summary=self.evidence_per_filing_summary,
            graph_max_hops=self.graph_max_hops,
            graph_seed_k=self.graph_seed_k,
            graph_candidate_k=self.graph_candidate_k,
            graph_max_neighbors=self.graph_max_neighbors,
            llm=self.graph_llm,
            embedder=self.embedder,
            generate_answer=False,
            seed=self.seed,
        )
        filters = result.get("filters") or {}
        graph_plan = result.get("graph_query_plan") or {}
        graph_retrieval = result.get("graph_retrieval") or {}
        self._last_diagnostics = {
            "graph_query_plan": graph_plan,
            "graph_retrieval": graph_retrieval,
        }
        return _dedupe_docs(
            RetrievedDoc(
                doc_id=str(item.get("doc_id") or ""),
                score=_safe_float(item.get("score")),
                source="engram_temporal_graph",
                detail={
                    "episode_id": item.get("episode_id"),
                    "ticker": item.get("ticker"),
                    "filing_date": item.get("filing_date"),
                    "retrieval_routes": item.get("retrieval_routes") or [],
                    "graph_paths": item.get("graph_paths") or [],
                    "query_scope": result.get("query_scope"),
                    "retrieval_method": result.get("retrieval_method"),
                    "filters": filters,
                    "graph_query_plan": graph_plan,
                },
            )
            for item in result.get("retrieved_documents") or []
            if isinstance(item, dict)
        )[:n_results]

    def benchmark_diagnostics(self) -> dict[str, Any]:
        return self._last_diagnostics


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
        elif name == "chroma_metadata_nsq":
            backends[name] = ChromaMetadataNSQBackend(
                name=name,
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
                nsq_branch_candidates=args.nsq_branch_candidates,
                nsq_rrf_k=args.nsq_rrf_k,
            )
        elif name in {
            "chroma_metadata",
            "chroma_metadata_multilayer",
            "chroma_metadata_verified",
            "chroma_metadata_yearly",
        }:
            backends[name] = ChromaMetadataBackend(
                name=name,
                module_name=(
                    "ChromaSetupMetaData_modified"
                    if name == "chroma_metadata_multilayer"
                    else (
                        "ChromaSetupMetaData_verified"
                        if name in {"chroma_metadata_verified", "chroma_metadata_yearly"}
                        else "ChromaSetupMetaData"
                    )
                ),
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
                verification_enabled=name == "chroma_metadata_verified",
                verification_recovery_results=args.verification_recovery_results,
                verification_max_keypoints=args.verification_max_keypoints,
                verification_llm_model=args.verification_llm_model,
                verification_llm_context_chars=args.verification_llm_context_chars,
                year_branching_enabled=name == "chroma_metadata_yearly",
                year_branch_candidates=args.year_branch_candidates,
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
                llm_query_time_company_resolver=args.temporal_llm_query_time_company_resolver,
                llm_model=args.temporal_llm_model,
                llm_temperature=args.temporal_llm_temperature,
                filing_summary_top_k=args.temporal_filing_summary_top_k,
                evidence_per_filing_summary=args.temporal_evidence_per_filing_summary,
            )
        elif name == "engram_temporal_nsq":
            backends[name] = TemporalEngramNSQBackend(
                store_dir=args.engram_temporal_store_dir,
                namespace=args.engram_temporal_namespace,
                seed=args.seed,
                ollama_embed_model=args.engram_ollama_embed_model,
                ollama_url=args.ollama_url,
                # Resolve the parent company once with the instruction-tuned
                # model; all precomputed branches reuse that validated result.
                llm_query_time_company_resolver=True,
                llm_model=args.nsq_temporal_company_model,
                llm_temperature=args.temporal_llm_temperature,
                filing_summary_top_k=args.temporal_filing_summary_top_k,
                evidence_per_filing_summary=args.temporal_evidence_per_filing_summary,
                nsq_branch_candidates=args.nsq_branch_candidates,
                nsq_rrf_k=args.nsq_rrf_k,
            )
        elif name == "engram_temporal_graph":
            backends[name] = TemporalGraphEngramBackend(
                store_dir=args.engram_temporal_store_dir,
                namespace=args.engram_temporal_namespace,
                seed=args.seed,
                ollama_embed_model=args.engram_ollama_embed_model,
                ollama_url=args.ollama_url,
                llm_model=args.temporal_graph_llm_model,
                llm_temperature=args.temporal_llm_temperature,
                filing_summary_top_k=args.temporal_filing_summary_top_k,
                evidence_per_filing_summary=args.temporal_evidence_per_filing_summary,
                llm_query_decomposition=not args.temporal_graph_no_llm_query_decomposition,
                llm_query_time_company_resolver=(
                    not args.temporal_graph_no_llm_company_resolver
                ),
                graph_max_hops=args.temporal_graph_max_hops,
                graph_seed_k=args.temporal_graph_seed_k,
                graph_candidate_k=args.temporal_graph_candidate_k,
                graph_max_neighbors=args.temporal_graph_max_neighbors,
            )
        else:
            raise ValueError(
                "Unknown system "
                f"{name!r}. Use chroma, chroma_metadata, chroma_metadata_multilayer, "
                "chroma_metadata_verified, chroma_metadata_yearly, chroma_metadata_nsq, "
                "engram_base, engram_temporal, engram_temporal_nsq, and/or engram_temporal_graph."
            )
    return backends


def _build_answer_comparison_backends(args: argparse.Namespace) -> dict[str, RetrievalBackend]:
    """Build the four fixed retrieval configurations used for answer-level comparison."""
    base_args = argparse.Namespace(**vars(args))
    base_args.systems = "chroma_metadata,engram_base,engram_temporal"
    base_args.temporal_llm_query_time_company_resolver = False
    backends = _build_backends(base_args)
    backends["engram_llm"] = TemporalEngramBackend(
        store_dir=args.engram_temporal_store_dir,
        namespace=args.engram_temporal_namespace,
        seed=args.seed,
        ollama_embed_model=args.engram_ollama_embed_model,
        ollama_url=args.ollama_url,
        llm_query_time_company_resolver=True,
        llm_model=args.temporal_llm_model,
        llm_temperature=args.temporal_llm_temperature,
        filing_summary_top_k=args.temporal_filing_summary_top_k,
        evidence_per_filing_summary=args.temporal_evidence_per_filing_summary,
    )
    return backends


def _run_backend(
    backend: RetrievalBackend,
    query: str,
    *,
    n_results: int,
    subqueries: Iterable[str] | None = None,
) -> BackendResult:
    start = time.perf_counter()
    try:
        docs = backend.retrieve_with_plan(query, n_results=n_results, subqueries=subqueries)
        error = ""
        diagnostics = backend.benchmark_diagnostics()
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures and continue.
        docs = []
        error = f"{type(exc).__name__}: {exc}"
        diagnostics = {}
    latency_ms = (time.perf_counter() - start) * 1000.0
    return BackendResult(
        docs=docs,
        latency_ms=latency_ms,
        error=error,
        diagnostics=diagnostics,
    )


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
        verification_reports = [
            item.get("diagnostics", {}).get("retrieval_verification")
            for item in items
            if isinstance(item.get("diagnostics", {}).get("retrieval_verification"), dict)
        ]
        if verification_reports:
            def _missing_count(report: dict[str, Any], field: str) -> int:
                coverage = report.get("coverage") or {}
                values = coverage.get(field) or []
                return len(values) if isinstance(values, list) else 0

            payload["verification"] = {
                "queries_verified": len(verification_reports),
                "sufficient_context_rate": statistics.fmean(
                    1.0 if report.get("is_sufficient") else 0.0
                    for report in verification_reports
                ),
                "mean_missing_company_tickers": statistics.fmean(
                    _missing_count(report, "company_tickers_missing")
                    for report in verification_reports
                ),
                "mean_missing_years": statistics.fmean(
                    _missing_count(report, "years_missing")
                    for report in verification_reports
                ),
                "mean_missing_keypoints": statistics.fmean(
                    _missing_count(report, "keypoints_missing")
                    for report in verification_reports
                ),
                "mean_recovery_queries": statistics.fmean(
                    len(report.get("recovery_queries") or [])
                    for report in verification_reports
                ),
                "mean_verification_retrieval_seconds": statistics.fmean(
                    float(report.get("retrieval_seconds") or 0.0)
                    for report in verification_reports
                ),
                "llm_verifier_errors": sum(
                    (report.get("llm_keypoint_verification") or {}).get("status") == "error"
                    for report in verification_reports
                ),
                "mean_llm_verification_seconds": statistics.fmean(
                    float(
                        (report.get("llm_keypoint_verification") or {}).get(
                            "verification_seconds"
                        )
                        or 0.0
                    )
                    for report in verification_reports
                ),
            }
        summary[system] = payload
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


ANSWER_COMPARISON_SYSTEM_PROMPT = (
    "Answer the financial QA question using only the retrieved evidence below. "
    "Do not use outside knowledge or assume facts that are absent from the evidence. "
    "If the evidence is insufficient or conflicting, say so clearly. "
    "Cite the supporting retrieved excerpts with their bracketed source labels, such as [1] or [2]."
)


def _answer_prompt(question: str, context: str) -> str:
    return f"Question:\n{question}\n\nRetrieved evidence:\n{context}\n\nAnswer:"


def _serialise_retrieved_docs(docs: Iterable[RetrievedDoc]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": doc.doc_id,
            "score": doc.score,
            "source": doc.source,
            "detail": doc.detail,
        }
        for doc in docs
    ]


def generate_answer_comparison(
    *,
    qa_file: Path,
    output_dir: Path,
    backends: dict[str, RetrievalBackend],
    answer_llm: Any,
    answer_model: str,
    sample_size: int = 20,
    n_results: int = 15,
    context_chars: int = 16_000,
    seed: int = DEFAULT_SEED,
    output_prefix: str | None = None,
) -> dict[str, Path]:
    """Generate comparable grounded answers for a seeded LT-QA sample.

    A JSONL record is flushed after every system answer. The grouped JSON output retains each question's
    golden answer, key points, retrieved evidence, and all four generated answers for later review.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if n_results <= 0:
        raise ValueError("n_results must be positive")
    if context_chars <= 0:
        raise ValueError("context_chars must be positive")
    required_systems = ("chroma_metadata", "engram_base", "engram_temporal", "engram_llm")
    missing = [system for system in required_systems if system not in backends]
    if missing:
        raise ValueError(f"Answer comparison is missing retrieval systems: {', '.join(missing)}")

    selected = _select_examples(
        _load_qa(qa_file),
        limit=sample_size,
        offset=0,
        shuffle=True,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or f"ltqa_answer_comparison_{_utc_now_slug()}"
    answers_path = output_dir / f"{prefix}_answers.jsonl"
    comparison_path = output_dir / f"{prefix}_comparison.json"
    started_at = datetime.now(timezone.utc).isoformat()
    questions: list[dict[str, Any]] = []

    with answers_path.open("w", encoding="utf-8") as answers_file:
        for sample_index, example in enumerate(selected, start=1):
            question = str(example.get("question") or "").strip()
            qid = str(example.get("q_id") or example.get("qid") or sample_index)
            question_record: dict[str, Any] = {
                "sample_index": sample_index,
                "qid": qid,
                "question": question,
                "gold_answer": str(example.get("answer") or ""),
                "gold_key_points": example.get("key_points") or [],
                "gold_doc_ids": [str(doc_id) for doc_id in example.get("doc_ids") or []],
                "systems": {},
            }
            for system in required_systems:
                print(
                    f"[Answer comparison] question {sample_index}/{len(selected)} | {system}",
                    flush=True,
                )
                retrieval = _run_backend(backends[system], question, n_results=n_results)
                context = ""
                context_error = ""
                if not retrieval.error:
                    try:
                        context = backends[system].answer_context(
                            retrieval.docs,
                            max_chars=context_chars,
                        )
                    except Exception as exc:  # noqa: BLE001 -- save the retrieval even if context rendering fails
                        context_error = f"{type(exc).__name__}: {exc}"
                answer = ""
                answer_error = ""
                answer_latency_ms = 0.0
                if retrieval.error:
                    answer_error = "Answer skipped because retrieval failed."
                elif context_error:
                    answer_error = "Answer skipped because evidence rendering failed."
                elif not context.strip():
                    answer_error = "Answer skipped because this system returned no readable evidence."
                else:
                    answer_started = time.perf_counter()
                    try:
                        answer = str(
                            answer_llm.complete(
                                _answer_prompt(question, context),
                                system=ANSWER_COMPARISON_SYSTEM_PROMPT,
                                num_predict=768,
                            )
                        ).strip()
                    except Exception as exc:  # noqa: BLE001 -- a failed generation must not discard prior systems
                        answer_error = f"{type(exc).__name__}: {exc}"
                    answer_latency_ms = (time.perf_counter() - answer_started) * 1000.0

                system_record = {
                    "answer_model": answer_model,
                    "retrieval_latency_ms": round(retrieval.latency_ms, 3),
                    "answer_generation_latency_ms": round(answer_latency_ms, 3),
                    "retrieval_error": retrieval.error,
                    "context_error": context_error,
                    "answer_error": answer_error,
                    "retrieved_doc_ids": [doc.doc_id for doc in retrieval.docs],
                    "retrieved": _serialise_retrieved_docs(retrieval.docs),
                    "context": context,
                    "context_chars": len(context),
                    "generated_answer": answer,
                }
                question_record["systems"][system] = system_record
                answers_file.write(
                    json.dumps(
                        {
                            "sample_index": sample_index,
                            "qid": qid,
                            "question": question,
                            "gold_answer": question_record["gold_answer"],
                            "gold_key_points": question_record["gold_key_points"],
                            "gold_doc_ids": question_record["gold_doc_ids"],
                            "system": system,
                            **system_record,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                answers_file.flush()
            questions.append(question_record)

    payload = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "qa_file": str(qa_file),
        "sample_size": len(selected),
        "sample_seed": seed,
        "n_results": n_results,
        "context_chars_per_system": context_chars,
        "answer_model": answer_model,
        "systems": list(required_systems),
        "answer_system_prompt": ANSWER_COMPARISON_SYSTEM_PROMPT,
        "answers_jsonl": str(answers_path),
        "questions": questions,
    }
    comparison_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"answers_jsonl": answers_path, "comparison_json": comparison_path}


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
            "Comma-separated systems: chroma, chroma_metadata, chroma_metadata_multilayer, "
            "chroma_metadata_verified, chroma_metadata_yearly, chroma_metadata_nsq, engram_base, "
            "engram_temporal, engram_temporal_nsq, and engram_temporal_graph. "
            "The legacy name "
            "engram aliases engram_base."
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
    parser.add_argument(
        "--answer-comparison",
        action="store_true",
        help=(
            "Sample LT-QA questions, retrieve from Chroma Metadata, Engram Base, original Temporal, "
            "and LLM Temporal, then save one grounded qwen answer per system."
        ),
    )
    parser.add_argument(
        "--answer-sample-size",
        type=int,
        default=20,
        help="Seeded random LT-QA sample size used with --answer-comparison.",
    )
    parser.add_argument(
        "--answer-context-chars",
        type=int,
        default=16_000,
        help="Maximum retrieved-context characters provided to qwen for each generated answer.",
    )
    parser.add_argument(
        "--answer-model",
        default="qwen3:4b",
        help="Ollama model that writes the grounded answers with --answer-comparison.",
    )
    parser.add_argument(
        "--answer-output-prefix",
        help="Prefix for --answer-comparison JSON and JSONL output files.",
    )

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
    parser.add_argument(
        "--verification-recovery-results",
        type=int,
        default=3,
        help=(
            "Chunks requested by each targeted recovery search in "
            "chroma_metadata_verified."
        ),
    )
    parser.add_argument(
        "--verification-max-keypoints",
        type=int,
        default=8,
        help=(
            "Maximum meaningful query terms required by the verification layer in "
            "chroma_metadata_verified."
        ),
    )
    parser.add_argument(
        "--verification-llm-model",
        default="qwen3:4b",
        help=(
            "Ollama model that semantically audits key concepts in the final selected "
            "chunks for chroma_metadata_verified."
        ),
    )
    parser.add_argument(
        "--verification-llm-context-chars",
        type=int,
        default=16_000,
        help="Maximum final-context characters supplied to the verification LLM.",
    )
    parser.add_argument(
        "--year-branch-candidates",
        type=int,
        default=20,
        help=(
            "Candidate chunks retrieved separately for each explicit year by "
            "chroma_metadata_yearly."
        ),
    )
    parser.add_argument(
        "--nsq-branch-candidates",
        type=int,
        default=15,
        help=(
            "Candidate documents retrieved for the original-question anchor and each precomputed "
            "NSQ branch by chroma_metadata_nsq and engram_temporal_nsq."
        ),
    )
    parser.add_argument(
        "--nsq-rrf-k",
        type=int,
        default=60,
        help="Reciprocal-rank fusion constant used to merge NSQ branch rankings.",
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
        "--temporal-llm-query-time-company-resolver",
        action="store_true",
        help=(
            "Use an LLM to extract the company named in each query, then validate its legal name "
            "against Temporal Engram's existing raw chunks before filtering."
        ),
    )
    parser.add_argument(
        "--temporal-llm-model",
        default="qwen3:4b",
        help="Ollama completion model used only with --temporal-llm-query-time-company-resolver.",
    )
    parser.add_argument(
        "--temporal-llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Temporal Engram's query-time company LLM.",
    )
    parser.add_argument(
        "--nsq-temporal-company-model",
        default="qwen3:4b-instruct",
        help=(
            "Ollama model used once per parent question by engram_temporal_nsq to validate the "
            "company constraint before it retrieves the precomputed branches."
        ),
    )
    parser.add_argument(
        "--temporal-graph-llm-model",
        default="qwen3:4b",
        help=(
            "Ollama model used by engram_temporal_graph for question-only graph planning "
            "and company-name resolution."
        ),
    )
    parser.add_argument(
        "--temporal-graph-no-llm-query-decomposition",
        action="store_true",
        help="Use deterministic graph-plan extraction instead of qwen3:4b for engram_temporal_graph.",
    )
    parser.add_argument(
        "--temporal-graph-no-llm-company-resolver",
        action="store_true",
        help="Use metadata aliases instead of qwen3:4b company resolution for engram_temporal_graph.",
    )
    parser.add_argument(
        "--temporal-graph-max-hops",
        type=int,
        default=2,
        help="Maximum structural graph hops from a seed filing chunk for engram_temporal_graph.",
    )
    parser.add_argument(
        "--temporal-graph-seed-k",
        type=int,
        default=12,
        help="Direct vector candidates used as graph traversal anchors per temporal graph query.",
    )
    parser.add_argument(
        "--temporal-graph-candidate-k",
        type=int,
        default=40,
        help="Graph-connected chunks reranked against the original question.",
    )
    parser.add_argument(
        "--temporal-graph-max-neighbors",
        type=int,
        default=80,
        help="Maximum graph edges examined from an expansion node.",
    )
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
    if args.answer_sample_size <= 0:
        raise ValueError("--answer-sample-size must be positive")
    if args.answer_context_chars <= 0:
        raise ValueError("--answer-context-chars must be positive")
    if args.temporal_filing_summary_top_k <= 0:
        raise ValueError("--temporal-filing-summary-top-k must be positive")
    if args.temporal_evidence_per_filing_summary <= 0:
        raise ValueError("--temporal-evidence-per-filing-summary must be positive")
    if args.verification_recovery_results <= 0:
        raise ValueError("--verification-recovery-results must be positive")
    if args.verification_max_keypoints < 0:
        raise ValueError("--verification-max-keypoints cannot be negative")
    if args.verification_llm_context_chars <= 0:
        raise ValueError("--verification-llm-context-chars must be positive")
    if args.year_branch_candidates <= 0:
        raise ValueError("--year-branch-candidates must be positive")
    if args.nsq_branch_candidates <= 0:
        raise ValueError("--nsq-branch-candidates must be positive")
    if args.nsq_rrf_k <= 0:
        raise ValueError("--nsq-rrf-k must be positive")
    if args.temporal_graph_max_hops < 0:
        raise ValueError("--temporal-graph-max-hops cannot be negative")
    if args.temporal_graph_seed_k <= 0:
        raise ValueError("--temporal-graph-seed-k must be positive")
    if args.temporal_graph_candidate_k <= 0:
        raise ValueError("--temporal-graph-candidate-k must be positive")
    if args.temporal_graph_max_neighbors <= 0:
        raise ValueError("--temporal-graph-max-neighbors must be positive")
    seed_everything(args.seed)
    top_ks = tuple(sorted({int(k.strip()) for k in args.top_ks.split(",") if k.strip()}))
    if not top_ks or any(k <= 0 for k in top_ks):
        raise ValueError("--top-ks must contain positive integers")
    n_results = args.n_results or max(top_ks)
    if n_results < max(top_ks):
        raise ValueError("--n-results must be >= max(--top-ks)")

    if args.answer_comparison:
        comparison_backends = _build_answer_comparison_backends(args)
        temporal_module = comparison_backends["engram_llm"].module
        answer_llm = temporal_module.OllamaLLM(
            args.answer_model,
            base_url=args.ollama_url,
            temperature=0.0,
            seed=args.seed,
        )
        paths = generate_answer_comparison(
            qa_file=args.qa_file,
            output_dir=args.output_dir,
            backends=comparison_backends,
            answer_llm=answer_llm,
            answer_model=args.answer_model,
            sample_size=args.answer_sample_size,
            n_results=n_results,
            context_chars=args.answer_context_chars,
            seed=args.seed,
            output_prefix=args.answer_output_prefix,
        )
        print(f"\nanswers JSONL: {paths['answers_jsonl']}")
        print(f"comparison JSON: {paths['comparison_json']}")
        return

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
                _run_backend(
                    backend,
                    str(example.get("question", "")),
                    n_results=n_results,
                    subqueries=example.get("subqueries"),
                )

    rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for run_id in range(1, args.runs + 1):
        for index, example in enumerate(examples):
            question = str(example.get("question", ""))
            subqueries = example.get("subqueries")
            gold_doc_ids = [str(doc_id) for doc_id in example.get("doc_ids", [])]
            qid = str(example.get("q_id") or example.get("qid") or index)
            for system, backend in backends.items():
                result = _run_backend(
                    backend,
                    question,
                    n_results=n_results,
                    subqueries=subqueries,
                )
                ranked_doc_ids = [item.doc_id for item in result.docs]
                rows.append(
                    {
                        "run_id": run_id,
                        "example_index": index,
                        "qid": qid,
                        "system": system,
                        "question": question,
                        "subqueries": subqueries or [],
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
                        "diagnostics": result.diagnostics,
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
            "verification_recovery_results": args.verification_recovery_results,
            "verification_max_keypoints": args.verification_max_keypoints,
            "verification_llm_model": args.verification_llm_model,
            "verification_llm_context_chars": args.verification_llm_context_chars,
            "year_branch_candidates": args.year_branch_candidates,
            "nsq_branch_candidates": args.nsq_branch_candidates,
            "nsq_rrf_k": args.nsq_rrf_k,
            "nsq_temporal_company_model": args.nsq_temporal_company_model,
            "corpus_path": str(args.corpus),
            "engram_mode": args.engram_mode,
            "engram_base_store_dir": str(args.engram_base_store_dir),
            "engram_base_namespace": args.engram_base_namespace,
            "engram_temporal_store_dir": str(args.engram_temporal_store_dir),
            "engram_temporal_namespace": args.engram_temporal_namespace,
            "temporal_llm_query_time_company_resolver": args.temporal_llm_query_time_company_resolver,
            "temporal_llm_model": args.temporal_llm_model,
            "temporal_llm_temperature": args.temporal_llm_temperature,
            "temporal_graph_llm_model": args.temporal_graph_llm_model,
            "temporal_graph_llm_query_decomposition": (
                not args.temporal_graph_no_llm_query_decomposition
            ),
            "temporal_graph_llm_company_resolver": (
                not args.temporal_graph_no_llm_company_resolver
            ),
            "temporal_graph_max_hops": args.temporal_graph_max_hops,
            "temporal_graph_seed_k": args.temporal_graph_seed_k,
            "temporal_graph_candidate_k": args.temporal_graph_candidate_k,
            "temporal_graph_max_neighbors": args.temporal_graph_max_neighbors,
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
