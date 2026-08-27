"""Graph-guided temporal RAG over an existing Temporal Engram store.

This module deliberately shares Temporal Engram's storage and ingestion schema.
It changes only the query path: metadata filters first constrain company and time,
then bounded graph paths generate additional evidence candidates before a common
vector rerank selects the raw chunks supplied to the answer model. If an exact
company/year filter is empty solely because a legacy filing lacks a reporting
period, the query path can conservatively retain the proven company and relax
only that empty time predicate; this is recorded in the timing diagnostics.

The existing Temporal Engram build store can therefore be reused directly.  No
copy of the LT-QA corpus, embedding index, or answer data is created here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .reproducibility import DEFAULT_SEED, seed_everything
except ImportError:
    from reproducibility import DEFAULT_SEED, seed_everything

import Engram_Temporal as temporal


# The graph layer is intentionally a read-path extension.  Re-exporting these
# public building blocks means notebook code can use this module in the same
# way as Engram_Temporal while pointing at the same persisted store.
build_store = temporal.build_store
open_memory = temporal.open_memory
graph_store = temporal.graph_store
OllamaLLM = temporal.OllamaLLM
OllamaEmbedder = temporal.OllamaEmbedder
OllamaClient = temporal.OllamaClient

DEFAULT_CORPUS_PATH = temporal.DEFAULT_CORPUS_PATH
DEFAULT_DATA_DIR = temporal.DEFAULT_DATA_DIR
DEFAULT_NAMESPACE = temporal.DEFAULT_NAMESPACE
DEFAULT_GRAPH_LLM_MODEL = "qwen3:4b"

GRAPH_QUERY_PLAN_SYSTEM = (
    "You plan graph-guided retrieval for a financial filing question. Use only the question text. "
    "Return ONLY a JSON object with exactly these keys: topics (array of at most 5 concise strings), "
    "intent (one of lookup, comparison, evolution, trend, risk, calculation), and "
    "section_hints (array using only business, risk_factors, financial_statements, market_risk, "
    "legal_proceedings, executive_compensation, ownership, or empty). Do not answer the question, "
    "do not invent facts, and do not introduce companies or years not present in the question."
)

_STRUCTURAL_PREDICATES = frozenset(
    {
        "belongs_to_company",
        "belongs_to_filing",
        "has_year",
        "has_section",
        "has_section_family",
        "has_comparison_group",
        "has_member_document",
        "has_next_document_in_filing",
        "has_previous_document_in_filing",
        "has_next_same_company_section_doc",
        "has_previous_same_company_section_doc",
        "has_next_same_company_form_filing",
        "has_previous_same_company_form_filing",
    }
)
_COMPARISON_PREDICATES = frozenset(
    {
        "has_section",
        "has_section_family",
        "has_comparison_group",
        "has_member_document",
        "has_next_same_company_section_doc",
        "has_previous_same_company_section_doc",
        "has_next_same_company_form_filing",
        "has_previous_same_company_form_filing",
    }
)
_TOPIC_STOPWORDS = frozenset(
    {
        "about", "across", "after", "analyze", "and", "are", "between", "change", "changes",
        "compare", "comparing", "company", "did", "does", "each", "from", "how", "identify",
        "including", "into", "its", "main", "more", "of", "or", "report", "specific", "the",
        "their", "these", "this", "through", "to", "trend", "what", "which", "with", "year",
        "years",
    }
)
_LEGAL_COMPANY_NAME_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&'’.-]*\s+){1,7}"
    r"(?:Inc(?:orporated)?|Corp(?:oration)?|Company|Co|Ltd|Limited|LLC|L\.L\.C\.|PLC)\.?\b"
)


@dataclass(frozen=True)
class GraphQueryPlan:
    """Question-only retrieval plan; no answer or source text is used to create it."""

    topics: tuple[str, ...]
    intent: str
    section_hints: tuple[str, ...]
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "topics": list(self.topics),
            "intent": self.intent,
            "section_hints": list(self.section_hints),
            "source": self.source,
        }


def _normalise_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _unique_text(values: Iterable[Any], *, limit: int) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalise_space(str(value or ""))[:160]
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if len(output) >= limit:
            break
    return tuple(output)


def _legal_company_names_in_question(query: str) -> tuple[str, ...]:
    """Return explicit legal-name-looking phrases without guessing nicknames.

    Qwen remains the normal company extractor. This fallback exists only for a
    malformed or empty model response, and each candidate still has to resolve
    to exactly one ticker through raw stored filing text before it is used.
    """
    return _unique_text(
        (match.group(0).rstrip(". ") for match in _LEGAL_COMPANY_NAME_PATTERN.finditer(query)),
        limit=4,
    )


def _apply_legal_name_company_fallback(
    query: str,
    filters: temporal.QueryFilters,
    episodes: Iterable[Any],
    namespace: str,
) -> temporal.QueryFilters:
    """Recover an explicit legal company name only when the LLM found no ticker."""
    if filters.company_tickers:
        return filters
    legal_names = _legal_company_names_in_question(query)
    if not legal_names:
        return filters
    resolution = temporal.resolve_llm_company_names_in_existing_chunks(
        legal_names,
        episodes,
        namespace,
    )
    if not resolution.company_tickers:
        return filters
    return temporal.QueryFilters(
        company_tickers=resolution.company_tickers,
        company_names=resolution.company_names,
        years=filters.years,
        source=f"{filters.source}+legal_name_content_fallback",
        start_date=filters.start_date,
        end_date=filters.end_date,
        relative_time_expression=filters.relative_time_expression,
        relative_time_reference=filters.relative_time_reference,
        company_resolution="legal_name_content_exact_fallback",
        company_aliases_matched=resolution.matched_company_names,
    )


def _fallback_plan(query: str) -> GraphQueryPlan:
    words = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", query)
        if token.lower() not in _TOPIC_STOPWORDS and not temporal.YEAR_RE.fullmatch(token)
    ]
    query_lower = query.lower()
    if any(word in query_lower for word in ("risk", "cyber", "legal", "litigation", "permit")):
        intent = "risk"
    elif any(word in query_lower for word in ("calculate", "percentage", "total", "amount", "fees")):
        intent = "calculation"
    elif any(word in query_lower for word in ("trend", "growth", "increase", "decrease")):
        intent = "trend"
    elif any(word in query_lower for word in ("compare", "versus", " vs ", "between")):
        intent = "comparison"
    elif any(word in query_lower for word in ("evolve", "evolution", "changed", "develop")):
        intent = "evolution"
    else:
        intent = "lookup"

    section_hints: list[str] = []
    if any(word in query_lower for word in ("risk", "cyber", "privacy", "permit", "legal")):
        section_hints.append("risk_factors")
    if any(word in query_lower for word in ("revenue", "income", "fees", "loss", "financial")):
        section_hints.append("financial_statements")
    if any(word in query_lower for word in ("compensation", "director", "executive", "ltip")):
        section_hints.append("executive_compensation")
    if any(word in query_lower for word in ("ownership", "shareholding", "voting")):
        section_hints.append("ownership")
    if any(word in query_lower for word in ("market risk", "foreign exchange", "commodity")):
        section_hints.append("market_risk")
    return GraphQueryPlan(
        topics=_unique_text(words, limit=5),
        intent=intent,
        section_hints=tuple(section_hints),
        source="deterministic_fallback",
    )


def extract_graph_query_plan(
    query: str,
    llm: Any | None,
    *,
    use_llm: bool,
) -> GraphQueryPlan:
    """Derive a small, question-only graph plan with a safe deterministic fallback."""
    fallback = _fallback_plan(query)
    if not use_llm or llm is None:
        return fallback
    try:
        parsed = temporal._parse_json_object(  # noqa: SLF001 -- shared local JSON parser
            llm.complete(
                f"Question:\n{query}\n\nRetrieval plan JSON:",
                system=GRAPH_QUERY_PLAN_SYSTEM,
                num_predict=256,
            )
        )
        topics = _unique_text(parsed.get("topics", []), limit=5)
        intent = str(parsed.get("intent") or fallback.intent).strip().lower()
        if intent not in {"lookup", "comparison", "evolution", "trend", "risk", "calculation"}:
            intent = fallback.intent
        allowed_sections = {
            "business",
            "risk_factors",
            "financial_statements",
            "market_risk",
            "legal_proceedings",
            "executive_compensation",
            "ownership",
        }
        section_hints = tuple(
            section
            for section in _unique_text(parsed.get("section_hints", []), limit=4)
            if section in allowed_sections
        )
        return GraphQueryPlan(
            topics=topics or fallback.topics,
            intent=intent,
            section_hints=section_hints or fallback.section_hints,
            source="ollama_qwen3_4b",
        )
    except Exception:  # noqa: BLE001 -- query retrieval must remain available when Ollama is unavailable
        return GraphQueryPlan(
            topics=fallback.topics,
            intent=fallback.intent,
            section_hints=fallback.section_hints,
            source="deterministic_after_llm_failure",
        )


def _episode_doc_id(episode: Any) -> str:
    return str(getattr(episode, "metadata", {}).get("doc_id") or getattr(episode, "id", ""))


def _entity_names_by_id(graph: Any) -> dict[str, str]:
    entities = getattr(graph, "entities", {})
    if not isinstance(entities, dict):
        return {}
    return {
        str(entity_id): str(getattr(entity, "name", ""))
        for entity_id, entity in entities.items()
        if getattr(entity, "name", "")
    }


def _seed_graph_document_paths(
    mem: Any,
    *,
    namespace: str,
    seed_doc_ids: Iterable[str],
    allowed_doc_ids: set[str],
    max_hops: int,
    max_neighbors_per_node: int,
    comparison_intent: bool,
) -> dict[str, list[str]]:
    """Traverse structural Engram edges from semantically relevant document anchors.

    The graph is bounded to two short evidence hops by default.  Every returned
    document must already satisfy the company/time metadata filter represented by
    ``allowed_doc_ids``; graph paths expand structure, not search scope.
    """
    if max_hops <= 0 or not allowed_doc_ids:
        return {}
    graph = mem.graph
    names = _entity_names_by_id(graph)
    if not names:
        return {}

    allowed_predicates = set(_STRUCTURAL_PREDICATES)
    if comparison_intent:
        allowed_predicates.update(_COMPARISON_PREDICATES)

    starts: list[str] = []
    for doc_id in seed_doc_ids:
        entity = graph.get_entity(namespace, doc_id)
        if entity is not None:
            starts.append(str(entity.id))
    if not starts:
        return {}

    paths_by_doc: dict[str, list[str]] = defaultdict(list)
    frontier: deque[tuple[str, int, tuple[str, ...]]] = deque(
        (entity_id, 0, ()) for entity_id in dict.fromkeys(starts)
    )
    best_depth: dict[str, int] = {entity_id: 0 for entity_id in starts}
    max_edges = max(1, max_neighbors_per_node)

    while frontier:
        entity_id, depth, path_parts = frontier.popleft()
        if depth >= max_hops:
            continue
        edges: list[tuple[str, Any, str]] = []
        for direction in ("out", "in"):
            for relation in graph.neighbors(entity_id, direction=direction):
                if relation.predicate not in allowed_predicates:
                    continue
                neighbor_id = (
                    str(relation.object_id) if direction == "out" else str(relation.subject_id)
                )
                neighbor_name = names.get(neighbor_id, "")
                if not neighbor_name:
                    continue
                edges.append((direction, relation, neighbor_name))
        edges.sort(key=lambda item: (item[1].predicate, item[2], item[1].id))

        current_name = names.get(entity_id, entity_id)
        for direction, relation, neighbor_name in edges[:max_edges]:
            arrow = "→" if direction == "out" else "←"
            next_path = path_parts + (f"{current_name} {arrow}[{relation.predicate}] {neighbor_name}",)
            neighbor_id = (
                str(relation.object_id) if direction == "out" else str(relation.subject_id)
            )
            if neighbor_name in allowed_doc_ids:
                rendered = " | ".join(next_path)
                doc_paths = paths_by_doc[neighbor_name]
                if rendered not in doc_paths and len(doc_paths) < 3:
                    doc_paths.append(rendered)
            next_depth = depth + 1
            if next_depth >= max_hops:
                continue
            previous_depth = best_depth.get(neighbor_id)
            if previous_depth is not None and previous_depth <= next_depth:
                continue
            best_depth[neighbor_id] = next_depth
            frontier.append((neighbor_id, next_depth, next_path))
    return dict(paths_by_doc)


def _add_candidates(
    candidate_map: dict[str, dict[str, Any]],
    ranked: Iterable[tuple[float, Any]],
    *,
    route: str,
    graph_paths_by_doc: dict[str, list[str]] | None = None,
    score_bonus: float = 0.0,
) -> None:
    for raw_score, episode in ranked:
        episode_id = str(getattr(episode, "id", ""))
        if not episode_id:
            continue
        score = float(raw_score) + score_bonus
        doc_id = _episode_doc_id(episode)
        entry = candidate_map.get(episode_id)
        if entry is None:
            entry = {
                "episode": episode,
                "score": score,
                "routes": set(),
                "graph_paths": set(),
            }
            candidate_map[episode_id] = entry
        else:
            entry["score"] = max(float(entry["score"]), score)
        entry["routes"].add(route)
        for path in (graph_paths_by_doc or {}).get(doc_id, []):
            entry["graph_paths"].add(path)


def _select_temporally_covered_candidates(
    candidate_map: dict[str, dict[str, Any]],
    *,
    requested_years: Iterable[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Reserve one high-relevance result per explicit year, then fill globally."""
    ranked = sorted(
        candidate_map.values(),
        key=lambda item: (-float(item["score"]), str(getattr(item["episode"], "id", ""))),
    )
    selected: list[dict[str, Any]] = []
    selected_episode_ids: set[str] = set()
    selected_doc_ids: set[str] = set()

    for year in requested_years:
        for item in ranked:
            episode = item["episode"]
            episode_id = str(getattr(episode, "id", ""))
            doc_id = _episode_doc_id(episode)
            if episode_id in selected_episode_ids or doc_id in selected_doc_ids:
                continue
            if year not in temporal._episode_years(episode):  # noqa: SLF001 -- shared temporal semantics
                continue
            selected.append(item)
            selected_episode_ids.add(episode_id)
            selected_doc_ids.add(doc_id)
            break

    for item in ranked:
        if len(selected) >= limit:
            break
        episode = item["episode"]
        episode_id = str(getattr(episode, "id", ""))
        doc_id = _episode_doc_id(episode)
        if episode_id in selected_episode_ids or doc_id in selected_doc_ids:
            continue
        selected.append(item)
        selected_episode_ids.add(episode_id)
        selected_doc_ids.add(doc_id)
    return selected[:limit]


def _retrieved_payload(selected: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in selected:
        episode = item["episode"]
        metadata = getattr(episode, "metadata", {}) or {}
        payload.append(
            {
                "doc_id": _episode_doc_id(episode),
                "episode_id": str(getattr(episode, "id", "")),
                "score": round(float(item["score"]), 8),
                "ticker": metadata.get("ticker", ""),
                "document_type": metadata.get("document_type", metadata.get("form_type", "")),
                "filing_date": metadata.get("filing_date", ""),
                "reporting_periods": metadata.get("reporting_periods", []),
                "retrieval_routes": sorted(item["routes"]),
                "graph_paths": sorted(item["graph_paths"]),
            }
        )
    return payload


def query_store(
    *,
    store_dir: Path,
    namespace: str,
    query: str,
    top_k: int,
    context_chars: int,
    answer_context_chars: int,
    include_profile_context: bool = False,
    query_timings_path: Path | None = None,
    llm_query_filters: bool = False,
    llm_query_time_company_resolver: bool = True,
    llm_query_decomposition: bool = True,
    use_filing_summaries: bool = True,
    filing_summary_top_k: int = 4,
    evidence_per_filing_summary: int = 2,
    graph_max_hops: int = 2,
    graph_seed_k: int = 12,
    graph_candidate_k: int = 40,
    graph_max_neighbors: int = 80,
    llm: Any | None = None,
    embedder: Any | None = None,
    generate_answer: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Retrieve raw filing evidence through temporal filters, graph paths, and vector reranking.

    The graph never expands the selected query scope. It only connects documents
    already admitted by the metadata stage, then a common query embedding reranks
    that structural candidate set with direct vector evidence. A recorded
    company-only fallback can be selected only when the exact company/time scope
    is empty because legacy reporting-period metadata is unavailable.
    """
    del include_profile_context
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if graph_max_hops < 0:
        raise ValueError("graph_max_hops cannot be negative")
    if graph_seed_k <= 0 or graph_candidate_k <= 0 or graph_max_neighbors <= 0:
        raise ValueError("graph candidate limits must be positive")
    if filing_summary_top_k <= 0 or evidence_per_filing_summary <= 0:
        raise ValueError("filing summary limits must be positive")

    seed_everything(seed)
    if isinstance(llm, OllamaLLM):
        llm.seed = seed
    timings_ms: dict[str, float] = {}
    total_start = time.perf_counter()
    filters = temporal.QueryFilters((), (), (), "not_started")
    graph_plan = GraphQueryPlan((), "lookup", (), "not_started")
    all_episodes: list[Any] = []
    raw_candidates: list[Any] = []
    filing_summary_candidates: list[Any] = []
    metadata_scope = "company_and_time"
    filing_summaries: list[tuple[float, Any]] = []
    graph_paths_by_doc: dict[str, list[str]] = {}
    selected: list[dict[str, Any]] = []
    broad_query = False
    success = False
    error_message: str | None = None

    try:
        step_start = time.perf_counter()
        mem = open_memory(store_dir, embedder=embedder)
        timings_ms["open_store"] = (time.perf_counter() - step_start) * 1000
        all_episodes = list(mem.episodes_doc.values())

        step_start = time.perf_counter()
        filter_timings: dict[str, float] = {}
        filters = temporal.extract_query_filters(
            query,
            all_episodes,
            namespace,
            llm=llm,
            llm_query_filters=llm_query_filters,
            llm_query_time_company_resolver=llm_query_time_company_resolver,
            timings_ms=filter_timings,
        )
        timings_ms.update(filter_timings)
        fallback_started = time.perf_counter()
        filters = _apply_legal_name_company_fallback(
            query,
            filters,
            all_episodes,
            namespace,
        )
        timings_ms["legal_name_company_fallback"] = (
            time.perf_counter() - fallback_started
        ) * 1000
        timings_ms["query_decomposition"] = (time.perf_counter() - step_start) * 1000

        step_start = time.perf_counter()
        graph_plan = extract_graph_query_plan(
            query,
            llm,
            use_llm=llm_query_decomposition,
        )
        timings_ms["graph_query_planning"] = (time.perf_counter() - step_start) * 1000

        step_start = time.perf_counter()
        raw_candidates = temporal.filter_episodes_by_query_metadata(
            all_episodes,
            namespace,
            filters,
            artifact_type="filing_chunk",
        )
        filing_summary_candidates = temporal.filter_episodes_by_query_metadata(
            all_episodes,
            namespace,
            filters,
            artifact_type="filing_summary",
        )
        if not raw_candidates and filters.company_tickers and filters.years:
            # Some historical LT-QA filings describe a requested reporting year
            # but carry only their later SEC filing date in existing metadata.
            # Preserve the resolved company constraint and relax only the empty
            # time predicate; the vector and graph stages still rank evidence.
            company_only_filters = temporal.QueryFilters(
                company_tickers=filters.company_tickers,
                company_names=filters.company_names,
                years=(),
                source=f"{filters.source}+company_only_temporal_fallback",
                company_resolution=filters.company_resolution,
                company_aliases_matched=filters.company_aliases_matched,
            )
            raw_candidates = temporal.filter_episodes_by_query_metadata(
                all_episodes,
                namespace,
                company_only_filters,
                artifact_type="filing_chunk",
            )
            filing_summary_candidates = temporal.filter_episodes_by_query_metadata(
                all_episodes,
                namespace,
                company_only_filters,
                artifact_type="filing_summary",
            )
            metadata_scope = "company_only_after_empty_time_filter"
        broad_query = bool(use_filing_summaries and temporal._is_broad_query(query, filters))
        timings_ms["metadata_filter"] = (time.perf_counter() - step_start) * 1000

        candidate_map: dict[str, dict[str, Any]] = {}
        direct_ranked: list[tuple[float, Any]] = []
        graph_ranked: list[tuple[float, Any]] = []
        summary_expanded: list[tuple[float, Any]] = []
        seed_doc_ids: set[str] = set()

        if raw_candidates:
            raw_ids = {str(episode.id) for episode in raw_candidates}
            raw_doc_ids = {_episode_doc_id(episode) for episode in raw_candidates}
            step_start = time.perf_counter()
            query_embedding = mem.embedder.embed(query)
            timings_ms["query_embedding"] = (time.perf_counter() - step_start) * 1000

            step_start = time.perf_counter()
            direct_ranked = mem.episodes_vec.search(
                query_embedding,
                top_k=min(max(top_k, graph_seed_k), len(raw_ids)),
                where=lambda episode: str(episode.id) in raw_ids,
            )
            seed_doc_ids.update(_episode_doc_id(episode) for _, episode in direct_ranked)
            timings_ms["raw_vector_retrieval"] = (time.perf_counter() - step_start) * 1000

            step_start = time.perf_counter()
            # Topic embeddings broaden only the graph anchors.  Final ranking
            # still uses the original question embedding, keeping scores comparable.
            topic_seed_count = 0
            for topic in graph_plan.topics[:5]:
                topic_embedding = mem.embedder.embed(f"{query}\nFocus: {topic}")
                topic_hits = mem.episodes_vec.search(
                    topic_embedding,
                    top_k=min(3, len(raw_ids)),
                    where=lambda episode: str(episode.id) in raw_ids,
                )
                seed_doc_ids.update(_episode_doc_id(episode) for _, episode in topic_hits)
                topic_seed_count += len(topic_hits)
            timings_ms["topic_anchor_retrieval"] = (time.perf_counter() - step_start) * 1000

            step_start = time.perf_counter()
            graph_paths_by_doc = _seed_graph_document_paths(
                mem,
                namespace=namespace,
                seed_doc_ids=seed_doc_ids,
                allowed_doc_ids=raw_doc_ids,
                max_hops=graph_max_hops,
                max_neighbors_per_node=graph_max_neighbors,
                comparison_intent=graph_plan.intent in {"comparison", "evolution", "trend"},
            )
            graph_doc_ids = set(graph_paths_by_doc)
            graph_episode_ids = {
                str(episode.id)
                for episode in raw_candidates
                if _episode_doc_id(episode) in graph_doc_ids
            }
            if graph_episode_ids:
                graph_ranked = mem.episodes_vec.search(
                    query_embedding,
                    top_k=min(max(top_k, graph_candidate_k), len(graph_episode_ids)),
                    where=lambda episode: str(episode.id) in graph_episode_ids,
                )
            timings_ms["graph_path_expansion"] = (time.perf_counter() - step_start) * 1000

            step_start = time.perf_counter()
            if broad_query and filing_summary_candidates:
                summary_ids = {str(episode.id) for episode in filing_summary_candidates}
                filing_summaries = mem.summary_vec.search(
                    query_embedding,
                    top_k=min(filing_summary_top_k, len(summary_ids)),
                    where=lambda episode: str(episode.id) in summary_ids,
                )
                source_episode_ids = temporal._summary_source_episode_ids(filing_summaries)
                if source_episode_ids:
                    summary_expanded = mem.episodes_vec.search(
                        query_embedding,
                        top_k=min(
                            max(1, evidence_per_filing_summary * len(filing_summaries)),
                            len(source_episode_ids),
                        ),
                        where=lambda episode: str(episode.id) in source_episode_ids,
                    )
            timings_ms["filing_summary_retrieval"] = (time.perf_counter() - step_start) * 1000

            _add_candidates(candidate_map, direct_ranked, route="direct_vector")
            _add_candidates(
                candidate_map,
                graph_ranked,
                route="graph_path_rerank",
                graph_paths_by_doc=graph_paths_by_doc,
                score_bonus=0.01,
            )
            _add_candidates(candidate_map, summary_expanded, route="filing_summary_evidence")
            selected = _select_temporally_covered_candidates(
                candidate_map,
                requested_years=filters.years,
                limit=top_k,
            )
            timings_ms["candidate_merge_rerank"] = (time.perf_counter() - step_start) * 1000
            graph_diagnostics = {
                "graph_available": bool(_entity_names_by_id(mem.graph)),
                "max_hops": graph_max_hops,
                "seed_documents": len(seed_doc_ids),
                "topic_seed_hits": topic_seed_count,
                "graph_connected_documents": len(graph_paths_by_doc),
                "graph_reranked_chunks": len(graph_ranked),
                "graph_paths_retained": sum(len(paths) for paths in graph_paths_by_doc.values()),
            }
        else:
            timings_ms.update(
                {
                    "query_embedding": 0.0,
                    "raw_vector_retrieval": 0.0,
                    "topic_anchor_retrieval": 0.0,
                    "graph_path_expansion": 0.0,
                    "filing_summary_retrieval": 0.0,
                    "candidate_merge_rerank": 0.0,
                }
            )
            graph_diagnostics = {
                "graph_available": False,
                "max_hops": graph_max_hops,
                "seed_documents": 0,
                "topic_seed_hits": 0,
                "graph_connected_documents": 0,
                "graph_reranked_chunks": 0,
                "graph_paths_retained": 0,
            }

        step_start = time.perf_counter()
        retrieved = [(float(item["score"]), item["episode"]) for item in selected]
        summary_context = temporal._render_filing_summary_context(filing_summaries)
        raw_context = temporal._render_raw_rag_context(retrieved)
        full_context = "\n\n".join(part for part in (summary_context, raw_context) if part)
        context = full_context if context_chars <= 0 else full_context[:context_chars]
        answer_context = (
            full_context if answer_context_chars <= 0 else full_context[:answer_context_chars]
        )
        timings_ms["context_assembly"] = (time.perf_counter() - step_start) * 1000

        payload: dict[str, Any] = {
            "query": query,
            "filters": filters.as_dict(),
            "graph_query_plan": graph_plan.as_dict(),
            "query_scope": "broad" if broad_query else "specific",
            "metadata_scope": metadata_scope,
            "documents_before_filter": sum(
                episode.user_id == namespace and temporal._artifact_type(episode) == "filing_chunk"
                for episode in all_episodes
            ),
            "documents_after_filter": len(raw_candidates),
            "filing_summaries_after_filter": len(filing_summary_candidates),
            "retrieval_method": "temporal metadata-filtered graph-guided hybrid RAG",
            "graph_retrieval": graph_diagnostics,
            "filing_summaries": temporal._filing_summary_payload(filing_summaries),
            "expanded_evidence_chunks": len(summary_expanded),
            "retrieved_documents": _retrieved_payload(selected),
            "context_chars": len(full_context),
            "context_truncated": len(context) < len(full_context),
            "context": context,
        }

        step_start = time.perf_counter()
        if not selected:
            payload["answer"] = (
                "No raw evidence documents matched the extracted company and year filters. "
                "Inspect `filters` and the persisted temporal metadata."
            )
        elif llm is None or not generate_answer:
            payload["answer"] = "Relevant graph-guided documents were retrieved."
        else:
            payload["answer"] = llm.complete(
                f"Question:\n{query}\n\nGraph-guided retrieval context:\n{answer_context}\n\nAnswer:",
                system=temporal.FIN_RATE_ANSWER_SYSTEM,
                num_predict=512,
            )
            payload["answer_context_chars"] = len(answer_context)
            payload["answer_context_truncated"] = len(answer_context) < len(full_context)
        timings_ms["answer_generation"] = (time.perf_counter() - step_start) * 1000
        success = True
        return payload
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        timings_ms["total"] = (time.perf_counter() - total_start) * 1000
        temporal._append_query_timing(  # noqa: SLF001 -- keep the JSONL timing schema consistent
            query_timings_path or store_dir / "graph_query_timings.jsonl",
            {
                "recorded_at_utc": temporal._utc_now_iso(),
                "query": query,
                "namespace": namespace,
                "filters": filters.as_dict(),
                "graph_query_plan": graph_plan.as_dict(),
                "query_scope": "broad" if broad_query else "specific",
                "metadata_scope": metadata_scope,
                "documents_before_filter": sum(
                    episode.user_id == namespace and temporal._artifact_type(episode) == "filing_chunk"
                    for episode in all_episodes
                ),
                "documents_after_filter": len(raw_candidates),
                "filing_summaries_after_filter": len(filing_summary_candidates),
                "filing_summaries_retrieved": len(filing_summaries),
                "documents_retrieved": len(selected),
                "graph_retrieval": graph_diagnostics if "graph_diagnostics" in locals() else {},
                "durations_ms": {name: round(value, 3) for name, value in timings_ms.items()},
                "success": success,
                "error": error_message,
            },
        )


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or graph-query an existing Temporal Engram store using qwen3:4b."
    )
    parser.add_argument("--build", action="store_true", help="Build using the compatible Temporal Engram schema.")
    parser.add_argument("--query", help="Run graph-guided temporal retrieval for a question.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--document-ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--context-chars", type=int, default=5_000)
    parser.add_argument("--answer-context-chars", type=int, default=8_000)
    parser.add_argument("--query-timings-file", type=Path)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default=DEFAULT_GRAPH_LLM_MODEL)
    parser.add_argument("--ollama-embed-model")
    parser.add_argument("--ollama-timeout", type=float, default=300.0)
    parser.add_argument("--no-answer", action="store_true")
    parser.add_argument("--no-llm-query-decomposition", action="store_true")
    parser.add_argument("--no-llm-company-resolver", action="store_true")
    parser.add_argument("--llm-query-filters", action="store_true")
    parser.add_argument("--no-filing-summary-retrieval", action="store_true")
    parser.add_argument("--filing-summary-top-k", type=int, default=4)
    parser.add_argument("--evidence-per-filing-summary", type=int, default=2)
    parser.add_argument("--graph-max-hops", type=int, default=2)
    parser.add_argument("--graph-seed-k", type=int, default=12)
    parser.add_argument("--graph-candidate-k", type=int, default=40)
    parser.add_argument("--graph-max-neighbors", type=int, default=80)
    parser.add_argument("--no-summaries", action="store_true")
    parser.add_argument("--no-filing-summaries", action="store_true")
    parser.add_argument("--extract-text-facts", action="store_true")
    parser.add_argument("--llm-summaries", action="store_true")
    parser.add_argument("--llm-document-metadata", action="store_true")
    parser.add_argument("--llm-filing-summaries", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    store_dir = args.store_dir or temporal._service_namespace_dir(args.data_dir, args.namespace)
    seed_everything(args.seed)
    llm = OllamaLLM(
        args.ollama_model,
        base_url=args.ollama_url,
        timeout=args.ollama_timeout,
        seed=args.seed,
    )
    embedder = (
        OllamaEmbedder(args.ollama_embed_model, base_url=args.ollama_url, timeout=args.ollama_timeout)
        if args.ollama_embed_model
        else None
    )

    if args.build:
        document_ids = (
            temporal.load_document_ids_file(args.document_ids_file)
            if args.document_ids_file is not None
            else None
        )
        stats = build_store(
            corpus_path=args.corpus,
            store_dir=store_dir,
            namespace=args.namespace,
            limit=args.limit,
            reset=args.reset,
            summarize=not args.no_summaries,
            extract_text_facts=args.extract_text_facts,
            deep_relationships=True,
            llm=(
                llm
                if any(
                    (
                        args.extract_text_facts,
                        args.llm_summaries,
                        args.llm_document_metadata,
                        args.llm_filing_summaries,
                    )
                )
                else None
            ),
            embedder=embedder,
            llm_summaries=args.llm_summaries,
            llm_document_metadata=args.llm_document_metadata,
            filing_summaries=not args.no_filing_summaries,
            llm_filing_summaries=args.llm_filing_summaries,
            document_ids=document_ids,
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
            query_timings_path=args.query_timings_file,
            llm_query_filters=args.llm_query_filters,
            llm_query_time_company_resolver=not args.no_llm_company_resolver,
            llm_query_decomposition=not args.no_llm_query_decomposition,
            use_filing_summaries=not args.no_filing_summary_retrieval,
            filing_summary_top_k=args.filing_summary_top_k,
            evidence_per_filing_summary=args.evidence_per_filing_summary,
            graph_max_hops=args.graph_max_hops,
            graph_seed_k=args.graph_seed_k,
            graph_candidate_k=args.graph_candidate_k,
            graph_max_neighbors=args.graph_max_neighbors,
            llm=llm,
            embedder=embedder,
            generate_answer=not args.no_answer,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if not (args.build or args.query):
        parser.print_help()


if __name__ == "__main__":
    _main()
