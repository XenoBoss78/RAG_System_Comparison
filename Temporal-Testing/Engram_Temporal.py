"""Build and query a local Engram memory store from the Fin-RATE corpus.

This mirrors the local corpus.jsonl workflow used by the vector/Chroma scripts in
the parent workspace, but writes into Engram instead:

  * raw filing chunks are stored as dated episodes with ingestion, company, type, and reporting-period metadata
  * deterministic metadata facts seed Engram's semantic graph locally
  * corpus-derived edges add longitudinal and comparison relationships
  * the result is persisted as JSONL under Fin-RATE/engram_data

Start small from the Temporal-Testing directory:

    python Engram_Temporal.py --build --limit 200 --reset
    python Engram_Temporal.py --query "Valero financial results" --ollama-model llama3.1:8b
    python Engram_Temporal.py --graph-query VLO --export-graph graph.json

Then remove --limit for a full local build.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
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
DEFAULT_OLLAMA_EMBED_BATCH_SIZE = 16

sys.path.insert(0, str(ENGRAM_DIR))

from engram import Memory  # noqa: E402
from engram.consolidate.classify import classify_fact  # noqa: E402
from engram.embed.base import Embedder  # noqa: E402
from engram.llm.base import LLM  # noqa: E402
from engram.types import Fact  # noqa: E402


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
RELATIVE_TIME_RE = re.compile(
    r"\b(?:within\s+)?(?:the\s+)?(?P<qualifier>past|last|previous)\s+"
    r"(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?P<unit>years?|quarters?|months?)\b",
    re.IGNORECASE,
)
CURRENT_TIME_RE = re.compile(r"\b(?:this|current|latest)\s+(?P<unit>year|quarter|month)\b", re.IGNORECASE)
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
BROAD_QUERY_RE = re.compile(
    r"\b(?:across|all|compare|comparison|change(?:d|s)?|develop(?:ed|ment|s)?|evolv(?:e|ed|ing|ion)|"
    r"history|how\s+did|over\s+time|progress(?:ed|ion)?|trend(?:s)?|versus|vs\.?)\b",
    re.IGNORECASE,
)
MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
NATURAL_DATE_RE = re.compile(
    rf"\b(?P<date>{MONTH_NAMES})\W+\d{{1,2}}\W+\d{{4}}\b",
    re.IGNORECASE,
)
PERIOD_ENDED_RE = re.compile(
    rf"\b(?P<label>(?:(?:fiscal|financial)\s+)?(?:first|second|third|fourth|[1-4](?:st|nd|rd|th))?\s*"
    rf"(?:quarter|year|month|half[- ]year|six[- ]month(?:s)?)\s+ended\s+"
    rf"(?P<date>{MONTH_NAMES}\W+\d{{1,2}}\W+\d{{4}}))\b",
    re.IGNORECASE,
)
PERIOD_RANGE_RE = re.compile(
    rf"\b(?P<label>(?:from|between)\s+(?P<start>{MONTH_NAMES}\W+\d{{1,2}}\W+\d{{4}})\s+"
    rf"(?:to|through|and)\s+(?P<end>{MONTH_NAMES}\W+\d{{1,2}}\W+\d{{4}}))\b",
    re.IGNORECASE,
)
AS_OF_RE = re.compile(
    rf"\b(?P<label>as\s+of\s+(?P<date>{MONTH_NAMES}\W+\d{{1,2}}\W+\d{{4}}))\b",
    re.IGNORECASE,
)
COMPANY_NAME_RE = re.compile(
    r"(?P<company>[A-Z][A-Z0-9&.,'()\- /]{2,})\s*\n\s*\(Exact name of registrant",
    re.IGNORECASE,
)
QUERY_TIME_GENERIC_COMPANY_NAMES = frozenset(
    {
        "company",
        "the company",
        "our company",
        "this company",
        "the registrant",
        "registrant",
        "we",
        "us",
    }
)
KEYWORD_STOPWORDS = frozenset(
    {
        "about", "after", "also", "among", "and", "are", "been", "being", "between", "but",
        "can", "company", "could", "date", "did", "does", "during", "each", "for", "from",
        "have", "herein", "into", "its", "not", "our", "report", "that", "the", "their", "this",
        "these", "they", "through", "under", "was", "were", "will", "with", "would", "you",
    }
)

FIN_RATE_DOCUMENT_METADATA_SYSTEM = (
    "You extract metadata for one SEC filing document chunk that will be persisted in a temporal graph RAG "
    "index. Use only text supplied in the prompt. Return ONLY a JSON object with these exact keys: "
    "company_name (string), document_type (string), reporting_periods (array), and keywords (array). "
    "Each reporting_periods item must have label, period_type, start_date, and end_date as plain strings. "
    "Use YYYY-MM-DD for dates and an empty string when a date is not explicit. period_type must be one of "
    "quarter, annual, semiannual, monthly, range, instant, or unknown. Report only periods the document "
    "actually describes; do not mistake the filing date for a reporting period. company_name must be the "
    "registrant/company as written, or an empty string when absent. Give 5-12 concise, retrieval-useful "
    "keywords. Do not invent, infer, or calculate."
)

FIN_RATE_SUMMARY_SYSTEM = (
    "You summarize one SEC filing chunk for a financial retrieval memory index. Preserve exact ticker, "
    "company names, filing form, filing date, fiscal periods, section/item names, financial metrics, "
    "amounts, units, percentages, risks, exhibits, named executives, auditors, accounting standards, and "
    "cross-references. Be concise but do not drop numbers or period labels. No preamble."
)

FIN_RATE_FILING_SUMMARY_SYSTEM = (
    "You create a concise, source-faithful filing-level retrieval summary from excerpts of one SEC filing. "
    "Use only the supplied filing metadata and excerpts. Structure the result with compact labels: "
    "Reporting period, Results and operations, Financial position/liquidity, Risks and strategy, and "
    "Other material disclosures. Include exact figures, units, dates, segment names, and qualifications "
    "only when they occur in the excerpts. If a topic is absent, omit it. Do not calculate, fill gaps, "
    "or state that the summary proves anything not present in the excerpts. This is a navigation layer: "
    "be comprehensive enough to find supporting raw chunks but remain under 700 words. No preamble."
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
    "insufficient. Filing summaries are navigation aids; substantiate material claims and exact figures "
    "with the RAW EVIDENCE chunks. Do not use outside knowledge."
)

FIN_RATE_QUERY_FILTER_SYSTEM = (
    "You identify metadata filters for a financial filing retrieval system. Return ONLY a JSON object with "
    "two keys: company_names (an array of company or ticker strings explicitly named in the query) and "
    "years (an array of four-digit years explicitly named in the query). Do not infer missing companies or "
    "years, calculate relative dates, or include explanations."
)

FIN_RATE_QUERY_COMPANY_NAME_SYSTEM = (
    "Extract only the company or issuer that the user is asking about. Return ONLY a JSON object with the "
    "exact key company_names, whose value is an array of strings. Copy each company name as it appears in "
    "the query. Do not infer a legal name, ticker, parent, subsidiary, peer, or company merely mentioned as "
    "context. If the query names only a ticker, return that ticker string. Return [] when no company is named."
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


@dataclass(frozen=True)
class ReportingPeriod:
    """A period explicitly described by a filing, separate from its filing date."""

    label: str
    period_type: str
    start_date: str = ""
    end_date: str = ""
    source: str = "deterministic"

    def as_dict(self) -> dict[str, str]:
        return {
            "label": self.label,
            "period_type": self.period_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "source": self.source,
        }


@dataclass(frozen=True)
class DocumentTemporalMetadata:
    """Persisted ingestion profile used by retrieval and the document graph."""

    company_name: str
    document_type: str
    ingestion_time: float
    ingestion_time_utc: str
    reporting_periods: tuple[ReportingPeriod, ...]
    keywords: tuple[str, ...]
    source: str

    @property
    def primary_period(self) -> ReportingPeriod | None:
        return self.reporting_periods[0] if self.reporting_periods else None


@dataclass(frozen=True)
class QueryFilters:
    """Company and year constraints extracted before retrieval."""

    company_tickers: tuple[str, ...]
    company_names: tuple[str, ...]
    years: tuple[str, ...]
    source: str
    start_date: str = ""
    end_date: str = ""
    relative_time_expression: str = ""
    relative_time_reference: str = ""
    company_resolution: str = "not_attempted"
    company_aliases_matched: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "company_tickers": list(self.company_tickers),
            "company_names": list(self.company_names),
            "years": list(self.years),
            "source": self.source,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "relative_time_expression": self.relative_time_expression,
            "relative_time_reference": self.relative_time_reference,
            "company_resolution": self.company_resolution,
            "company_aliases_matched": list(self.company_aliases_matched),
        }


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    def request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self.base_url + path
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:  # noqa: S310
                    raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace").strip()
                try:
                    error_message = str(json.loads(error_body).get("error") or error_body)
                except json.JSONDecodeError:
                    error_message = error_body
                # During a model switch Ollama can temporarily return HTTP 400
                # because its internal tokenizer runner has not started yet.
                # That response is safe to retry; other 400 errors are not.
                worker_restart = (
                    exc.code == 400
                    and "connection refused" in error_message.lower()
                    and (
                        "tokenize" in error_message.lower()
                        or "dial tcp" in error_message.lower()
                    )
                )
                if (exc.code in {500, 502, 503, 504} or worker_restart) and attempt < self.max_retries:
                    time.sleep(float(attempt + 1))
                    continue
                raise RuntimeError(
                    f"Ollama request to {self.base_url}{path} failed with HTTP {exc.code}: "
                    f"{error_message or exc.reason}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(float(attempt + 1))
                    continue
                raise RuntimeError(
                    f"Could not reach Ollama at {self.base_url}. Start Ollama and check the URL."
                ) from exc

        raise RuntimeError("Ollama request retry loop ended unexpectedly.")

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
            # Qwen3 otherwise may put its complete result in `thinking`,
            # which is unsuitable for the structured metadata prompts used here.
            "think": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        payload.update(kwargs)
        result = self.client.request("/api/generate", payload)
        return str(result.get("response") or result.get("thinking") or "").strip()


class OllamaEmbedder(Embedder):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 300.0,
        batch_size: int = DEFAULT_OLLAMA_EMBED_BATCH_SIZE,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Ollama embedding batch_size must be positive")
        self.model_name = f"ollama:{model}"
        self.model = model
        self.client = OllamaClient(base_url, timeout)
        self.batch_size = batch_size
        self.dim = 0

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def ensure_dimension(self) -> int:
        """Initialize ``dim`` before reopening a persisted EmbeddingGemma store."""
        if self.dim <= 0:
            self.embed("Embedding dimension initialization probe.")
        return self.dim

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        try:
            all_vectors: list[list[float]] = []
            for start in range(0, len(items), self.batch_size):
                batch = items[start : start + self.batch_size]
                response = self.client.request(
                    "/api/embed",
                    {"model": self.model, "input": batch},
                )
                vectors = response.get("embeddings")
                if (
                    not isinstance(vectors, list)
                    or len(vectors) != len(batch)
                    or not all(isinstance(vector, list) for vector in vectors)
                ):
                    raise ValueError("Ollama /api/embed returned incomplete embeddings")
                all_vectors.extend(vectors)
            if all_vectors:
                self.dim = len(all_vectors[0])
                return all_vectors
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


def _print_build_stage(
    stage: str,
    duration_ms: float,
    **details: int,
) -> None:
    """Emit concise notebook/terminal progress for each long build stage."""
    suffix = ""
    if details:
        suffix = " | " + ", ".join(f"{key}={value}" for key, value in details.items())
    print(f"[Engram build] completed {stage} in {duration_ms / 1000:.1f}s{suffix}", flush=True)


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


def load_document_ids_file(path: Path) -> set[str]:
    """Load a subset of corpus document IDs from a JSON list or a text file.

    A JSON object containing ``document_ids`` is accepted too, which keeps the
    build input convenient for both notebook and scripted workflows. Plain-text
    files use one document ID per non-empty line.
    """
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document ID file not found: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Document ID file is empty: {path}")

    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [line.strip() for line in raw.splitlines() if line.strip()]

    if isinstance(parsed, dict):
        parsed = parsed.get("document_ids")
    if not isinstance(parsed, list):
        raise ValueError(
            "Document ID file must contain a JSON list, a JSON object with a "
            "'document_ids' list, or one document ID per line."
        )

    document_ids = {str(document_id).strip() for document_id in parsed if str(document_id).strip()}
    if not document_ids:
        raise ValueError(f"Document ID file contains no usable IDs: {path}")
    return document_ids


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


def _normalise_space(value: str) -> str:
    """Repair common SEC-export spacing before matching dates or entity names."""
    return re.sub(r"\s+", " ", value.replace("\xa0", " ").replace("Â", " ")).strip()


def _iso_date(value: str) -> str:
    """Return an ISO date only when the source provides a complete calendar date."""
    candidate = _normalise_space(value).replace(",", "")
    try:
        return datetime.strptime(candidate, "%B %d %Y").date().isoformat()
    except ValueError:
        return ""


def _period_type(label: str) -> str:
    text = label.lower()
    if "quarter" in text:
        return "quarter"
    if "half" in text or "six month" in text:
        return "semiannual"
    if "year" in text:
        return "annual"
    if "month" in text:
        return "monthly"
    return "unknown"


def _dedupe_periods(periods: Iterable[ReportingPeriod]) -> tuple[ReportingPeriod, ...]:
    unique: list[ReportingPeriod] = []
    seen: set[tuple[str, str, str, str]] = set()
    for period in periods:
        label = _normalise_space(period.label)
        key = (label.lower(), period.period_type, period.start_date, period.end_date)
        if not label or key in seen:
            continue
        seen.add(key)
        unique.append(
            ReportingPeriod(
                label=label,
                period_type=period.period_type,
                start_date=period.start_date,
                end_date=period.end_date,
                source=period.source,
            )
        )
    return tuple(unique)


def extract_reporting_periods(text: str, *, limit: int = 8) -> tuple[ReportingPeriod, ...]:
    """Extract only explicit reporting periods; never invent a fiscal-period start date.

    SEC filings often mention a filing date and several historical dates. The patterns below retain a
    reporting-period label and exact endpoint(s), leaving a start blank unless the text explicitly states
    one. An optional LLM can subsequently enrich these candidates for unusual wording.
    """
    cleaned = _normalise_space(text)
    periods: list[ReportingPeriod] = []
    for match in PERIOD_ENDED_RE.finditer(cleaned):
        label = _normalise_space(match.group("label"))
        periods.append(
            ReportingPeriod(
                label=label,
                period_type=_period_type(label),
                end_date=_iso_date(match.group("date")),
            )
        )
    for match in PERIOD_RANGE_RE.finditer(cleaned):
        periods.append(
            ReportingPeriod(
                label=_normalise_space(match.group("label")),
                period_type="range",
                start_date=_iso_date(match.group("start")),
                end_date=_iso_date(match.group("end")),
            )
        )
    for match in AS_OF_RE.finditer(cleaned):
        periods.append(
            ReportingPeriod(
                label=_normalise_space(match.group("label")),
                period_type="instant",
                end_date=_iso_date(match.group("date")),
            )
        )
    return _dedupe_periods(periods)[:limit]


def _company_name_from_record(record: dict[str, Any], text: str, ticker: str) -> str:
    for key in ("company_name", "company", "issuer", "registrant", "organization"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return _normalise_space(value)

    match = COMPANY_NAME_RE.search(text)
    if match:
        return _normalise_space(match.group("company"))
    return ticker


def extract_keywords(text: str, meta: FilingMeta, *, limit: int = 12) -> tuple[str, ...]:
    """Produce stable keyword metadata when an LLM is unavailable."""
    counts: Counter[str] = Counter()
    for token in TOKEN_PATTERN.findall(text.lower()):
        token = token.strip(".-/'&")
        if len(token) < 3 or token in KEYWORD_STOPWORDS or token.isdigit():
            continue
        counts[token] += 1

    seed_terms = [meta.ticker.lower(), meta.form_type.lower()]
    if meta.section_family != "unknown":
        seed_terms.extend(token.lower() for token in TOKEN_PATTERN.findall(meta.section_family))
    ordered = list(dict.fromkeys(term for term in seed_terms if term))
    ordered.extend(
        token
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if token not in ordered
    )
    return tuple(ordered[:limit])


def _parse_json_object(response: str) -> dict[str, Any]:
    """Accept a strict JSON object or the object wrapped in an accidental Markdown fence."""
    response = response.strip()
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _llm_reporting_periods(value: Any) -> tuple[ReportingPeriod, ...]:
    if not isinstance(value, list):
        return ()
    periods: list[ReportingPeriod] = []
    allowed_types = {"quarter", "annual", "semiannual", "monthly", "range", "instant", "unknown"}
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        label = _normalise_space(str(item.get("label", "")))[:180]
        period_type = str(item.get("period_type", "unknown")).lower().strip()
        period_type = period_type if period_type in allowed_types else "unknown"
        start_date = str(item.get("start_date", "")).strip()
        end_date = str(item.get("end_date", "")).strip()
        periods.append(
            ReportingPeriod(
                label=label,
                period_type=period_type,
                start_date=start_date if DATE_RE.match(start_date) else "",
                end_date=end_date if DATE_RE.match(end_date) else "",
                source="llm",
            )
        )
    return _dedupe_periods(periods)


def _llm_keywords(value: Any, *, limit: int = 12) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    keywords: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        keyword = _normalise_space(item).lower()[:80]
        if keyword and keyword not in keywords:
            keywords.append(keyword)
        if len(keywords) >= limit:
            break
    return tuple(keywords)


def extract_document_temporal_metadata(
    record: dict[str, Any],
    meta: FilingMeta,
    text: str,
    *,
    ingestion_time: float,
    llm: LLM | None = None,
    llm_text_chars: int = 6000,
    known_company_name: str = "",
) -> DocumentTemporalMetadata:
    """Build a complete, persisted temporal profile for one corpus document.

    Deterministic fields make a local build repeatable and provide a safe fallback. When an LLM is
    configured, it can resolve a human company name, uncommon reporting-period wording, and better
    retrieval keywords; its output is validated before it is stored.
    """
    company_name = _company_name_from_record(record, text, meta.ticker)
    if company_name == meta.ticker and known_company_name:
        company_name = known_company_name
    document_type = f"SEC Form {meta.form_type}" if meta.form_type != "unknown-form" else "SEC filing"
    periods = extract_reporting_periods(text)
    keywords = extract_keywords(text, meta)
    source = "deterministic"

    if llm is not None:
        prompt = (
            "Known metadata (use as context but do not fabricate missing values):\n"
            f"- ticker: {meta.ticker}\n"
            f"- filing form: {meta.form_type}\n"
            f"- filing date: {meta.filing_date}\n"
            f"- title: {meta.title}\n\n"
            f"Document text:\n{text[:llm_text_chars]}\n\nMetadata JSON:"
        )
        try:
            enriched = _parse_json_object(
                llm.complete(prompt, system=FIN_RATE_DOCUMENT_METADATA_SYSTEM, num_predict=768)
            )
            llm_company = enriched.get("company_name")
            llm_document_type = enriched.get("document_type")
            if isinstance(llm_company, str) and llm_company.strip():
                company_name = _normalise_space(llm_company)[:200]
            if isinstance(llm_document_type, str) and llm_document_type.strip():
                document_type = _normalise_space(llm_document_type)[:120]
            llm_periods = _llm_reporting_periods(enriched.get("reporting_periods"))
            if llm_periods:
                periods = llm_periods
            llm_keywords = _llm_keywords(enriched.get("keywords"))
            if llm_keywords:
                keywords = tuple(dict.fromkeys((*llm_keywords, *keywords)))[:12]
            source = "llm_enriched"
        except Exception:  # noqa: BLE001 -- metadata enrichment must not abort a corpus build
            source = "deterministic_llm_fallback"

    return DocumentTemporalMetadata(
        company_name=company_name,
        document_type=document_type,
        ingestion_time=ingestion_time,
        ingestion_time_utc=datetime.fromtimestamp(ingestion_time, tz=timezone.utc).isoformat(),
        reporting_periods=periods,
        keywords=keywords,
        source=source,
    )


def _document_metadata_context(metadata: DocumentTemporalMetadata) -> str:
    """Include structured metadata in the chunk text so the saved embedding can retrieve it."""
    lines = [
        "DOCUMENT METADATA",
        f"Company: {metadata.company_name}",
        f"Document type: {metadata.document_type}",
        f"Ingested at: {metadata.ingestion_time_utc}",
    ]
    for period in metadata.reporting_periods:
        endpoints = " to ".join(value for value in (period.start_date, period.end_date) if value)
        lines.append(f"Reporting period: {period.label}" + (f" ({endpoints})" if endpoints else ""))
    if metadata.keywords:
        lines.append("Keywords: " + ", ".join(metadata.keywords))
    return "\n".join(lines)


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
    if isinstance(embedder, OllamaEmbedder) and (store_dir / "manifest.json").exists():
        # Engram validates the saved vector dimension while opening the store.
        # A freshly constructed Ollama embedder has dim=0 until its first call.
        print("[Engram build] verifying saved-store embedding compatibility", flush=True)
        embedder.ensure_dimension()
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


def _artifact_type(episode: Any) -> str:
    """Treat episodes written before filing summaries existed as raw filing chunks."""
    return str(episode.metadata.get("artifact_type", "filing_chunk"))


def _unique_strings(values: Iterable[Any], *, limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
        if limit is not None and len(output) >= limit:
            break
    return output


def _filing_source_text(chunks: list[Any], *, char_limit: int) -> str:
    """Pack section-diverse filing evidence for one bounded filing-level LLM call."""
    blocks: list[str] = []
    remaining = char_limit
    for chunk in sorted(
        chunks,
        key=lambda episode: (
            str(episode.metadata.get("section_family", "")),
            int(episode.metadata.get("chunk_index", 0)),
            episode.id,
        ),
    ):
        metadata = chunk.metadata
        text = chunk.summary or chunk.content
        header = (
            f"[document={metadata.get('doc_id', chunk.id)} | section={metadata.get('section', 'unknown')} | "
            f"chunk={metadata.get('chunk_index', 0)}]"
        )
        block = f"{header}\n{text.strip()}"
        if remaining <= len(header) + 4:
            break
        blocks.append(block[:remaining])
        remaining -= len(blocks[-1]) + 2
    return "\n\n".join(blocks)


def _deterministic_filing_summary(chunks: list[Any], *, excerpt_chars: int = 3600) -> str:
    """Produce a useful no-LLM filing digest from saved metadata and source excerpts."""
    first = chunks[0].metadata
    periods = _unique_strings(
        period.get("label", "")
        for chunk in chunks
        for period in chunk.metadata.get("reporting_periods", [])
        if isinstance(period, dict)
    )
    sections = _unique_strings((chunk.metadata.get("section", "") for chunk in chunks), limit=20)
    keywords = _unique_strings(
        (keyword for chunk in chunks for keyword in chunk.metadata.get("keywords", [])),
        limit=30,
    )
    excerpts = _filing_source_text(chunks, char_limit=excerpt_chars)
    lines = [
        f"Company: {first.get('company_name') or first.get('ticker', 'unknown')}",
        f"Filing: {first.get('form_type', 'unknown')} filed {first.get('filing_date', 'unknown')}",
    ]
    if periods:
        lines.append("Reporting period(s): " + "; ".join(periods))
    if sections:
        lines.append("Covered sections: " + "; ".join(sections))
    if keywords:
        lines.append("Key topics: " + ", ".join(keywords))
    if excerpts:
        lines.extend(("Source evidence excerpts:", excerpts))
    return "\n".join(lines)


def _remove_existing_filing_summaries(mem: Memory, namespace: str, filing_nodes: set[str]) -> int:
    """Replace summary artifacts for rebuilt filings while leaving raw source episodes append-only."""
    removed = 0
    for episode in list(mem.episodes_doc.values()):
        if (
            episode.user_id == namespace
            and _artifact_type(episode) == "filing_summary"
            and episode.session_id in filing_nodes
        ):
            mem.episodes_doc.delete(episode.id)
            mem.episodes_vec.delete(episode.id)
            mem.summary_vec.delete(episode.id)
            removed += 1
    return removed


def create_filing_summaries(
    mem: Memory,
    seen_facts: set[tuple[str, str, str]],
    *,
    namespace: str,
    filing_nodes: Iterable[str],
    llm: LLM | None,
    use_llm: bool,
    input_chars: int,
) -> dict[str, int]:
    """Persist one summary episode per filing, linked to the exact source chunk IDs it represents."""
    nodes = set(filing_nodes)
    if not nodes:
        return {
            "filing_summaries_added": 0,
            "llm_filing_summaries": 0,
            "replaced_filing_summaries": 0,
            "filing_summary_facts_added": 0,
        }

    raw_by_filing: dict[str, list[Any]] = defaultdict(list)
    for episode in mem.episodes_doc.values():
        if (
            episode.user_id == namespace
            and _artifact_type(episode) == "filing_chunk"
            and episode.session_id in nodes
        ):
            raw_by_filing[episode.session_id].append(episode)

    replaced = _remove_existing_filing_summaries(mem, namespace, nodes)
    added = 0
    llm_created = 0
    facts_added = 0
    for filing_node in sorted(raw_by_filing):
        chunks = raw_by_filing[filing_node]
        if not chunks:
            continue
        first = chunks[0].metadata
        source_text = _filing_source_text(chunks, char_limit=input_chars)
        summary = _deterministic_filing_summary(chunks)
        summary_source = "deterministic"
        if use_llm and llm is not None:
            prompt = (
                "Filing metadata:\n"
                f"- company: {first.get('company_name') or first.get('ticker', 'unknown')}\n"
                f"- ticker: {first.get('ticker', 'unknown')}\n"
                f"- form: {first.get('form_type', 'unknown')}\n"
                f"- filing date: {first.get('filing_date', 'unknown')}\n"
                f"- reporting periods: {json.dumps(first.get('reporting_periods', []), ensure_ascii=False)}\n\n"
                f"Filing excerpts:\n{source_text}\n\nFiling-level summary:"
            )
            try:
                generated = llm.complete(prompt, system=FIN_RATE_FILING_SUMMARY_SYSTEM, num_predict=1024)
                if generated.strip():
                    summary = generated.strip()
                    summary_source = "llm"
                    llm_created += 1
            except Exception:  # noqa: BLE001 -- a single filing must never abort the corpus build
                summary_source = "deterministic_llm_fallback"

        source_episode_ids = [chunk.id for chunk in chunks]
        source_document_ids = _unique_strings(
            chunk.metadata.get("doc_id", "") for chunk in chunks
        )
        periods = [
            period
            for chunk in chunks
            for period in chunk.metadata.get("reporting_periods", [])
            if isinstance(period, dict)
        ]
        keywords = _unique_strings(
            (keyword for chunk in chunks for keyword in chunk.metadata.get("keywords", [])),
            limit=40,
        )
        summary_content = f"FILING-LEVEL SUMMARY\n{summary}"
        episode = mem.add(
            summary_content,
            user_id=namespace,
            session_id=filing_node,
            speaker="filing_summary",
            event_time=float(first.get("source_event_time", chunks[0].event_time)),
        )
        episode.consolidated = True
        episode.summary = summary
        episode.summary_embedding = mem.embedder.embed(summary)
        mem.summary_vec.upsert(episode.id, episode.summary_embedding, episode)
        episode.metadata.update(
            {
                "artifact_type": "filing_summary",
                "filing_summary_node": f"Filing summary | {filing_node}",
                "ticker": first.get("ticker", ""),
                "company_name": first.get("company_name", ""),
                "document_type": first.get("document_type", first.get("form_type", "")),
                "form_type": first.get("form_type", ""),
                "filing_date": first.get("filing_date", ""),
                "source_event_time": first.get("source_event_time", chunks[0].event_time),
                "ingested_at": episode.ingested_at,
                "ingested_at_utc": datetime.fromtimestamp(
                    episode.ingested_at, tz=timezone.utc
                ).isoformat(),
                "reporting_periods": [period for period in periods],
                "keywords": keywords,
                "source_episode_ids": source_episode_ids,
                "source_document_ids": source_document_ids,
                "source_chunk_count": len(source_episode_ids),
                "summary_source": summary_source,
                "embedding_model": getattr(
                    mem.embedder,
                    "model_name",
                    mem.embedder.__class__.__name__,
                ),
                "embedding_dimensions": len(episode.embedding or []),
                "summary_embedding_dimensions": len(episode.summary_embedding or []),
            }
        )
        summary_node = str(episode.metadata["filing_summary_node"])
        provenance = source_episode_ids[:1]
        if add_metadata_fact(
            mem,
            seen_facts,
            user_id=namespace,
            subject=filing_node,
            predicate="has_filing_summary",
            object_=summary_node,
            valid_at=episode.event_time,
            provenance=provenance,
        ):
            facts_added += 1
        if add_metadata_fact(
            mem,
            seen_facts,
            user_id=namespace,
            subject=summary_node,
            predicate="summarizes_filing",
            object_=filing_node,
            valid_at=episode.event_time,
            provenance=provenance,
        ):
            facts_added += 1
        added += 1
    return {
        "filing_summaries_added": added,
        "llm_filing_summaries": llm_created,
        "replaced_filing_summaries": replaced,
        "filing_summary_facts_added": facts_added,
    }


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
    llm_document_metadata: bool = False,
    metadata_llm_text_chars: int = 6000,
    filing_summaries: bool = True,
    llm_filing_summaries: bool = False,
    filing_summary_input_chars: int = 24000,
    ingestion_timings_path: Path | None = None,
    document_ids: Iterable[str] | None = None,
    progress_every: int = 25,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    seed_everything(seed)
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if isinstance(llm, OllamaLLM):
        llm.seed = seed
    corpus_path = corpus_path.resolve()
    store_dir = store_dir.resolve()
    target_document_ids = (
        {str(document_id).strip() for document_id in document_ids if str(document_id).strip()}
        if document_ids is not None
        else None
    )
    matched_document_ids: set[str] = set()
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    if reset and store_dir.exists():
        shutil.rmtree(store_dir)

    timing_path = (ingestion_timings_path or store_dir / "ingestion_timings.jsonl").resolve()
    timing_writer = JsonlTimingWriter(timing_path)
    timing_run_id = f"ingest-{uuid.uuid4().hex}"
    started_at = _utc_now_iso()
    start = time.perf_counter()

    print("[Engram build] starting raw document ingestion", flush=True)

    mem = open_memory(
        store_dir,
        llm=llm,
        embedder=embedder,
        fin_rate_prompts=llm is not None,
    )
    seen_facts = _existing_fact_keys(mem)
    existing_document_ids = {
        str(episode.metadata.get("doc_id", "")).strip()
        for episode in mem.episodes_doc.values()
        if (
            episode.user_id == namespace
            and _artifact_type(episode) == "filing_chunk"
            and str(episode.metadata.get("doc_id", "")).strip()
        )
    }
    if existing_document_ids:
        print(
            "[Engram build] resuming from checkpoint "
            f"({len(existing_document_ids)} documents already saved)",
            flush=True,
        )

    records_read = 0
    documents_selected = 0
    documents_seen = 0
    documents_skipped_existing = 0
    chunks_added = 0
    metadata_facts_added = 0
    deep_relationship_facts_added = 0
    llm_metadata_documents = 0
    document_step_totals_ms: defaultdict[str, float] = defaultdict(float)
    new_episodes = []
    metas: list[FilingMeta] = []
    doc_provenance: dict[str, list[str]] = {}
    filing_company_names: dict[str, str] = {}
    selected_document_ids: set[str] = set()

    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue

            records_read += 1
            record = json.loads(line)
            doc_id = _doc_id(record, line_number)
            if target_document_ids is not None and doc_id not in target_document_ids:
                continue
            if limit is not None and documents_selected >= limit:
                break
            documents_selected += 1
            selected_document_ids.add(doc_id)
            if doc_id in existing_document_ids:
                matched_document_ids.add(doc_id)
                documents_skipped_existing += 1
                continue

            document_started = time.perf_counter()
            document_steps_ms: dict[str, float] = {}
            step_started = time.perf_counter()
            meta = parse_filing_meta(record, line_number)
            text = _record_text(record)
            document_steps_ms["parse_document"] = (time.perf_counter() - step_started) * 1000
            matched_document_ids.add(doc_id)

            step_started = time.perf_counter()
            document_metadata = extract_document_temporal_metadata(
                record,
                meta,
                text,
                ingestion_time=time.time(),
                llm=llm if llm_document_metadata else None,
                llm_text_chars=metadata_llm_text_chars,
                known_company_name=filing_company_names.get(meta.filing_node, ""),
            )
            document_steps_ms["metadata_extraction"] = (time.perf_counter() - step_started) * 1000
            if document_metadata.company_name and document_metadata.company_name != meta.ticker:
                filing_company_names.setdefault(meta.filing_node, document_metadata.company_name)
            documents_seen += 1
            metas.append(meta)
            doc_episode_ids: list[str] = []
            if document_metadata.source == "llm_enriched":
                llm_metadata_documents += 1

            step_started = time.perf_counter()
            chunk_windows = list(
                _token_windows(
                    text,
                    max_tokens=chunk_size_tokens,
                    overlap_tokens=chunk_overlap_tokens,
                )
            )
            document_steps_ms["chunking"] = (time.perf_counter() - step_started) * 1000

            episode_ingestion_ms = 0.0
            episode_metadata_ms = 0.0
            for chunk_index, (chunk_text, token_start, token_end) in enumerate(chunk_windows):
                content_parts = [_document_metadata_context(document_metadata)]
                if meta.title:
                    content_parts.insert(0, meta.title)
                content_parts.append(chunk_text)
                content = "\n\n".join(content_parts)
                step_started = time.perf_counter()
                ep = mem.add(
                    content,
                    user_id=namespace,
                    session_id=meta.filing_node,
                    speaker="filing",
                    event_time=meta.event_time,
                )
                episode_ingestion_ms += (time.perf_counter() - step_started) * 1000

                step_started = time.perf_counter()
                ep.metadata.update(
                    {
                        "artifact_type": "filing_chunk",
                        "doc_id": meta.doc_id,
                        "title": meta.title,
                        "ticker": meta.ticker,
                        "company_name": document_metadata.company_name,
                        "document_type": document_metadata.document_type,
                        "filing_date": meta.filing_date,
                        "source_event_time": meta.event_time,
                        "ingested_at": document_metadata.ingestion_time,
                        "ingested_at_utc": document_metadata.ingestion_time_utc,
                        "form_type": meta.form_type,
                        "section": meta.section,
                        "section_family": meta.section_family,
                        "comparison_group": meta.comparison_group,
                        "reporting_periods": [
                            period.as_dict() for period in document_metadata.reporting_periods
                        ],
                        "reporting_period": (
                            document_metadata.primary_period.label
                            if document_metadata.primary_period is not None
                            else "unknown"
                        ),
                        "reporting_period_start": (
                            document_metadata.primary_period.start_date
                            if document_metadata.primary_period is not None
                            else ""
                        ),
                        "reporting_period_end": (
                            document_metadata.primary_period.end_date
                            if document_metadata.primary_period is not None
                            else ""
                        ),
                        "keywords": list(document_metadata.keywords),
                        "metadata_extraction": document_metadata.source,
                        "embedding_model": getattr(
                            mem.embedder,
                            "model_name",
                            mem.embedder.__class__.__name__,
                        ),
                        "embedding_dimensions": len(ep.embedding or []),
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
                episode_metadata_ms += (time.perf_counter() - step_started) * 1000
            document_steps_ms["episode_ingestion_and_embedding"] = episode_ingestion_ms
            document_steps_ms["episode_metadata"] = episode_metadata_ms

            step_started = time.perf_counter()
            doc_provenance[meta.doc_id] = doc_episode_ids
            provenance = doc_episode_ids[:1]
            metadata_edges = [
                (meta.ticker, "has_filing", meta.filing_node),
                (meta.filing_node, "has_form_type", meta.form_type),
                (meta.filing_node, "has_filing_date", meta.filing_date),
                (meta.filing_node, "has_year", meta.year),
                (meta.filing_node, "has_company", meta.ticker),
                (meta.filing_node, "has_company_name", document_metadata.company_name),
                (meta.filing_node, "has_document_type", document_metadata.document_type),
                (meta.filing_node, "has_section", meta.section_node),
                (meta.section_node, "has_section_family", meta.section_family),
                (meta.section_node, "has_document", meta.doc_id),
                (meta.doc_id, "from_company", document_metadata.company_name),
                (meta.doc_id, "has_document_type", document_metadata.document_type),
                (meta.doc_id, "was_ingested_at", document_metadata.ingestion_time_utc),
            ]
            for period in document_metadata.reporting_periods:
                metadata_edges.append((meta.doc_id, "describes_reporting_period", period.label))
                if period.start_date:
                    metadata_edges.append((meta.doc_id, "has_reporting_period_start", period.start_date))
                if period.end_date:
                    metadata_edges.append((meta.doc_id, "has_reporting_period_end", period.end_date))
            for keyword in document_metadata.keywords:
                metadata_edges.append((meta.doc_id, "has_keyword", keyword))
            for subject, predicate, object_ in metadata_edges:
                if add_metadata_fact(
                    mem,
                    seen_facts,
                    user_id=namespace,
                    subject=subject,
                    predicate=predicate,
                    object_=object_,
                    # The ingestion timestamp is transaction-time metadata. It is true only when we
                    # recorded the document; all other document metadata is valid at the filing event.
                    valid_at=(
                        document_metadata.ingestion_time
                        if predicate == "was_ingested_at"
                        else meta.event_time
                    ),
                    provenance=provenance,
                ):
                    metadata_facts_added += 1
            document_steps_ms["graph_metadata"] = (time.perf_counter() - step_started) * 1000

            document_steps_ms["total"] = (time.perf_counter() - document_started) * 1000
            for step_name, duration_ms in document_steps_ms.items():
                document_step_totals_ms[step_name] += duration_ms
            timing_writer.write(
                {
                    "record_type": "document",
                    "run_id": timing_run_id,
                    "recorded_at_utc": _utc_now_iso(),
                    "document_number": documents_seen,
                    "doc_id": meta.doc_id,
                    "ticker": meta.ticker,
                    "company_name": document_metadata.company_name,
                    "filing_date": meta.filing_date,
                    "reporting_periods": [
                        period.as_dict() for period in document_metadata.reporting_periods
                    ],
                    "chunks_added": len(doc_episode_ids),
                    "durations_ms": {
                        step_name: round(duration_ms, 3)
                        for step_name, duration_ms in document_steps_ms.items()
                    },
                }
            )

            if progress_every and documents_seen % progress_every == 0:
                print(
                    "[Engram build] completed document ingestion progress "
                    f"({documents_seen} new docs, {documents_skipped_existing} resumed docs, "
                    f"{chunks_added} chunks, {metadata_facts_added} metadata facts)",
                    flush=True,
                )

            if (
                target_document_ids is not None
                and matched_document_ids == target_document_ids
            ):
                break

    batch_durations_ms: dict[str, float] = {}
    raw_ingestion_duration_ms = (time.perf_counter() - start) * 1000
    _print_build_stage(
        "raw document ingestion",
        raw_ingestion_duration_ms,
        documents_ingested=documents_seen,
        documents_resumed=documents_skipped_existing,
        chunks_added=chunks_added,
    )
    step_started = time.perf_counter()
    if deep_relationships and metas:
        deep_relationship_facts_added = add_deep_relationships(
            mem,
            seen_facts,
            metas=metas,
            doc_provenance=doc_provenance,
            namespace=namespace,
        )
    batch_durations_ms["deep_relationships"] = (time.perf_counter() - step_started) * 1000
    _print_build_stage(
        "deep relationship creation",
        batch_durations_ms["deep_relationships"],
        facts_added=deep_relationship_facts_added,
    )

    extracted_facts = 0
    step_started = time.perf_counter()
    if extract_text_facts and new_episodes:
        extracted_facts = mem.consolidate(new_episodes).get("facts_added", 0)
    batch_durations_ms["fact_extraction"] = (time.perf_counter() - step_started) * 1000
    _print_build_stage(
        "text fact extraction",
        batch_durations_ms["fact_extraction"],
        facts_added=extracted_facts,
    )

    raw_scope_episodes = [
        episode
        for episode in mem.episodes_doc.values()
        if (
            episode.user_id == namespace
            and _artifact_type(episode) == "filing_chunk"
            and str(episode.metadata.get("doc_id", "")) in selected_document_ids
        )
    ]
    pending_summary_episodes = [episode for episode in raw_scope_episodes if not episode.summary]

    # A raw-ingestion checkpoint makes summary failures recoverable.  Re-run with
    # reset=False and the existing documents are skipped above; only the pending
    # summary and filing-summary stages are repeated.
    step_started = time.perf_counter()
    print(
        "[Engram build] saving raw-ingestion checkpoint; do not interrupt until its completion message",
        flush=True,
    )
    mem.save(str(store_dir))
    batch_durations_ms["raw_ingestion_checkpoint_save"] = (time.perf_counter() - step_started) * 1000
    _print_build_stage(
        "raw-ingestion checkpoint save",
        batch_durations_ms["raw_ingestion_checkpoint_save"],
        raw_chunk_episodes=len(raw_scope_episodes),
        pending_summaries=len(pending_summary_episodes),
    )
    print(
        "[Engram build] raw-ingestion checkpoint is complete — safe to pause now. "
        "Rerun the same build without --reset to continue with summaries.",
        flush=True,
    )
    timing_writer.write(
        {
            "record_type": "checkpoint",
            "run_id": timing_run_id,
            "recorded_at_utc": _utc_now_iso(),
            "stage": "raw_ingestion_complete",
            "documents_selected": documents_selected,
            "documents_ingested": documents_seen,
            "documents_skipped_existing": documents_skipped_existing,
            "raw_chunk_episodes": len(raw_scope_episodes),
            "pending_summary_episodes": len(pending_summary_episodes),
        }
    )

    step_started = time.perf_counter()
    summary_llm = getattr(mem.summarizer, "llm", None)
    if not llm_summaries:
        mem.summarizer.llm = None
    summaries = mem.summarize_episodes(pending_summary_episodes) if summarize else 0
    mem.summarizer.llm = summary_llm
    for episode in pending_summary_episodes:
        # Episode embeddings and summary embeddings are persisted by Engram itself. Recording the
        # dimensions and summary in metadata makes each Graph RAG artifact inspectable without decoding
        # a vector, while keeping the actual vectors in their native persisted fields.
        episode.metadata["embedding_dimensions"] = len(episode.embedding or [])
        if episode.summary:
            episode.metadata["summary"] = episode.summary
            episode.metadata["summary_embedding_dimensions"] = len(episode.summary_embedding or [])
    batch_durations_ms["summary_indexing"] = (time.perf_counter() - step_started) * 1000
    _print_build_stage(
        "chunk summary indexing",
        batch_durations_ms["summary_indexing"],
        summaries_added=summaries,
    )

    step_started = time.perf_counter()
    print(
        "[Engram build] saving summary-index checkpoint; do not interrupt until its completion message",
        flush=True,
    )
    mem.save(str(store_dir))
    batch_durations_ms["summary_index_checkpoint_save"] = (time.perf_counter() - step_started) * 1000
    _print_build_stage(
        "summary-index checkpoint save",
        batch_durations_ms["summary_index_checkpoint_save"],
    )
    print(
        "[Engram build] summary-index checkpoint is complete — safe to pause now. "
        "Rerun the same build without --reset to continue with filing summaries.",
        flush=True,
    )
    timing_writer.write(
        {
            "record_type": "checkpoint",
            "run_id": timing_run_id,
            "recorded_at_utc": _utc_now_iso(),
            "stage": "summary_indexing_complete",
            "raw_chunk_episodes": len(raw_scope_episodes),
            "summaries_added": summaries,
        }
    )

    step_started = time.perf_counter()
    filing_summary_stats = (
        create_filing_summaries(
            mem,
            seen_facts,
            namespace=namespace,
            filing_nodes=(episode.session_id for episode in raw_scope_episodes),
            llm=llm,
            use_llm=llm_filing_summaries,
            input_chars=filing_summary_input_chars,
        )
        if filing_summaries
        else {
            "filing_summaries_added": 0,
            "llm_filing_summaries": 0,
            "replaced_filing_summaries": 0,
            "filing_summary_facts_added": 0,
        }
    )
    batch_durations_ms["filing_summary_indexing"] = (time.perf_counter() - step_started) * 1000
    _print_build_stage(
        "filing-summary indexing",
        batch_durations_ms["filing_summary_indexing"],
        filing_summaries_added=filing_summary_stats["filing_summaries_added"],
    )

    step_started = time.perf_counter()
    print(
        "[Engram build] saving final database; do not interrupt until its completion message",
        flush=True,
    )
    mem.save(str(store_dir))
    batch_durations_ms["store_save"] = (time.perf_counter() - step_started) * 1000
    _print_build_stage("final database save", batch_durations_ms["store_save"])
    print(
        "[Engram build] final database save is complete — the store is ready for querying.",
        flush=True,
    )

    document_step_averages_ms = {
        step_name: total_ms / documents_seen
        for step_name, total_ms in document_step_totals_ms.items()
    } if documents_seen else {}
    build_finished_at = _utc_now_iso()
    build_seconds = time.perf_counter() - start
    timing_writer.write(
        {
            "record_type": "build_summary",
            "run_id": timing_run_id,
            "recorded_at_utc": build_finished_at,
            "records_read": records_read,
            "documents_selected": documents_selected,
            "documents_seen": documents_seen,
            "documents_skipped_existing": documents_skipped_existing,
            "document_ids_filter_count": len(target_document_ids or ()),
            "document_ids_matched_count": len(matched_document_ids),
            "document_ids_missing": (
                sorted(target_document_ids - matched_document_ids)
                if target_document_ids is not None
                else []
            ),
            "chunks_added": chunks_added,
            "filing_summary_stats": filing_summary_stats,
            "document_step_average_ms": {
                step_name: round(duration_ms, 3)
                for step_name, duration_ms in document_step_averages_ms.items()
            },
            "document_step_total_ms": {
                step_name: round(duration_ms, 3)
                for step_name, duration_ms in document_step_totals_ms.items()
            },
            "batch_durations_ms": {
                step_name: round(duration_ms, 3)
                for step_name, duration_ms in batch_durations_ms.items()
            },
            "total_build_ms": round(build_seconds * 1000, 3),
        }
    )
    timing_writer.close()

    stats = {
        "corpus_path": str(corpus_path),
        "store_dir": str(store_dir),
        "namespace": namespace,
        "records_read": records_read,
        "documents_selected": documents_selected,
        "documents_seen": documents_seen,
        "documents_skipped_existing": documents_skipped_existing,
        "document_ids_filter": sorted(target_document_ids) if target_document_ids is not None else None,
        "document_ids_matched": sorted(matched_document_ids),
        "document_ids_missing": (
            sorted(target_document_ids - matched_document_ids)
            if target_document_ids is not None
            else []
        ),
        "chunks_added": chunks_added,
        "metadata_facts_added": metadata_facts_added,
        "llm_metadata_documents": llm_metadata_documents,
        "deep_relationship_facts_added": deep_relationship_facts_added,
        "text_facts_extracted": extracted_facts,
        "summaries_added": summaries,
        **filing_summary_stats,
        "llm_configured": llm is not None,
        "llm_summaries": bool(llm is not None and llm_summaries),
        "llm_document_metadata": bool(llm is not None and llm_document_metadata),
        "llm_filing_summaries_enabled": bool(llm is not None and llm_filing_summaries),
        "seed": seed,
        "embedder": getattr(embedder, "model_name", embedder.__class__.__name__ if embedder else "hashing"),
        "graph_nodes": len(mem.graph.entities),
        "graph_edges": len(mem.graph.relations()),
        "ingestion_timings_path": str(timing_path),
        "document_step_average_ms": {
            step_name: round(duration_ms, 3)
            for step_name, duration_ms in document_step_averages_ms.items()
        },
        "batch_durations_ms": {
            step_name: round(duration_ms, 3)
            for step_name, duration_ms in batch_durations_ms.items()
        },
        "build_started_at_utc": started_at,
        "build_finished_at_utc": build_finished_at,
        "build_seconds": build_seconds,
        "local_only": True,
    }
    (store_dir / "fin_rate_build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def _normalise_company_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _add_company_alias_variants(
    aliases: dict[str, set[str]],
    *,
    ticker: str,
    raw_alias: str,
    include_first_word: bool,
) -> None:
    """Add a normalized company alias and its legal-suffix-free variant."""
    alias = _normalise_company_alias(raw_alias)
    if not alias:
        return
    aliases[alias].add(ticker)
    suffixes = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "plc"}
    words = alias.split()
    while words and words[-1] in suffixes:
        words.pop()
    if not words:
        return
    aliases[" ".join(words)].add(ticker)
    if include_first_word and len(words[0]) >= 4:
        # Retained for the original metadata-only resolver's backward compatibility.
        aliases[words[0]].add(ticker)


def _company_catalog(episodes: Iterable[Any], namespace: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Build queryable company aliases from the persisted document metadata."""
    aliases: dict[str, set[str]] = defaultdict(set)
    names_by_ticker: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        if episode.user_id != namespace:
            continue
        metadata = episode.metadata
        ticker = str(metadata.get("ticker", "")).upper().strip()
        if not ticker or ticker == "UNKNOWN":
            continue
        company_name = str(metadata.get("company_name", "")).strip()
        names_by_ticker[ticker].add(company_name or ticker)
        for raw_alias in (ticker, company_name):
            _add_company_alias_variants(
                aliases,
                ticker=ticker,
                raw_alias=raw_alias,
                include_first_word=True,
            )
    return aliases, names_by_ticker


@dataclass(frozen=True)
class QueryTimeCompanyNameResolution:
    """A conservative, query-only mapping from LLM-extracted names to stored company evidence."""

    company_names: tuple[str, ...]
    company_tickers: tuple[str, ...]
    matched_company_names: tuple[str, ...]
    unresolved_company_names: tuple[str, ...]
    ambiguous_company_names: tuple[str, ...]

    @property
    def status(self) -> str:
        if not self.company_names:
            return "llm_no_company_extracted"
        if self.company_tickers and not (self.unresolved_company_names or self.ambiguous_company_names):
            return "llm_content_exact"
        if self.company_tickers:
            return "llm_content_partial"
        if self.ambiguous_company_names:
            return "llm_content_ambiguous"
        return "llm_content_unresolved"


def extract_query_company_names_with_llm(query: str, llm: LLM) -> tuple[str, ...]:
    """Ask the LLM only for companies explicitly named as the subject of the query."""
    extracted = _parse_json_object(
        llm.complete(
            f"Query:\n{query}\n\nCompany names JSON:",
            system=FIN_RATE_QUERY_COMPANY_NAME_SYSTEM,
            num_predict=256,
        )
    )
    values = extracted.get("company_names")
    if not isinstance(values, list):
        return ()
    company_names: list[str] = []
    seen_company_names: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        company_name = _normalise_space(value).strip(" \t\r\n|*_:-.,;")[:200]
        if not company_name or _normalise_company_alias(company_name) in QUERY_TIME_GENERIC_COMPANY_NAMES:
            continue
        lower_name = company_name.lower()
        if lower_name not in seen_company_names:
            company_names.append(company_name)
            seen_company_names.add(lower_name)
    return tuple(company_names)


def _company_name_search_variants(company_name: str) -> tuple[str, ...]:
    """Make only exact legal-name and legal-suffix-free variants; never create nickname aliases."""
    normalised = _normalise_company_alias(company_name)
    if not normalised:
        return ()
    variants = [normalised]
    suffixes = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited", "plc"}
    words = normalised.split()
    while len(words) > 1 and words[-1] in suffixes:
        words.pop()
    suffix_free = " ".join(words)
    if suffix_free and suffix_free not in variants:
        variants.append(suffix_free)
    return tuple(variants)


def resolve_llm_company_names_in_existing_chunks(
    company_names: Iterable[str],
    episodes: Iterable[Any],
    namespace: str,
) -> QueryTimeCompanyNameResolution:
    """Map LLM-extracted full names to stored companies using only exact raw-content evidence.

    A name is applied only if its legal-name phrase occurs in raw filing chunks for one ticker. Multiple
    candidate tickers are reported as ambiguous rather than guessed. No metadata is written or rebuilt.
    """
    names = tuple(dict.fromkeys(name for name in company_names if name))
    variants_by_name = {name: _company_name_search_variants(name) for name in names}
    candidate_tickers: dict[str, set[str]] = {name: set() for name in names}

    for episode in episodes:
        if episode.user_id != namespace or _artifact_type(episode) != "filing_chunk":
            continue
        ticker = str(episode.metadata.get("ticker", "")).upper().strip()
        if not ticker or ticker == "UNKNOWN":
            continue
        content = _normalise_company_alias(str(episode.content or ""))
        padded_content = f" {content} "
        for name, variants in variants_by_name.items():
            matches = {variant for variant in variants if f" {variant} " in padded_content}
            if matches:
                candidate_tickers[name].add(ticker)

    resolved_tickers: set[str] = set()
    matched_names: list[str] = []
    unresolved_names: list[str] = []
    ambiguous_names: list[str] = []
    for name in names:
        tickers = candidate_tickers[name]
        if len(tickers) == 1:
            resolved_tickers.update(tickers)
            matched_names.append(name)
        elif len(tickers) > 1:
            ambiguous_names.append(name)
        else:
            unresolved_names.append(name)

    return QueryTimeCompanyNameResolution(
        company_names=names,
        company_tickers=tuple(sorted(resolved_tickers)),
        matched_company_names=tuple(matched_names),
        unresolved_company_names=tuple(unresolved_names),
        ambiguous_company_names=tuple(ambiguous_names),
    )


def _matching_company_aliases(text: str, aliases: dict[str, set[str]]) -> set[str]:
    """Return normalized aliases explicitly present in text, longest first for stable inspection."""
    normalised = _normalise_company_alias(text)
    padded = f" {normalised} "
    return {
        alias
        for alias in aliases
        if alias and f" {alias} " in padded
    }


def _matching_company_tickers(text: str, aliases: dict[str, set[str]]) -> set[str]:
    matched: set[str] = set()
    for alias in _matching_company_aliases(text, aliases):
        matched.update(aliases[alias])
    return matched


def _valid_query_years(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    years = [str(item) for item in value if YEAR_RE.fullmatch(str(item).strip())]
    return tuple(sorted(set(years)))


def extract_query_filters(
    query: str,
    episodes: Iterable[Any],
    namespace: str,
    *,
    llm: LLM | None = None,
    llm_query_filters: bool = False,
    llm_query_time_company_resolver: bool = False,
    timings_ms: dict[str, float] | None = None,
) -> QueryFilters:
    """Identify company and year terms before metadata-constrained RAG retrieval.

    The original path uses persisted metadata aliases. The opt-in LLM path extracts only the company
    named by the user, then requires exact legal-name evidence in existing raw chunks before it filters.
    Neither path writes to or rebuilds the store.
    """
    aliases, names_by_ticker = _company_catalog(episodes, namespace)
    years = tuple(sorted(set(YEAR_RE.findall(query))))
    extracted_company_names: tuple[str, ...] = ()
    matched_company_names: tuple[str, ...] = ()

    if llm_query_time_company_resolver:
        if llm is None:
            raise ValueError("llm_query_time_company_resolver requires an LLM")
        step_started = time.perf_counter()
        extracted_company_names = extract_query_company_names_with_llm(query, llm)
        if timings_ms is not None:
            timings_ms["llm_company_name_extraction"] = (time.perf_counter() - step_started) * 1000
        step_started = time.perf_counter()
        name_resolution = resolve_llm_company_names_in_existing_chunks(
            extracted_company_names,
            episodes,
            namespace,
        )
        if timings_ms is not None:
            timings_ms["llm_company_content_resolution"] = (time.perf_counter() - step_started) * 1000
        tickers = set(name_resolution.company_tickers)
        matched_company_names = name_resolution.matched_company_names
        source = "llm_company_name_content_resolver"
        company_resolution = name_resolution.status
        company_names = name_resolution.company_names
    else:
        if timings_ms is not None:
            timings_ms["llm_company_name_extraction"] = 0.0
            timings_ms["llm_company_content_resolution"] = 0.0
        tickers = _matching_company_tickers(query, aliases)
        source = "deterministic"
        company_resolution = "metadata_alias" if tickers else "unresolved"
        company_names = ()

    if llm_query_filters:
        if llm is None:
            raise ValueError("llm_query_filters requires an LLM")
        try:
            enriched = _parse_json_object(
                llm.complete(
                    f"Query:\n{query}\n\nFilters JSON:",
                    system=FIN_RATE_QUERY_FILTER_SYSTEM,
                    num_predict=256,
                )
            )
            for company_name in enriched.get("company_names", []):
                if isinstance(company_name, str) and not llm_query_time_company_resolver:
                    resolved_tickers = _matching_company_tickers(company_name, aliases)
                    if resolved_tickers - tickers:
                        company_resolution = "llm_alias"
                    tickers.update(resolved_tickers)
            llm_years = _valid_query_years(enriched.get("years"))
            if llm_years:
                years = tuple(sorted(set((*years, *llm_years))))
            source += "+llm_years"
        except Exception:  # noqa: BLE001 -- a query should still run without the optional enrich step
            source += "+llm_filter_fallback"

    relative = resolve_relative_time_window(query, episodes, namespace, tickers)
    if relative is not None:
        relative_years, start_date, end_date, expression, reference = relative
        years = tuple(sorted(set((*years, *relative_years))))
        source += "+relative_time"
    else:
        start_date = ""
        end_date = ""
        expression = ""
        reference = ""

    if not company_names:
        company_names = tuple(
            sorted(
                {
                    sorted(names_by_ticker[ticker])[0] if names_by_ticker[ticker] else ticker
                    for ticker in tickers
                }
            )
        )
    return QueryFilters(
        company_tickers=tuple(sorted(tickers)),
        company_names=company_names,
        years=years,
        source=source,
        start_date=start_date,
        end_date=end_date,
        relative_time_expression=expression,
        relative_time_reference=reference,
        company_resolution=company_resolution,
        company_aliases_matched=tuple(sorted(matched_company_names)),
    )


def _episode_years(episode: Any) -> set[str]:
    """Read filing and reporting-period years from old and new persisted episode metadata."""
    metadata = episode.metadata
    values: list[str] = [
        str(metadata.get("filing_date", "")),
        str(metadata.get("reporting_period", "")),
        str(metadata.get("reporting_period_start", "")),
        str(metadata.get("reporting_period_end", "")),
    ]
    for period in metadata.get("reporting_periods", []):
        if isinstance(period, dict):
            values.extend(str(period.get(key, "")) for key in ("label", "start_date", "end_date"))
    return {year for value in values for year in YEAR_RE.findall(value)}


def _episode_temporal_dates(episode: Any) -> set[str]:
    """Return ISO filing/reporting endpoints used for metadata time-window filtering."""
    metadata = episode.metadata
    values = [str(metadata.get("filing_date", ""))]
    return {value for value in values if DATE_RE.match(value)} | _episode_reporting_dates(episode)


def _episode_reporting_dates(episode: Any) -> set[str]:
    """Return explicit period endpoints, excluding the administrative filing date."""
    metadata = episode.metadata
    values = [
        str(metadata.get("reporting_period_start", "")),
        str(metadata.get("reporting_period_end", "")),
    ]
    for period in metadata.get("reporting_periods", []):
        if isinstance(period, dict):
            values.extend(str(period.get(key, "")) for key in ("start_date", "end_date"))
    return {value for value in values if DATE_RE.match(value)}


def _relative_count(value: str) -> int:
    return int(value) if value.isdigit() else NUMBER_WORDS[value.lower()]


def _shift_iso_date_months(value: str, months: int) -> str:
    current = datetime.strptime(value, "%Y-%m-%d")
    absolute_month = (current.year * 12 + current.month - 1) + months
    year, month_index = divmod(absolute_month, 12)
    month = month_index + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day).date().isoformat()


def _latest_store_date(
    episodes: Iterable[Any],
    namespace: str,
    company_tickers: set[str],
) -> str:
    reporting_dates: set[str] = set()
    filing_dates: set[str] = set()
    for episode in episodes:
        if not (
            episode.user_id == namespace
            and _artifact_type(episode) == "filing_chunk"
            and (
                not company_tickers
                or str(episode.metadata.get("ticker", "")).upper().strip() in company_tickers
            )
        ):
            continue
        reporting_dates.update(_episode_reporting_dates(episode))
        filing_dates.update(
            date
            for date in (str(episode.metadata.get("filing_date", "")),)
            if DATE_RE.match(date)
        )
    # Prefer world-time reporting endpoints. Filing dates are only a fallback for records without periods.
    return max(reporting_dates or filing_dates) if (reporting_dates or filing_dates) else datetime.now(
        timezone.utc
    ).date().isoformat()


def resolve_relative_time_window(
    query: str,
    episodes: Iterable[Any],
    namespace: str,
    company_tickers: set[str],
) -> tuple[tuple[str, ...], str, str, str, str] | None:
    """Translate relative language into a fixed time window anchored to the relevant stored corpus.

    Anchoring on the latest filing/reporting date in the selected company avoids treating an older SEC
    corpus as though it contains material through the computer's wall-clock date. If no matching stored
    date exists, UTC today provides a transparent fallback reference.
    """
    relative_match = RELATIVE_TIME_RE.search(query)
    current_match = CURRENT_TIME_RE.search(query)
    if relative_match is None and current_match is None:
        return None

    reference = _latest_store_date(episodes, namespace, company_tickers)
    reference_year = int(reference[:4])
    if relative_match is not None:
        count = _relative_count(relative_match.group("count"))
        unit = relative_match.group("unit").lower().rstrip("s")
        expression = relative_match.group(0)
    else:
        count = 1
        unit = current_match.group("unit").lower()
        expression = current_match.group(0)

    if unit == "year":
        start_date = f"{reference_year - count + 1:04d}-01-01"
    elif unit == "quarter":
        start_date = _shift_iso_date_months(reference, -(3 * count) + 1)
    else:  # month
        start_date = _shift_iso_date_months(reference, -count + 1)

    years = tuple(str(year) for year in range(int(start_date[:4]), reference_year + 1))
    return years, start_date, reference, expression, reference


def filter_episodes_by_query_metadata(
    episodes: Iterable[Any],
    namespace: str,
    filters: QueryFilters,
    *,
    artifact_type: str | None = None,
) -> list[Any]:
    """Apply company and year constraints before vector retrieval, retaining either listed year."""
    filtered: list[Any] = []
    for episode in episodes:
        if episode.user_id != namespace:
            continue
        if artifact_type is not None and _artifact_type(episode) != artifact_type:
            continue
        metadata = episode.metadata
        ticker = str(metadata.get("ticker", "")).upper().strip()
        if filters.company_tickers and ticker not in filters.company_tickers:
            continue
        if filters.years and not (set(filters.years) & _episode_years(episode)):
            continue
        if filters.start_date and filters.end_date:
            # Relative phrases describe the world-time period, not merely the date the SEC form was filed.
            # Some cover/prologue chunks have no period metadata, so retain filing-date fallback for those.
            dates = _episode_reporting_dates(episode) or _episode_temporal_dates(episode)
            if not dates or not any(filters.start_date <= date <= filters.end_date for date in dates):
                continue
        filtered.append(episode)
    return filtered


def _is_broad_query(query: str, filters: QueryFilters) -> bool:
    """Broad prompts benefit from filing-level coverage, not only the nearest raw chunks."""
    return len(filters.years) >= 2 or bool(BROAD_QUERY_RE.search(query))


def _render_raw_rag_context(retrieved: list[tuple[float, Any]]) -> str:
    blocks: list[str] = []
    for score, episode in retrieved:
        metadata = episode.metadata
        periods = metadata.get("reporting_periods", [])
        period_labels = [
            str(item.get("label", ""))
            for item in periods
            if isinstance(item, dict) and item.get("label")
        ]
        header = (
            f"RAW EVIDENCE | DOCUMENT {metadata.get('doc_id', episode.id)} | "
            f"company={metadata.get('company_name') or metadata.get('ticker', 'unknown')} | "
            f"type={metadata.get('document_type') or metadata.get('form_type', 'unknown')} | "
            f"filing_date={metadata.get('filing_date', 'unknown')} | "
            f"reporting_periods={'; '.join(period_labels) or 'unknown'} | "
            f"score={score:.4f}"
        )
        blocks.append(f"{header}\n{episode.content}")
    return "\n\n".join(blocks)


def _render_filing_summary_context(retrieved: list[tuple[float, Any]]) -> str:
    blocks: list[str] = []
    for score, episode in retrieved:
        metadata = episode.metadata
        blocks.append(
            "FILING SUMMARY (navigation only; use raw evidence for claims) | "
            f"filing={episode.session_id} | company={metadata.get('company_name') or metadata.get('ticker', 'unknown')} | "
            f"filing_date={metadata.get('filing_date', 'unknown')} | score={score:.4f}\n"
            f"{episode.summary or episode.content}"
        )
    return "\n\n".join(blocks)


def _filing_summary_payload(retrieved: list[tuple[float, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "filing_node": episode.session_id,
            "summary_episode_id": episode.id,
            "score": score,
            "ticker": episode.metadata.get("ticker", ""),
            "company_name": episode.metadata.get("company_name", ""),
            "filing_date": episode.metadata.get("filing_date", ""),
            "reporting_periods": episode.metadata.get("reporting_periods", []),
            "source_document_ids": episode.metadata.get("source_document_ids", []),
            "source_chunk_count": episode.metadata.get("source_chunk_count", 0),
        }
        for score, episode in retrieved
    ]


def _summary_source_episode_ids(retrieved: Iterable[tuple[float, Any]]) -> set[str]:
    return {
        str(episode_id)
        for _, summary in retrieved
        for episode_id in summary.metadata.get("source_episode_ids", [])
        if episode_id
    }


def _merge_retrieved_chunks(
    *retrieval_groups: Iterable[tuple[float, Any]],
    limit: int,
) -> list[tuple[float, Any]]:
    """Deduplicate raw evidence while preserving its best similarity score across retrieval routes."""
    best: dict[str, tuple[float, Any]] = {}
    for group in retrieval_groups:
        for score, episode in group:
            prior = best.get(episode.id)
            if prior is None or score > prior[0]:
                best[episode.id] = (score, episode)
    return sorted(best.values(), key=lambda item: (-item[0], item[1].id))[:limit]


def _retrieved_document_payload(retrieved: list[tuple[float, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": episode.metadata.get("doc_id", episode.id),
            "episode_id": episode.id,
            "score": score,
            "ticker": episode.metadata.get("ticker", ""),
            "company_name": episode.metadata.get("company_name", ""),
            "document_type": episode.metadata.get("document_type", episode.metadata.get("form_type", "")),
            "filing_date": episode.metadata.get("filing_date", ""),
            "reporting_periods": episode.metadata.get("reporting_periods", []),
        }
        for score, episode in retrieved
    ]


def _append_query_timing(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as timing_file:
        timing_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


class JsonlTimingWriter:
    """Append build telemetry incrementally without holding a full-corpus timing trace in memory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        self._file.close()


def query_store(
    *,
    store_dir: Path,
    namespace: str,
    query: str,
    top_k: int,
    context_chars: int,
    answer_context_chars: int,
    include_profile_context: bool,
    query_timings_path: Path | None = None,
    query_filters_override: QueryFilters | None = None,
    llm_query_filters: bool = False,
    llm_query_time_company_resolver: bool = False,
    use_filing_summaries: bool = True,
    filing_summary_top_k: int = 4,
    evidence_per_filing_summary: int = 2,
    llm: LLM | None = None,
    embedder: Embedder | None = None,
    generate_answer: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run metadata-filtered raw RAG, adding filing summaries for broad temporal questions.

    Filing summaries are retrieved only after the same company/year filters as raw chunks. Their source
    chunk IDs are then expanded and ranked as evidence, so broad coverage never replaces exact text.
    llm_query_time_company_resolver optionally extracts the named company with an LLM and validates it
    against existing raw chunks without writing to or rebuilding the store. query_filters_override is
    a query-time-only hook for planned retrieval branches that have already resolved their parent
    company and time constraints; it never changes persisted metadata.
    """
    del include_profile_context  # The filtered RAG path intentionally avoids Engram's profile context.
    seed_everything(seed)
    if isinstance(llm, OllamaLLM):
        llm.seed = seed
    timings_ms: dict[str, float] = {}
    total_start = time.perf_counter()
    filters = QueryFilters((), (), (), "not_started")
    all_episodes: list[Any] = []
    raw_candidates: list[Any] = []
    filing_summary_candidates: list[Any] = []
    raw_direct: list[tuple[float, Any]] = []
    filing_summaries: list[tuple[float, Any]] = []
    expanded_evidence: list[tuple[float, Any]] = []
    retrieved: list[tuple[float, Any]] = []
    broad_query = False
    success = False
    error_message: str | None = None

    try:
        step_start = time.perf_counter()
        mem = open_memory(store_dir, embedder=embedder)
        timings_ms["open_store"] = (time.perf_counter() - step_start) * 1000

        all_episodes = mem.episodes_doc.values()
        step_start = time.perf_counter()
        if query_filters_override is None:
            query_filter_timings_ms: dict[str, float] = {}
            filters = extract_query_filters(
                query,
                all_episodes,
                namespace,
                llm=llm,
                llm_query_filters=llm_query_filters,
                llm_query_time_company_resolver=llm_query_time_company_resolver,
                timings_ms=query_filter_timings_ms,
            )
            timings_ms.update(query_filter_timings_ms)
        else:
            filters = query_filters_override
            timings_ms["llm_company_name_extraction"] = 0.0
            timings_ms["llm_company_content_resolution"] = 0.0
        timings_ms["query_decomposition"] = (time.perf_counter() - step_start) * 1000

        step_start = time.perf_counter()
        raw_candidates = filter_episodes_by_query_metadata(
            all_episodes,
            namespace,
            filters,
            artifact_type="filing_chunk",
        )
        filing_summary_candidates = filter_episodes_by_query_metadata(
            all_episodes,
            namespace,
            filters,
            artifact_type="filing_summary",
        )
        broad_query = bool(use_filing_summaries and _is_broad_query(query, filters))
        timings_ms["metadata_filter"] = (time.perf_counter() - step_start) * 1000

        if raw_candidates or (broad_query and filing_summary_candidates):
            step_start = time.perf_counter()
            query_embedding = mem.embedder.embed(query)
            timings_ms["query_embedding"] = (time.perf_counter() - step_start) * 1000

            raw_candidate_ids = {episode.id for episode in raw_candidates}
            step_start = time.perf_counter()
            if raw_candidate_ids:
                raw_direct = mem.episodes_vec.search(
                    query_embedding,
                    top_k=top_k,
                    where=lambda episode: episode.id in raw_candidate_ids,
                )
            timings_ms["raw_vector_retrieval"] = (time.perf_counter() - step_start) * 1000

            step_start = time.perf_counter()
            if broad_query and filing_summary_candidates:
                summary_candidate_ids = {episode.id for episode in filing_summary_candidates}
                filing_summaries = mem.summary_vec.search(
                    query_embedding,
                    top_k=filing_summary_top_k,
                    where=lambda episode: episode.id in summary_candidate_ids,
                )
            timings_ms["filing_summary_retrieval"] = (time.perf_counter() - step_start) * 1000

            step_start = time.perf_counter()
            source_ids = _summary_source_episode_ids(filing_summaries)
            if source_ids:
                expanded_evidence = mem.episodes_vec.search(
                    query_embedding,
                    top_k=max(1, evidence_per_filing_summary * len(filing_summaries)),
                    where=lambda episode: episode.id in source_ids,
                )
            retrieved = _merge_retrieved_chunks(
                raw_direct,
                expanded_evidence,
                limit=top_k + (evidence_per_filing_summary * len(filing_summaries)),
            )
            timings_ms["filing_evidence_expansion"] = (time.perf_counter() - step_start) * 1000
        else:
            timings_ms["query_embedding"] = 0.0
            timings_ms["raw_vector_retrieval"] = 0.0
            timings_ms["filing_summary_retrieval"] = 0.0
            timings_ms["filing_evidence_expansion"] = 0.0

        step_start = time.perf_counter()
        summary_context = _render_filing_summary_context(filing_summaries)
        raw_context = _render_raw_rag_context(retrieved)
        full_context = "\n\n".join(part for part in (summary_context, raw_context) if part)
        context = full_context if context_chars <= 0 else full_context[:context_chars]
        answer_context = full_context if answer_context_chars <= 0 else full_context[:answer_context_chars]
        timings_ms["context_assembly"] = (time.perf_counter() - step_start) * 1000

        payload: dict[str, Any] = {
            "query": query,
            "filters": filters.as_dict(),
            "llm_query_time_company_resolver": llm_query_time_company_resolver,
            "query_scope": "broad" if broad_query else "specific",
            "documents_before_filter": sum(
                episode.user_id == namespace and _artifact_type(episode) == "filing_chunk"
                for episode in all_episodes
            ),
            "documents_after_filter": len(raw_candidates),
            "filing_summaries_after_filter": len(filing_summary_candidates),
            "retrieval_method": (
                "metadata-filtered filing-summary + raw-evidence RAG"
                if filing_summaries
                else "metadata-filtered raw vector RAG"
            ),
            "filing_summaries": _filing_summary_payload(filing_summaries),
            "expanded_evidence_chunks": len(expanded_evidence),
            "retrieved_documents": _retrieved_document_payload(retrieved),
            "context_chars": len(full_context),
            "context_truncated": len(context) < len(full_context),
            "context": context,
        }

        step_start = time.perf_counter()
        if not retrieved:
            payload["answer"] = (
                "No raw evidence documents matched the extracted company and year filters. "
                "Inspect `filters` and the persisted document metadata."
            )
        elif llm is None or not generate_answer:
            payload["answer"] = (
                "Relevant documents were retrieved. Supply --ollama-model to generate a grounded answer "
                "from the returned context."
            )
        else:
            answer_prompt = (
                f"Question:\n{query}\n\nFiltered retrieval context:\n{answer_context}\n\nAnswer:"
            )
            payload["answer"] = llm.complete(
                answer_prompt,
                system=FIN_RATE_ANSWER_SYSTEM,
                num_predict=512,
            )
            payload["answer_context_chars"] = len(answer_context)
            payload["answer_context_truncated"] = len(answer_context) < len(full_context)
        timings_ms["answer_generation"] = (time.perf_counter() - step_start) * 1000
        success = True
        return payload
    except Exception as exc:
        error_message = f"{exc.__class__.__name__}: {exc}"
        raise
    finally:
        timings_ms["total"] = (time.perf_counter() - total_start) * 1000
        timing_record = {
            "recorded_at_utc": _utc_now_iso(),
            "query": query,
            "namespace": namespace,
            "filters": filters.as_dict(),
            "llm_query_time_company_resolver": llm_query_time_company_resolver,
            "query_scope": "broad" if broad_query else "specific",
            "documents_before_filter": sum(
                episode.user_id == namespace and _artifact_type(episode) == "filing_chunk"
                for episode in all_episodes
            ),
            "documents_after_filter": len(raw_candidates),
            "filing_summaries_after_filter": len(filing_summary_candidates),
            "filing_summaries_retrieved": len(filing_summaries),
            "expanded_evidence_chunks": len(expanded_evidence),
            "documents_retrieved": len(retrieved),
            "durations_ms": {name: round(value, 3) for name, value in timings_ms.items()},
            "success": success,
            "error": error_message,
        }
        _append_query_timing(query_timings_path or store_dir / "query_timings.jsonl", timing_record)


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
    parser.add_argument(
        "--limit",
        type=int,
        help="Only ingest the first N selected corpus records (or first N corpus records without a filter).",
    )
    parser.add_argument(
        "--document-ids-file",
        type=Path,
        help=(
            "Build from only the corpus document IDs in this JSON list (or a JSON object with "
            "document_ids) / one-ID-per-line text file."
        ),
    )
    parser.add_argument("--chunk-size-tokens", type=int, default=512)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=64)
    parser.add_argument(
        "--ingestion-timings-file",
        type=Path,
        help=(
            "Append per-document ingestion timings and a final build summary as JSONL. "
            "Defaults to ingestion_timings.jsonl in the store."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print raw-ingestion progress after this many newly ingested documents; use 0 to disable.",
    )
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
        help="Retained for compatibility; filtered SEC RAG never includes profile context.",
    )
    parser.add_argument(
        "--query-timings-file",
        type=Path,
        help="Append one JSON timing record per --query. Defaults to query_timings.jsonl in the store.",
    )
    parser.add_argument(
        "--llm-query-filters",
        action="store_true",
        help="Use --ollama-model to enrich company/year extraction before metadata filtering.",
    )
    parser.add_argument(
        "--llm-query-time-company-resolver",
        action="store_true",
        help=(
            "Use --ollama-model to extract the named company, then validate that legal name against "
            "existing raw chunks before filtering. It does not write to or rebuild the store."
        ),
    )
    parser.add_argument(
        "--no-filing-summary-retrieval",
        action="store_true",
        help="Use only metadata-filtered raw chunk retrieval, even for broad questions.",
    )
    parser.add_argument(
        "--filing-summary-top-k",
        type=int,
        default=4,
        help="Filing-level summaries to retrieve for a broad query.",
    )
    parser.add_argument(
        "--evidence-per-filing-summary",
        type=int,
        default=2,
        help="Raw evidence chunks to expand from each retrieved filing summary.",
    )
    parser.add_argument("--graph-limit", type=int, default=50, help="Use 0 for the full graph.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete the target Engram store before build. Without it, a checkpointed build resumes "
            "and skips documents already persisted."
        ),
    )
    parser.add_argument("--no-summaries", action="store_true", help="Skip offline summary indexing.")
    parser.add_argument(
        "--no-filing-summaries",
        action="store_true",
        help="Do not create one retrievable summary artifact per filing during a build.",
    )
    parser.add_argument(
        "--llm-filing-summaries",
        action="store_true",
        help="Use --ollama-model to generate filing-level summaries. Otherwise a deterministic digest is stored.",
    )
    parser.add_argument(
        "--filing-summary-input-chars",
        type=int,
        default=24000,
        help="Maximum source characters sent to Ollama for each filing-level summary.",
    )
    parser.add_argument(
        "--llm-summaries",
        action="store_true",
        help="Use --ollama-model for build-time summaries. Slow on the full corpus.",
    )
    parser.add_argument(
        "--llm-document-metadata",
        action="store_true",
        help=(
            "Use --ollama-model during ingestion to enrich company names, reporting periods, and "
            "keywords. Deterministic metadata remains the fallback."
        ),
    )
    parser.add_argument(
        "--metadata-llm-text-chars",
        type=int,
        default=6000,
        help="Maximum source characters sent to Ollama per document metadata extraction.",
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
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.filing_summary_top_k <= 0:
        raise ValueError("--filing-summary-top-k must be positive")
    if args.evidence_per_filing_summary <= 0:
        raise ValueError("--evidence-per-filing-summary must be positive")
    if args.metadata_llm_text_chars <= 0:
        raise ValueError("--metadata-llm-text-chars must be positive")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.filing_summary_input_chars <= 0:
        raise ValueError("--filing-summary-input-chars must be positive")
    if args.llm_document_metadata and not args.ollama_model:
        raise ValueError("--llm-document-metadata requires --ollama-model")
    if args.llm_filing_summaries and not args.ollama_model:
        raise ValueError("--llm-filing-summaries requires --ollama-model")
    if args.llm_query_filters and not args.ollama_model:
        raise ValueError("--llm-query-filters requires --ollama-model")
    if args.llm_query_time_company_resolver and not args.ollama_model:
        raise ValueError("--llm-query-time-company-resolver requires --ollama-model")
    document_ids = (
        load_document_ids_file(args.document_ids_file)
        if args.document_ids_file is not None
        else None
    )
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
            llm=(
                llm
                if (
                    args.extract_text_facts
                    or args.llm_summaries
                    or args.llm_document_metadata
                    or args.llm_filing_summaries
                )
                else None
            ),
            embedder=embedder,
            llm_summaries=args.llm_summaries,
            llm_document_metadata=args.llm_document_metadata,
            metadata_llm_text_chars=args.metadata_llm_text_chars,
            filing_summaries=not args.no_filing_summaries,
            llm_filing_summaries=args.llm_filing_summaries,
            filing_summary_input_chars=args.filing_summary_input_chars,
            ingestion_timings_path=args.ingestion_timings_file,
            document_ids=document_ids,
            progress_every=args.progress_every,
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
            query_timings_path=args.query_timings_file,
            llm_query_filters=args.llm_query_filters,
            llm_query_time_company_resolver=args.llm_query_time_company_resolver,
            use_filing_summaries=not args.no_filing_summary_retrieval,
            filing_summary_top_k=args.filing_summary_top_k,
            evidence_per_filing_summary=args.evidence_per_filing_summary,
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
