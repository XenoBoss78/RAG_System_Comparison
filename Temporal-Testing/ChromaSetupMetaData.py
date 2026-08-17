from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from .reproducibility import DEFAULT_SEED, seed_everything
except ImportError:
    from reproducibility import DEFAULT_SEED, seed_everything


ROOT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT_DIR.parent
FIN_RATE_DIR = WORKSPACE_ROOT / "Fin-RATE"
DEFAULT_CORPUS_PATH = FIN_RATE_DIR / "corpus" / "corpus" / "corpus.jsonl"
DEFAULT_DB_DIR = FIN_RATE_DIR / "chroma_db"
DEFAULT_COLLECTION_NAME = "fin_rate"
DEFAULT_EMBEDDING_BACKEND = "default"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BGE_MODEL = "BAAI/bge-m3"
DEFAULT_LLM_BACKEND = "ollama"
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "embeddinggemma:latest"
DEFAULT_OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
COMPANY_ALIAS_FILE = "company_alias_map.json"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
BATCH_SIZE = 256
MAX_CONTEXT_CHARS = 14000

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")
TITLE_PREFIX_PATTERN = re.compile(
    r"^(?P<ticker>[A-Z][A-Z0-9.-]{0,15})_(?P<date>\d{4})-\d{2}-\d{2}_(?P<form>[^_]+)"
)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b(?:INC|INCORPORATED|CORP|CORPORATION|COMPANY|CO|LTD|LIMITED|PLC|LLC|LP|N\.V)\b",
    re.IGNORECASE,
)
COMPANY_QUERY_PATTERN = re.compile(
    r"\b[A-Z][A-Z0-9/&.-]*(?:[ _-]+[A-Z0-9/&.-]+){0,8}"
    r"(?:[ _/-]+(?:INC|INCORPORATED|CORP|CORPORATION|COMPANY|CO|LTD|LIMITED|PLC|LLC|LP|N\.V|NV)(?:/[A-Z]{2})?)\b"
)
CIK_COMPANY_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9&.,/' -]{3,}?"
    r"(?:INC|INCORPORATED|CORP|CORPORATION|COMPANY|CO|LTD|LIMITED|PLC|LLC|LP|N\.V)(?:/[A-Z]{2})?)"
    r"0{3,}\d{4,}",
    re.IGNORECASE,
)

_COMPANY_ALIAS_MAP_CACHE: dict[str, set[str]] | None = None
COMPANY_SUFFIXES = {
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "COMPANY",
    "CO",
    "LTD",
    "LIMITED",
    "PLC",
    "LLC",
    "LP",
    "NV",
}
COMPANY_QUERY_STOPWORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "ABOUT",
    "BETWEEN",
    "BY",
    "COMPARE",
    "COMPARED",
    "DID",
    "DO",
    "DURING",
    "FOR",
    "FROM",
    "GIVEN",
    "HAD",
    "HAS",
    "HAVE",
    "HOW",
    "IN",
    "INTO",
    "IS",
    "KEY",
    "OF",
    "ON",
    "OR",
    "REPORTED",
    "THE",
    "THEIR",
    "THIS",
    "TO",
    "WAS",
    "WERE",
    "WHAT",
    "WHEN",
    "WHERE",
    "WHICH",
    "WHO",
    "WITH",
}

BGE_MODEL_ALIASES = {
    "m3": DEFAULT_BGE_MODEL,
    "bge-m3": DEFAULT_BGE_MODEL,
    "small": "BAAI/bge-small-en-v1.5",
    "small-en": "BAAI/bge-small-en-v1.5",
    "base": "BAAI/bge-base-en-v1.5",
    "base-en": "BAAI/bge-base-en-v1.5",
    "large": "BAAI/bge-large-en-v1.5",
    "large-en": "BAAI/bge-large-en-v1.5",
}


def _require_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for ChromaSetup.py. Install it with: pip install chromadb"
        ) from exc
    return chromadb


def _default_embedding_function():
    """Use Chroma's local default embedding model unless a caller provides one."""
    try:
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise ImportError(
            "Chroma embedding functions are unavailable. Reinstall chromadb or pass "
            "a custom local embedding_function."
        ) from exc

    return embedding_functions.DefaultEmbeddingFunction()


def _sentence_transformer_embedding_function(*, model_name: str, device: str) -> Any:
    try:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
    except ImportError as exc:
        raise ImportError(
            "SentenceTransformerEmbeddingFunction requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    return SentenceTransformerEmbeddingFunction(
        model_name=model_name,
        device=device,
    )


class OllamaEmbeddingFunction:
    """Chroma-compatible embedding function backed by a local Ollama model."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_OLLAMA_EMBEDDING_MODEL,
        ollama_embed_url: str = DEFAULT_OLLAMA_EMBED_URL,
        truncate: bool = True,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Ollama embedding batch_size must be positive")
        self.model_name = model_name
        self.ollama_embed_url = ollama_embed_url
        self.truncate = truncate
        self.batch_size = batch_size

    def __call__(self, input: Any) -> list[list[float]]:
        if isinstance(input, str):
            documents = [input]
        else:
            documents = [str(document) for document in input]

        embeddings: list[list[float]] = []
        for start in range(0, len(documents), self.batch_size):
            embeddings.extend(self._embed_batch(documents[start:start + self.batch_size]))
        return embeddings

    def embed_documents(
        self,
        texts: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """LangChain-compatible method for embedding multiple documents."""
        if texts is None:
            texts = kwargs.get("input") or kwargs.get("documents") or kwargs.get("texts")
        if texts is None:
            raise TypeError("embed_documents requires texts, input, or documents")
        return self(texts)

    def embed_query(
        self,
        text: Any = None,
        **kwargs: Any,
    ) -> list[float] | list[list[float]]:
        """Embed one query, or preserve Chroma's nested shape when given a list."""
        if text is None:
            text = kwargs.get("input") or kwargs.get("query") or kwargs.get("text")
        if text is None:
            raise TypeError("embed_query requires text, input, or query")
        if not isinstance(text, str):
            return self(text)
        embeddings = self([text])
        return embeddings[0] if embeddings else []

    def embed_query_for_chroma(self, text: str) -> list[list[float]]:
        """Return the nested query embedding shape expected by Chroma query_embeddings."""
        return self([text])

    def _embed_batch(self, documents: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input": documents,
            "truncate": self.truncate,
        }
        request = urllib.request.Request(
            self.ollama_embed_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama embeddings. Make sure Ollama is running locally "
                f"and the embedding model is available: ollama pull {self.model_name}"
            ) from exc

        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError(
                "Ollama embedding response did not contain an 'embeddings' list."
            )
        return embeddings

    @staticmethod
    def name() -> str:
        return "ollama"

    @staticmethod
    def default_space() -> str:
        return "cosine"

    @staticmethod
    def supported_spaces() -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "ollama_embed_url": self.ollama_embed_url,
            "truncate": self.truncate,
            "batch_size": self.batch_size,
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OllamaEmbeddingFunction":
        return OllamaEmbeddingFunction(
            model_name=config.get("model_name", DEFAULT_OLLAMA_EMBEDDING_MODEL),
            ollama_embed_url=config.get("ollama_embed_url", DEFAULT_OLLAMA_EMBED_URL),
            truncate=bool(config.get("truncate", True)),
            batch_size=int(config.get("batch_size", 32)),
        )


def _resolve_bge_model_name(model_name: str) -> str:
    normalized_model_name = model_name.strip()
    if not normalized_model_name or normalized_model_name == DEFAULT_EMBEDDING_MODEL:
        return DEFAULT_BGE_MODEL
    if normalized_model_name.lower() in {"default", "bge"}:
        return DEFAULT_BGE_MODEL
    return BGE_MODEL_ALIASES.get(normalized_model_name.lower(), normalized_model_name)


def create_embedding_function(
    *,
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    ollama_embed_url: str = DEFAULT_OLLAMA_EMBED_URL,
    ollama_embedding_batch_size: int = 32,
    seed: int = DEFAULT_SEED,
) -> Any:
    """
    Create a Chroma embedding function.

    Use backend="sentence-transformers" with device="cuda" to run embeddings
    on an NVIDIA GPU when CUDA-enabled PyTorch is installed.
    Use backend="bge" to load BGE through Sentence Transformers. If model_name
    is not supplied, BAAI/bge-m3 is used. Short BGE aliases include "small",
    "base", "large", and "m3".
    Use backend="ollama" to embed through a local Ollama embedding model.
    """
    seed_everything(seed)
    normalized_backend = backend.lower().replace("_", "-")
    if normalized_backend in {"default", "onnx"}:
        return _default_embedding_function()
    if normalized_backend in {"sentence-transformer", "sentence-transformers", "st"}:
        return _sentence_transformer_embedding_function(
            model_name=model_name,
            device=device,
        )
    if normalized_backend == "bge":
        return _sentence_transformer_embedding_function(
            model_name=_resolve_bge_model_name(model_name),
            device=device,
        )
    if normalized_backend == "ollama":
        selected_model = (
            DEFAULT_OLLAMA_EMBEDDING_MODEL
            if not model_name or model_name == DEFAULT_EMBEDDING_MODEL
            else model_name
        )
        return OllamaEmbeddingFunction(
            model_name=selected_model,
            ollama_embed_url=ollama_embed_url,
            batch_size=ollama_embedding_batch_size,
        )

    raise ValueError(
        "Unsupported embedding backend. Use 'default', 'sentence-transformers', 'bge', or 'ollama'."
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _analyze(text: str) -> list[re.Match[str]]:
    return list(TOKEN_PATTERN.finditer(text))


def _token_windows(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int, int]]:
    if overlap_tokens >= max_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    matches = _analyze(text)
    if not matches:
        stripped = text.strip()
        return [(stripped, 0, 0)] if stripped else []

    chunks: list[tuple[str, int, int]] = []
    step = max(1, max_tokens - overlap_tokens)
    for token_start in range(0, len(matches), step):
        token_end = min(len(matches), token_start + max_tokens)
        char_start = matches[token_start].start()
        char_end = matches[token_end - 1].end()
        chunk = text[char_start:char_end].strip()
        if chunk:
            chunks.append((chunk, token_start, token_end))
        if token_end >= len(matches):
            break
    return chunks


def _collection_exists(client: Any, collection_name: str) -> bool:
    collections = client.list_collections()
    for collection in collections:
        if collection == collection_name:
            return True
        if getattr(collection, "name", None) == collection_name:
            return True
    return False


def _record_text(record: dict[str, Any]) -> str:
    text = record.get("text", record.get("content", record.get("contents", "")))
    if isinstance(text, str):
        return text
    return json.dumps(text, ensure_ascii=False)


def _record_title(record: dict[str, Any]) -> str:
    title = record.get("title", "")
    return title if isinstance(title, str) else str(title)


def _record_id(record: dict[str, Any], line_number: int) -> str:
    doc_id = record.get("_id", record.get("id", f"line_{line_number}"))
    return str(doc_id)


def _normalize_document_ids(document_ids: Any) -> set[str] | None:
    if document_ids is None:
        return None
    if isinstance(document_ids, str):
        raw_value = document_ids.strip()
        if not raw_value:
            return None
        if raw_value.startswith("["):
            parsed = json.loads(raw_value)
            return _normalize_document_ids(parsed)
        document_ids = re.split(r"[\s,]+", raw_value)

    normalized = {
        str(document_id).strip()
        for document_id in document_ids
        if str(document_id).strip()
    }
    return normalized or None


def _parse_document_ids_argument(
    raw_document_ids: str | None,
    repeated_document_ids: list[str] | None,
) -> list[str] | None:
    normalized: set[str] = set()
    for candidate in (
        _normalize_document_ids(raw_document_ids),
        _normalize_document_ids(repeated_document_ids),
    ):
        if candidate:
            normalized.update(candidate)
    return sorted(normalized) if normalized else None


def load_document_ids_file(path: Path) -> set[str]:
    """Load subset IDs from a JSON list/object or a one-ID-per-line text file."""
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

    normalized = _normalize_document_ids(parsed)
    if normalized is None:
        raise ValueError(
            "Document ID file must contain a JSON list, a JSON object with a "
            "'document_ids' list, or one document ID per line."
        )
    return normalized


def _parse_title_metadata(title: str) -> dict[str, Any]:
    match = TITLE_PREFIX_PATTERN.match(title)
    if not match:
        return {}

    filing_year = int(match.group("date"))
    form = match.group("form").upper()
    period_year = filing_year - 1 if form == "10-K" and filing_year <= datetime.now().year + 1 else filing_year
    return {
        "company_ticker": match.group("ticker").upper(),
        "filing_year": filing_year,
        "period_year": period_year,
        "filing_form": form,
    }


def _extract_years(text: str) -> set[int]:
    return {
        int(match.group(0))
        for match in YEAR_PATTERN.finditer(text)
        if 1990 <= int(match.group(0)) <= datetime.now().year + 1
    }


def _add_year_flags(metadata: dict[str, Any], years: set[int]) -> None:
    for year in sorted(years):
        metadata[f"mentions_{year}"] = True


def _normalize_company_alias(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", value).upper()
    normalized = LEGAL_SUFFIX_PATTERN.sub(" ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _company_aliases_from_record(title: str, text: str) -> set[str]:
    aliases: set[str] = set()

    title_parts = title.split("_")
    if title_parts:
        aliases.add(title_parts[0])
    if len(title_parts) > 4:
        aliases.add(" ".join(title_parts[4:12]))

    for match in CIK_COMPANY_PATTERN.finditer(text[:5000]):
        aliases.add(match.group(1))
    for match in COMPANY_QUERY_PATTERN.finditer(text[:5000].upper()):
        aliases.add(match.group(0))

    normalized_aliases = {
        alias
        for alias in (_normalize_company_alias(candidate) for candidate in aliases)
        if len(alias) >= 2
    }
    return normalized_aliases


def _load_company_alias_map(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path | None = DEFAULT_DB_DIR,
) -> dict[str, set[str]]:
    global _COMPANY_ALIAS_MAP_CACHE
    if _COMPANY_ALIAS_MAP_CACHE is not None:
        return _COMPANY_ALIAS_MAP_CACHE

    alias_map: dict[str, set[str]] = {}
    if db_dir is not None:
        alias_path = Path(db_dir) / COMPANY_ALIAS_FILE
        if alias_path.exists():
            raw_alias_map = json.loads(alias_path.read_text(encoding="utf-8"))
            _COMPANY_ALIAS_MAP_CACHE = {
                ticker: set(aliases)
                for ticker, aliases in raw_alias_map.items()
            }
            return _COMPANY_ALIAS_MAP_CACHE

    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        _COMPANY_ALIAS_MAP_CACHE = alias_map
        return alias_map

    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            if not line.strip():
                continue
            record = json.loads(line)
            title = _record_title(record)
            title_metadata = _parse_title_metadata(title)
            ticker = title_metadata.get("company_ticker")
            if not ticker:
                continue
            text = _record_text(record)
            aliases = _company_aliases_from_record(title, text)
            aliases.add(_normalize_company_alias(ticker))
            alias_map.setdefault(ticker, set()).update(aliases)

    _COMPANY_ALIAS_MAP_CACHE = alias_map
    return alias_map


def _extract_query_company_aliases(query: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+(?:/[A-Za-z]{2})?", query)
    aliases: set[str] = set()

    for index, token in enumerate(tokens):
        normalized_token = token.upper().replace(".", "")
        suffix = normalized_token.split("/", 1)[0]
        if suffix not in COMPANY_SUFFIXES:
            continue

        company_tokens = [token]
        cursor = index - 1
        while cursor >= 0 and len(company_tokens) < 8:
            candidate = tokens[cursor]
            candidate_upper = candidate.upper().replace(".", "")
            if (
                candidate_upper in COMPANY_QUERY_STOPWORDS
                or candidate_upper in COMPANY_SUFFIXES
                or YEAR_PATTERN.fullmatch(candidate)
            ):
                break
            company_tokens.insert(0, candidate)
            cursor -= 1

        if len(company_tokens) >= 2:
            aliases.add(_normalize_company_alias(" ".join(company_tokens)))

    return {alias for alias in aliases if alias}


def _extract_query_ticker_tokens(query: str) -> set[str]:
    return {
        token.upper()
        for token in re.findall(r"\b[A-Z]{2,6}\b", query.upper())
        if token not in {"SEC", "GAAP", "ROPA", "FY", "CEO", "CFO"}
    }


def _resolve_query_tickers(
    query: str,
    *,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path | None = DEFAULT_DB_DIR,
) -> list[str]:
    query_aliases = _extract_query_company_aliases(query)
    query_ticker_tokens = _extract_query_ticker_tokens(query)
    if not query_aliases and not query_ticker_tokens:
        return []

    alias_map = _load_company_alias_map(corpus_path=corpus_path, db_dir=db_dir)
    matched_tickers: list[str] = []
    for ticker, aliases in sorted(alias_map.items()):
        normalized_ticker = _normalize_company_alias(ticker)
        if ticker in query_ticker_tokens or normalized_ticker in query_aliases:
            matched_tickers.append(ticker)
            continue
        if any(
            query_alias == alias
            or (
                len(query_alias.split()) >= 2
                and len(alias.split()) >= 2
                and (query_alias in alias or alias in query_alias)
            )
            for query_alias in query_aliases
            for alias in aliases
        ):
            matched_tickers.append(ticker)
    return matched_tickers


def _extract_query_years(query: str) -> list[int]:
    years = _extract_years(query)
    if years and re.search(r"\b(previous|prior|preceding|earlier)\s+year\b", query, re.IGNORECASE):
        years.update(year - 1 for year in list(years))
    return sorted(years)


def _or_filter(conditions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$or": conditions}


def _and_filter(conditions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _year_where_condition(years: list[int]) -> dict[str, Any] | None:
    year_conditions: list[dict[str, Any]] = []
    for year in years:
        year_conditions.extend(
            [
                {f"mentions_{year}": {"$eq": True}},
                {"filing_year": {"$eq": year}},
                {"period_year": {"$eq": year}},
            ]
        )
    return _or_filter(year_conditions)


def build_query_metadata_filter(
    query: str,
    *,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path | None = DEFAULT_DB_DIR,
) -> dict[str, Any] | None:
    """Extract companies/years from the query and convert them into a Chroma where filter."""
    tickers = _resolve_query_tickers(query, corpus_path=corpus_path, db_dir=db_dir)
    years = _extract_query_years(query)

    conditions: list[dict[str, Any]] = []
    if tickers:
        if len(tickers) == 1:
            conditions.append({"company_ticker": {"$eq": tickers[0]}})
        else:
            conditions.append({"company_ticker": {"$in": tickers}})

    year_condition = _year_where_condition(years)
    if year_condition is not None:
        conditions.append(year_condition)

    return _and_filter(conditions)


def _extract_json_object(value: str) -> dict[str, Any]:
    cleaned = value.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Expected the query breakdown response to be a JSON object.")
    return parsed


def _normalize_ollama_query_breakdown(payload: dict[str, Any]) -> dict[str, Any]:
    companies: list[dict[str, str | None]] = []
    for item in payload.get("companies") or []:
        if isinstance(item, str):
            companies.append({"name": item, "ticker": None})
        elif isinstance(item, dict):
            name = item.get("name") or item.get("company") or item.get("company_name")
            ticker = item.get("ticker") or item.get("symbol")
            if name or ticker:
                companies.append(
                    {
                        "name": str(name).strip() if name else None,
                        "ticker": str(ticker).strip().upper() if ticker else None,
                    }
                )

    tickers = {
        str(ticker).strip().upper()
        for ticker in payload.get("tickers") or []
        if str(ticker).strip()
    }
    tickers.update(
        str(company["ticker"]).strip().upper()
        for company in companies
        if company.get("ticker")
    )

    years: set[int] = set()
    for value in payload.get("years") or []:
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        if 1990 <= year <= datetime.now().year + 1:
            years.add(year)

    include_previous_year = bool(payload.get("include_previous_year", False))
    if years and include_previous_year:
        years.update(year - 1 for year in list(years))

    return {
        "companies": companies,
        "tickers": sorted(tickers),
        "years": sorted(years),
        "include_previous_year": include_previous_year,
        "periods": payload.get("periods") or [],
        "question_type": payload.get("question_type"),
    }


def break_down_query_with_ollama(
    query: str,
    *,
    llm_model: str = DEFAULT_LLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """
    Use Ollama to extract structured retrieval hints from a query.

    The returned dictionary is normalized to contain companies, tickers, years,
    include_previous_year, periods, and question_type.
    """
    prompt = (
        "Extract structured retrieval hints from this financial QA query.\n"
        "Return ONLY valid JSON. Do not explain your answer.\n"
        "Do not infer companies or years that are not stated in the query.\n"
        "If a company ticker is explicit or obvious from the query text, include it; "
        "otherwise use null and let downstream code resolve the company name.\n"
        "If the query says previous year, prior year, preceding year, or earlier year, "
        "set include_previous_year to true.\n\n"
        "JSON schema:\n"
        "{\n"
        '  "companies": [{"name": "company name from query", "ticker": "ticker or null"}],\n'
        '  "tickers": ["explicit ticker symbols from query"],\n'
        '  "years": [explicit four digit years],\n'
        '  "include_previous_year": false,\n'
        '  "periods": ["quarters or fiscal periods mentioned"],\n'
        '  "question_type": "short label"\n'
        "}\n\n"
        f"Query:\n{query.strip()}"
    )
    response = _generate_with_ollama(
        prompt,
        model=llm_model,
        ollama_url=ollama_url,
        temperature=temperature,
        seed=seed,
        json_mode=True,
    )
    return _normalize_ollama_query_breakdown(_extract_json_object(response))


def _resolve_breakdown_tickers(
    breakdown: dict[str, Any],
    *,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path | None = DEFAULT_DB_DIR,
) -> list[str]:
    alias_map = _load_company_alias_map(corpus_path=corpus_path, db_dir=db_dir)
    requested_tickers = {
        str(ticker).strip().upper()
        for ticker in breakdown.get("tickers", [])
        if str(ticker).strip()
    }
    company_aliases = {
        _normalize_company_alias(str(company.get("name")))
        for company in breakdown.get("companies", [])
        if isinstance(company, dict) and company.get("name")
    }

    if not alias_map:
        return sorted(requested_tickers)

    matched_tickers: set[str] = set()
    for ticker, aliases in alias_map.items():
        normalized_ticker = _normalize_company_alias(ticker)
        if ticker in requested_tickers or normalized_ticker in company_aliases:
            matched_tickers.add(ticker)
            continue
        if any(
            company_alias == alias
            or (
                len(company_alias.split()) >= 2
                and len(alias.split()) >= 2
                and (company_alias in alias or alias in company_alias)
            )
            for company_alias in company_aliases
            for alias in aliases
        ):
            matched_tickers.add(ticker)

    return sorted(matched_tickers)


def build_query_metadata_filter_with_ollama(
    query: str,
    *,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path | None = DEFAULT_DB_DIR,
    llm_model: str = DEFAULT_LLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any] | None:
    """
    Use Ollama to break down a query, then convert the result into a Chroma where filter.
    """
    breakdown = break_down_query_with_ollama(
        query,
        llm_model=llm_model,
        ollama_url=ollama_url,
        temperature=temperature,
        seed=seed,
    )
    tickers = _resolve_breakdown_tickers(
        breakdown,
        corpus_path=corpus_path,
        db_dir=db_dir,
    )
    years = [int(year) for year in breakdown.get("years", [])]

    conditions: list[dict[str, Any]] = []
    if tickers:
        if len(tickers) == 1:
            conditions.append({"company_ticker": {"$eq": tickers[0]}})
        else:
            conditions.append({"company_ticker": {"$in": tickers}})

    year_condition = _year_where_condition(years)
    if year_condition is not None:
        conditions.append(year_condition)

    return _and_filter(conditions)


def _build_auto_metadata_filter(
    query: str,
    *,
    mode: str,
    corpus_path: str | Path,
    db_dir: str | Path,
    llm_model: str,
    ollama_url: str,
    temperature: float,
    seed: int,
) -> dict[str, Any] | None:
    normalized_mode = mode.lower().replace("_", "-")
    if normalized_mode in {"heuristic", "regex", "rules"}:
        return build_query_metadata_filter(query, corpus_path=corpus_path, db_dir=db_dir)
    if normalized_mode == "ollama":
        return build_query_metadata_filter_with_ollama(
            query,
            corpus_path=corpus_path,
            db_dir=db_dir,
            llm_model=llm_model,
            ollama_url=ollama_url,
            temperature=temperature,
            seed=seed,
        )
    if normalized_mode in {"none", "off", "disabled"}:
        return None
    raise ValueError("metadata_filter_mode must be 'heuristic', 'ollama', or 'none'.")


def _merge_where_filters(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if left is None:
        return right
    if right is None:
        return left
    return {"$and": [left, right]}


def _flush_batch(collection: Any, batch: dict[str, list[Any]]) -> None:
    if not batch["ids"]:
        return
    collection.add(
        ids=batch["ids"],
        documents=batch["documents"],
        metadatas=batch["metadatas"],
    )
    batch["ids"].clear()
    batch["documents"].clear()
    batch["metadatas"].clear()


def get_chroma_collection(
    db_dir: str | Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    *,
    embedding_function: Any | None = None,
) -> Any:
    """Load an existing persistent Chroma collection for local querying."""
    chromadb = _require_chromadb()
    client = chromadb.PersistentClient(path=str(Path(db_dir)))
    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=embedding_function or _default_embedding_function(),
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Chroma collection '{collection_name}' was not found in {Path(db_dir)}. "
            "Run build_chroma_database(...) first."
        ) from exc


def build_chroma_database(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path = DEFAULT_DB_DIR,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    batch_size: int = BATCH_SIZE,
    embedding_function: Any | None = None,
    document_ids: list[str] | set[str] | tuple[str, ...] | str | None = None,
    reset_collection: bool = True,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """
    Build a fully local persistent Chroma database from the Fin-RATE corpus.jsonl.

    The corpus is expected to be JSONL with fields like _id, title, and text.
    Passing a custom local embedding_function lets you use Sentence Transformers,
    Ollama embeddings, or another local embedding backend.
    Passing document_ids builds a smaller database from only those corpus records.
    """
    seed_everything(seed)
    chromadb = _require_chromadb()
    corpus_path = Path(corpus_path)
    db_dir = Path(db_dir)
    target_document_ids = _normalize_document_ids(document_ids)
    matched_document_ids: set[str] = set()

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    if reset_collection and _collection_exists(client, collection_name):
        client.delete_collection(name=collection_name)

    embedder = embedding_function or _default_embedding_function()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    started_at = _utc_now_iso()
    start_time = time.perf_counter()
    records_read = 0
    documents_seen = 0
    chunks_indexed = 0
    tokens_processed = 0
    company_alias_map: dict[str, set[str]] = {}
    batch: dict[str, list[Any]] = {"ids": [], "documents": [], "metadatas": []}

    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue

            records_read += 1
            record = json.loads(line)

            doc_id = _record_id(record, line_number)
            if target_document_ids is not None and doc_id not in target_document_ids:
                continue

            documents_seen += 1
            matched_document_ids.add(doc_id)
            title = _record_title(record)
            text = _record_text(record)
            title_metadata = _parse_title_metadata(title)
            company_ticker = title_metadata.get("company_ticker")
            if company_ticker:
                aliases = _company_aliases_from_record(title, text)
                aliases.add(_normalize_company_alias(str(company_ticker)))
                company_alias_map.setdefault(str(company_ticker), set()).update(aliases)
            tokens_processed += len(_analyze(text))

            for chunk_index, (chunk_text, token_start, token_end) in enumerate(
                _token_windows(
                    text,
                    max_tokens=chunk_size_tokens,
                    overlap_tokens=chunk_overlap_tokens,
                )
            ):
                readable_title = title.replace("_", " ").strip()
                document_text = (
                    f"{readable_title}\n\n{chunk_text}" if readable_title else chunk_text
                )
                chunk_id = f"{doc_id}::chunk_{chunk_index:04d}"
                metadata = {
                    "doc_id": doc_id,
                    "title": title,
                    "source_line": line_number,
                    "chunk_index": chunk_index,
                    "token_start": token_start,
                    "token_end": token_end,
                }
                metadata.update(title_metadata)
                chunk_years = _extract_years(f"{title}\n{chunk_text}")
                if "filing_year" in title_metadata:
                    chunk_years.add(int(title_metadata["filing_year"]))
                if "period_year" in title_metadata:
                    chunk_years.add(int(title_metadata["period_year"]))
                _add_year_flags(metadata, chunk_years)

                batch["ids"].append(chunk_id)
                batch["documents"].append(document_text)
                batch["metadatas"].append(metadata)
                chunks_indexed += 1

                if len(batch["ids"]) >= batch_size:
                    _flush_batch(collection, batch)

            if (
                target_document_ids is not None
                and matched_document_ids == target_document_ids
            ):
                break

    _flush_batch(collection, batch)
    alias_payload = {
        ticker: sorted(aliases)
        for ticker, aliases in sorted(company_alias_map.items())
    }
    (db_dir / COMPANY_ALIAS_FILE).write_text(
        json.dumps(alias_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    global _COMPANY_ALIAS_MAP_CACHE
    _COMPANY_ALIAS_MAP_CACHE = company_alias_map

    stats = {
        "corpus_path": str(corpus_path),
        "database_dir": str(db_dir),
        "collection_name": collection_name,
        "records_read": records_read,
        "documents_seen": documents_seen,
        "document_ids_filter": sorted(target_document_ids) if target_document_ids else None,
        "document_ids_matched": sorted(matched_document_ids),
        "document_ids_missing": (
            sorted(target_document_ids - matched_document_ids)
            if target_document_ids
            else []
        ),
        "chunks_indexed": chunks_indexed,
        "collection_count": collection.count(),
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "embedding_function": type(embedder).__name__,
        "seed": seed,
        "build_started_at_utc": started_at,
        "build_finished_at_utc": _utc_now_iso(),
        "build_seconds": time.perf_counter() - start_time,
    }
    (db_dir / "build_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def retrieve_relevant_chunks(
    query: str,
    db_dir: str | Path = DEFAULT_DB_DIR,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    n_results: int = 15,
    embedding_function: Any | None = None,
    where: dict[str, Any] | None = None,
    where_document: dict[str, Any] | None = None,
    auto_metadata_filter: bool = True,
    metadata_filter_mode: str = "heuristic",
    metadata_llm_model: str = DEFAULT_LLM_MODEL,
    metadata_temperature: float = 0.0,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    strict_metadata_filter: bool = False,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant Chroma chunks for a query.

    Returns dictionaries with id, text, metadata, distance, and score. Chroma
    cosine distances are lower-is-better; score is 1 - distance for convenience.
    """
    seed_everything(seed)
    if not query.strip():
        return []
    if n_results <= 0:
        raise ValueError("n_results must be positive")

    collection = get_chroma_collection(
        db_dir=db_dir,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    collection_count = collection.count()
    if collection_count == 0:
        return []

    auto_where = (
        _build_auto_metadata_filter(
            query,
            mode=metadata_filter_mode,
            corpus_path=corpus_path,
            db_dir=db_dir,
            llm_model=metadata_llm_model,
            ollama_url=ollama_url,
            temperature=metadata_temperature,
            seed=seed,
        )
        if auto_metadata_filter
        else None
    )
    effective_where = _merge_where_filters(where, auto_where)

    def _query(active_where: dict[str, Any] | None) -> dict[str, Any]:
        query_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, collection_count),
            "include": ["documents", "metadatas", "distances"],
        }
        if active_where is not None:
            query_kwargs["where"] = active_where
        if where_document is not None:
            query_kwargs["where_document"] = where_document
        return collection.query(**query_kwargs)

    results = _query(effective_where)
    ids = results.get("ids", [[]])[0]
    if auto_where is not None and not ids and not strict_metadata_filter:
        results = _query(where)
        ids = results.get("ids", [[]])[0]

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        distance = distances[index] if index < len(distances) else None
        chunks.append(
            {
                "id": chunk_id,
                "text": documents[index] if index < len(documents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
                "distance": distance,
                "score": None if distance is None else 1.0 - float(distance),
            }
        )
    return chunks


def build_answer_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Build a grounded QA prompt from retrieved Chroma chunks."""
    context_blocks: list[str] = []
    chars_used = 0

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        doc_id = metadata.get("doc_id", chunk.get("id", f"chunk_{index}"))
        title = metadata.get("title", "")
        score = chunk.get("score")
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue

        source_line = f"[{index}] doc_id={doc_id}"
        if title:
            source_line += f" title={title}"
        if score is not None:
            source_line += f" score={float(score):.4f}"

        remaining_chars = max_context_chars - chars_used
        if remaining_chars <= 0:
            break

        block = f"{source_line}\n{text}"
        if len(block) > remaining_chars:
            block = block[:remaining_chars].rsplit(" ", 1)[0].rstrip()
        if block:
            context_blocks.append(block)
            chars_used += len(block)

    context = "\n\n---\n\n".join(context_blocks) or "No retrieved context was provided."
    return (
        "You are answering questions about SEC filing excerpts.\n"
        "Use only the retrieved context below. If the context is insufficient, "
        "say that the answer cannot be determined from the retrieved chunks.\n"
        "Cite supporting chunks inline using bracketed source numbers like [1] or [2].\n"
        "Keep the answer concise, but include the key numbers, entities, and dates when present.\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Answer:"
    )


def _generate_with_ollama(
    prompt: str,
    *,
    model: str = DEFAULT_LLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.1,
    seed: int = DEFAULT_SEED,
    json_mode: bool = False,
) -> str:
    # Most callers use the full generate endpoint, but callers that also use
    # Ollama's embedding API commonly keep only its base URL.  Accept either
    # form so a metadata-filter request cannot accidentally be sent to `/`.
    normalized_url = ollama_url.rstrip("/")
    if not normalized_url.endswith("/api/generate"):
        normalized_url = (
            f"{normalized_url}/generate"
            if normalized_url.endswith("/api")
            else f"{normalized_url}/api/generate"
        )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed},
    }
    if json_mode:
        payload["format"] = "json"
        # Qwen3 otherwise puts a valid JSON answer in its `thinking` field
        # and leaves `response` empty, which breaks structured filtering.
        payload["think"] = False
    request = urllib.request.Request(
        normalized_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Ollama request to {normalized_url} failed with HTTP {exc.code}"
            f"{f': {detail}' if detail else ''}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {normalized_url}. Make sure Ollama is "
            f"running locally and the model is available: ollama pull {model}"
        ) from exc

    # The fallback supports Qwen3 instances where `think: false` is ignored
    # by an older Ollama server.
    return str(result.get("response") or result.get("thinking") or "").strip()


def generate_answer_from_chunks(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    llm: Callable[[str], str] | None = None,
    llm_backend: str = DEFAULT_LLM_BACKEND,
    llm_model: str = DEFAULT_LLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.1,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """
    Generate an answer from retrieved chunks.

    Pass llm=<callable> to use any model client. If no callable is passed, the
    default backend calls a local Ollama model.
    """
    seed_everything(seed)
    prompt = build_answer_prompt(
        question,
        chunks,
        max_context_chars=max_context_chars,
    )

    with (ROOT_DIR / "example.txt").open("w", encoding="utf-8") as file:
        file.write(prompt)

    if llm is not None:
        answer = llm(prompt)
    elif llm_backend == "ollama":
        answer = _generate_with_ollama(
            prompt,
            model=llm_model,
            ollama_url=ollama_url,
            temperature=temperature,
            seed=seed,
        )
    else:
        raise ValueError("Unsupported llm_backend. Use 'ollama' or pass llm=<callable>.")

    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "prompt": prompt,
    }


def answer_question(
    question: str,
    db_dir: str | Path = DEFAULT_DB_DIR,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    n_results: int = 15,
    embedding_function: Any | None = None,
    llm: Callable[[str], str] | None = None,
    llm_backend: str = DEFAULT_LLM_BACKEND,
    llm_model: str = DEFAULT_LLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.1,
    max_context_chars: int = MAX_CONTEXT_CHARS,
    auto_metadata_filter: bool = True,
    metadata_filter_mode: str = "heuristic",
    metadata_llm_model: str = DEFAULT_LLM_MODEL,
    metadata_temperature: float = 0.0,
    strict_metadata_filter: bool = False,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Retrieve relevant chunks and generate a grounded answer."""
    seed_everything(seed)
    chunks = retrieve_relevant_chunks(
        question,
        db_dir=db_dir,
        collection_name=collection_name,
        n_results=n_results,
        embedding_function=embedding_function,
        auto_metadata_filter=auto_metadata_filter,
        metadata_filter_mode=metadata_filter_mode,
        metadata_llm_model=metadata_llm_model,
        metadata_temperature=metadata_temperature,
        ollama_url=ollama_url,
        strict_metadata_filter=strict_metadata_filter,
        corpus_path=corpus_path,
        seed=seed,
    )
    return generate_answer_from_chunks(
        question,
        chunks,
        llm=llm,
        llm_backend=llm_backend,
        llm_model=llm_model,
        ollama_url=ollama_url,
        temperature=temperature,
        max_context_chars=max_context_chars,
        seed=seed,
    )


def build_database_from_corpus(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias with a plainer name for callers that do not care about Chroma details."""
    return build_chroma_database(*args, **kwargs)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build or query a local Chroma DB.")
    parser.add_argument("--build", action="store_true", help="Build the Chroma database.")
    parser.add_argument("--query", type=str, help="Query an already-built Chroma database.")
    parser.add_argument("--answer", type=str, help="Retrieve chunks and answer a question.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--doc-id",
        dest="document_ids",
        action="append",
        default=[],
        help="Only build from this corpus document ID. Can be passed more than once.",
    )
    parser.add_argument(
        "--doc-ids",
        default=None,
        help="Only build from these document IDs. Accepts comma-separated IDs or a JSON list.",
    )
    parser.add_argument(
        "--document-ids-file",
        type=Path,
        help=(
            "Build from only the corpus document IDs in this JSON list (or a JSON object with "
            "document_ids) / one-ID-per-line text file."
        ),
    )
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=5)
    parser.add_argument(
        "--embedding-backend",
        choices=["default", "sentence-transformers", "bge", "ollama"],
        default=DEFAULT_EMBEDDING_BACKEND,
        help="Embedding backend to use for build/query.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Sentence Transformers model name. With --embedding-backend bge, "
            "defaults to BAAI/bge-m3 and also accepts aliases: small, base, "
            "large, m3. With --embedding-backend ollama, defaults to embeddinggemma:latest."
        ),
    )
    parser.add_argument(
        "--ollama-embed-url",
        default=DEFAULT_OLLAMA_EMBED_URL,
        help="Ollama embedding endpoint.",
    )
    parser.add_argument(
        "--ollama-embedding-batch-size",
        type=int,
        default=32,
        help="Number of texts to send to Ollama per embedding request.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device, e.g. cpu or cuda, for sentence-transformers.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["ollama"],
        default=DEFAULT_LLM_BACKEND,
        help="Local answer-generation backend.",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help="Local LLM model name for answer generation.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama generate endpoint.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument(
        "--no-auto-metadata-filter",
        action="store_true",
        help="Disable automatic company/year metadata filtering before vector search.",
    )
    parser.add_argument(
        "--metadata-filter-mode",
        choices=["heuristic", "ollama", "none"],
        default="heuristic",
        help="How to extract company/year metadata filters before vector search.",
    )
    parser.add_argument(
        "--metadata-llm-model",
        default=None,
        help="Ollama model for LLM-based query breakdown. Defaults to --llm-model.",
    )
    parser.add_argument(
        "--metadata-temperature",
        type=float,
        default=0.0,
        help="Temperature for LLM-based query breakdown.",
    )
    parser.add_argument(
        "--strict-metadata-filter",
        action="store_true",
        help="Do not fall back to unfiltered retrieval when the automatic metadata filter returns no chunks.",
    )
    parser.add_argument("--no-reset", action="store_true", help="Do not delete an existing collection.")
    args = parser.parse_args()
    seed_everything(args.seed)

    embedding_function = create_embedding_function(
        backend=args.embedding_backend,
        model_name=args.embedding_model,
        device=args.device,
        ollama_embed_url=args.ollama_embed_url,
        ollama_embedding_batch_size=args.ollama_embedding_batch_size,
        seed=args.seed,
    )

    if args.build:
        document_ids = _parse_document_ids_argument(args.doc_ids, args.document_ids)
        if args.document_ids_file is not None:
            document_ids = sorted(
                set(document_ids or []) | load_document_ids_file(args.document_ids_file)
            )
        stats = build_chroma_database(
            corpus_path=args.corpus,
            db_dir=args.db_dir,
            collection_name=args.collection,
            embedding_function=embedding_function,
            document_ids=document_ids,
            reset_collection=not args.no_reset,
            seed=args.seed,
        )
        print(json.dumps(stats, indent=2))

    if args.query:
        chunks = retrieve_relevant_chunks(
            args.query,
            db_dir=args.db_dir,
            collection_name=args.collection,
            n_results=args.top_k,
            embedding_function=embedding_function,
            auto_metadata_filter=not args.no_auto_metadata_filter,
            metadata_filter_mode=args.metadata_filter_mode,
            metadata_llm_model=args.metadata_llm_model or args.llm_model,
            metadata_temperature=args.metadata_temperature,
            ollama_url=args.ollama_url,
            strict_metadata_filter=args.strict_metadata_filter,
            corpus_path=args.corpus,
            seed=args.seed,
        )
        for chunk in chunks:
            print(json.dumps(chunk, ensure_ascii=False, indent=2))
            print("\n---\n")

    if args.answer:
        result = answer_question(
            args.answer,
            db_dir=args.db_dir,
            collection_name=args.collection,
            n_results=args.top_k,
            embedding_function=embedding_function,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            ollama_url=args.ollama_url,
            temperature=args.temperature,
            max_context_chars=args.max_context_chars,
            auto_metadata_filter=not args.no_auto_metadata_filter,
            metadata_filter_mode=args.metadata_filter_mode,
            metadata_llm_model=args.metadata_llm_model or args.llm_model,
            metadata_temperature=args.metadata_temperature,
            strict_metadata_filter=args.strict_metadata_filter,
            corpus_path=args.corpus,
            seed=args.seed,
        )
        sources = [
            {
                "rank": index,
                "id": chunk.get("id"),
                "doc_id": (chunk.get("metadata") or {}).get("doc_id"),
                "title": (chunk.get("metadata") or {}).get("title"),
                "company_ticker": (chunk.get("metadata") or {}).get("company_ticker"),
                "filing_year": (chunk.get("metadata") or {}).get("filing_year"),
                "period_year": (chunk.get("metadata") or {}).get("period_year"),
                "score": chunk.get("score"),
            }
            for index, chunk in enumerate(result["chunks"], start=1)
        ]
        datajson = {
            "question": result["question"],
            "answer": result["answer"],
            "sources": sources,}
        with (ROOT_DIR / "data.json").open("w", encoding="utf-8") as f:
            json.dump(datajson, f, ensure_ascii=False, indent=2)
        print(json.dumps(datajson, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
