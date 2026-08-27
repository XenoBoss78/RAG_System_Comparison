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

# These words describe the shape of a question rather than the information that
# must be present in retrieved evidence.  They are excluded from the lightweight
# lexical verification check below.  Domain terms such as "leadership",
# "returns", "headcount", and "compensation" are deliberately retained.
VERIFICATION_STOPWORDS = COMPANY_QUERY_STOPWORDS | {
    "ACROSS",
    "AFTER",
    "AGAINST",
    "ALL",
    "ALSO",
    "AMONG",
    "ANY",
    "AROUND",
    "BE",
    "BEEN",
    "BEFORE",
    "BEING",
    "BETWEEN",
    "BOTH",
    "CAN",
    "COULD",
    "CURRENT",
    "DATA",
    "DESCRIBE",
    "DETAIL",
    "DIFFER",
    "DIFFERENT",
    "DID",
    "DO",
    "DOES",
    "EACH",
    "END",
    "EXPLAIN",
    "FIRST",
    "GIVE",
    "GIVEN",
    "HAD",
    "HAS",
    "HAVE",
    "IMPACT",
    "INFORMATION",
    "ITS",
    "KEY",
    "LAST",
    "MAIN",
    "MAKE",
    "MOST",
    "MORE",
    "MUCH",
    "NEED",
    "NEW",
    "NEXT",
    "NO",
    "NOT",
    "NOW",
    "OF",
    "ONE",
    "OTHER",
    "OUR",
    "OUT",
    "OVER",
    "PAST",
    "PER",
    "PLEASE",
    "QUESTION",
    "RECENT",
    "RELEVANT",
    "REPORTED",
    "RESULT",
    "RESULTS",
    "SAME",
    "SECOND",
    "SHOW",
    "SINCE",
    "SO",
    "SUCH",
    "SUMMARIZE",
    "TELL",
    "THAN",
    "THAT",
    "THE",
    "THESE",
    "THEY",
    "THIRD",
    "THOSE",
    "TIME",
    "TIMES",
    "TODAY",
    "TOTAL",
    "TOWARD",
    "THROUGH",
    "UP",
    "USE",
    "USING",
    "VERSUS",
    "VERY",
    "WANT",
    "WAY",
    "WE",
    "WERE",
    "WHAT",
    "WHEN",
    "WHICH",
    "WHO",
    "WHY",
    "WILL",
    "WOULD",
    "YEAR",
    "YEARS",
    "YOU",
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


def _normalize_verification_term(value: str) -> str:
    """Normalize a word for conservative lexical evidence matching."""
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    if normalized.endswith("ies") and len(normalized) > 4:
        return normalized[:-3] + "y"
    if (
        normalized.endswith("s")
        and len(normalized) > 4
        and not normalized.endswith(("ss", "us"))
    ):
        return normalized[:-1]
    return normalized


def _verification_terms(text: str) -> set[str]:
    return {
        normalized
        for token in TOKEN_PATTERN.findall(text)
        if (normalized := _normalize_verification_term(token))
    }


def _extract_query_keypoints(
    query: str,
    *,
    tickers: list[str],
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path | None = DEFAULT_DB_DIR,
    max_keypoints: int = 8,
) -> list[str]:
    """
    Extract a small, explainable set of content terms to verify after retrieval.

    This is intentionally not a second semantic-query parser.  It only identifies
    meaningful words from the original question so that the verifier can report
    what its retrieved evidence does and does not literally cover.
    """
    if max_keypoints <= 0:
        return []

    company_terms = {_normalize_verification_term(ticker) for ticker in tickers}
    alias_map = _load_company_alias_map(corpus_path=corpus_path, db_dir=db_dir)
    for ticker in tickers:
        for alias in alias_map.get(ticker, set()):
            company_terms.update(_verification_terms(alias))

    keypoints: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_PATTERN.findall(query):
        normalized = _normalize_verification_term(token)
        if (
            not normalized
            or normalized.isdigit()
            or len(normalized) < 3
            or token.upper() in VERIFICATION_STOPWORDS
            or normalized.upper() in COMPANY_SUFFIXES
            or normalized in company_terms
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        keypoints.append(normalized)
        if len(keypoints) >= max_keypoints:
            break
    return keypoints


def extract_query_verification_requirements(
    query: str,
    *,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path | None = DEFAULT_DB_DIR,
    max_keypoints: int = 8,
) -> dict[str, list[Any]]:
    """Return the company, year, and content requirements checked after retrieval."""
    tickers = _resolve_query_tickers(
        query,
        corpus_path=corpus_path,
        db_dir=db_dir,
    )
    return {
        "company_tickers": tickers,
        "years": _extract_query_years(query),
        "keypoints": _extract_query_keypoints(
            query,
            tickers=tickers,
            corpus_path=corpus_path,
            db_dir=db_dir,
            max_keypoints=max_keypoints,
        ),
    }


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


def _company_where_condition(tickers: list[str]) -> dict[str, Any] | None:
    if not tickers:
        return None
    if len(tickers) == 1:
        return {"company_ticker": {"$eq": tickers[0]}}
    return {"company_ticker": {"$in": tickers}}


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
    company_condition = _company_where_condition(tickers)
    if company_condition is not None:
        conditions.append(company_condition)

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
    company_condition = _company_where_condition(tickers)
    if company_condition is not None:
        conditions.append(company_condition)

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


def _chunk_matches_year(chunk: dict[str, Any], year: int) -> bool:
    metadata = chunk.get("metadata") or {}
    if metadata.get(f"mentions_{year}") is True:
        return True
    for field in ("filing_year", "period_year"):
        try:
            if int(metadata.get(field)) == year:
                return True
        except (TypeError, ValueError):
            continue
    return bool(YEAR_PATTERN.search(str(chunk.get("text", ""))) and re.search(
        rf"\b{re.escape(str(year))}\b", str(chunk.get("text", ""))
    ))


def _chunk_evidence_units(
    chunk: dict[str, Any],
    requirements: dict[str, list[Any]],
) -> set[str]:
    """Return the explicit verification requirements supported by one chunk."""
    metadata = chunk.get("metadata") or {}
    units: set[str] = set()
    chunk_ticker = str(metadata.get("company_ticker") or "").strip().upper()
    for ticker in requirements.get("company_tickers", []):
        if chunk_ticker == str(ticker).upper():
            units.add(f"company:{ticker}")

    for year in requirements.get("years", []):
        if _chunk_matches_year(chunk, int(year)):
            units.add(f"year:{int(year)}")

    text_terms = _verification_terms(
        f"{metadata.get('title', '')}\n{chunk.get('text', '')}"
    )
    for keypoint in requirements.get("keypoints", []):
        if str(keypoint) in text_terms:
            units.add(f"keypoint:{keypoint}")
    return units


def _requirement_units(
    requirements: dict[str, list[Any]],
    *,
    include_keypoints: bool = True,
) -> set[str]:
    units = {
        *(f"company:{ticker}" for ticker in requirements.get("company_tickers", [])),
        *(f"year:{int(year)}" for year in requirements.get("years", [])),
    }
    if include_keypoints:
        units.update(
            f"keypoint:{keypoint}" for keypoint in requirements.get("keypoints", [])
        )
    return units


def verify_retrieved_evidence(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path | None = DEFAULT_DB_DIR,
    max_keypoints: int = 8,
    requirements: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Audit whether retrieved chunks explicitly cover the query's company, years,
    and meaningful content terms.  It does not claim that the answer is true;
    it reports whether the supplied retrieval context contains the requested
    evidence signals.
    """
    started_at = time.perf_counter()
    requirements = requirements or extract_query_verification_requirements(
        query,
        corpus_path=corpus_path,
        db_dir=db_dir,
        max_keypoints=max_keypoints,
    )
    supported_units: set[str] = set()
    evidence_by_unit: dict[str, list[str]] = {}
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        for unit in _chunk_evidence_units(chunk, requirements):
            supported_units.add(unit)
            evidence_by_unit.setdefault(unit, []).append(chunk_id)

    required_units = _requirement_units(requirements)
    metadata_required_units = _requirement_units(requirements, include_keypoints=False)
    missing_units = required_units - supported_units
    missing_metadata_units = metadata_required_units - supported_units
    requested_tickers = [str(ticker) for ticker in requirements.get("company_tickers", [])]
    requested_years = [int(year) for year in requirements.get("years", [])]
    requested_keypoints = [str(keypoint) for keypoint in requirements.get("keypoints", [])]

    def _found(prefix: str, values: list[Any]) -> list[Any]:
        return [value for value in values if f"{prefix}:{value}" in supported_units]

    def _missing(prefix: str, values: list[Any]) -> list[Any]:
        return [value for value in values if f"{prefix}:{value}" in missing_units]

    return {
        "enabled": True,
        "requirements": {
            "company_tickers": requested_tickers,
            "years": requested_years,
            "keypoints": requested_keypoints,
        },
        "coverage": {
            "company_tickers_found": _found("company", requested_tickers),
            "company_tickers_missing": _missing("company", requested_tickers),
            "years_found": _found("year", requested_years),
            "years_missing": _missing("year", requested_years),
            "keypoints_found": _found("keypoint", requested_keypoints),
            "keypoints_missing": _missing("keypoint", requested_keypoints),
        },
        "is_sufficient": not missing_units,
        "metadata_sufficient": not missing_metadata_units,
        "lexical_keypoints_sufficient": not any(
            unit.startswith("keypoint:") for unit in missing_units
        ),
        "chunks_checked": len(chunks),
        "evidence_chunk_ids": evidence_by_unit,
        "verification_seconds": time.perf_counter() - started_at,
    }


def _deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id or chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        unique_chunks.append(chunk)
    return unique_chunks


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _select_chunks_for_evidence(
    chunks: list[dict[str, Any]],
    *,
    n_results: int,
    requirements: dict[str, list[Any]],
    include_keypoints: bool = True,
) -> list[dict[str, Any]]:
    """Keep high-ranked chunks while reserving room for uncovered evidence."""
    candidates = _deduplicate_chunks(chunks)
    if n_results >= len(candidates):
        return candidates

    required_units = _requirement_units(
        requirements,
        include_keypoints=include_keypoints,
    )
    uncovered_units = set(required_units)
    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    while uncovered_units and remaining and len(selected) < n_results:
        best_index, best_new_units, best_score = -1, set(), float("-inf")
        for index, chunk in enumerate(remaining):
            chunk_units = _chunk_evidence_units(chunk, requirements)
            if not include_keypoints:
                chunk_units = {
                    unit for unit in chunk_units if not unit.startswith("keypoint:")
                }
            new_units = chunk_units & uncovered_units
            score = chunk.get("score")
            numeric_score = float(score) if isinstance(score, (int, float)) else float("-inf")
            if (len(new_units), numeric_score) > (len(best_new_units), best_score):
                best_index, best_new_units, best_score = index, new_units, numeric_score
        if not best_new_units:
            break
        selected.append(remaining.pop(best_index))
        uncovered_units -= best_new_units

    for chunk in remaining:
        if len(selected) >= n_results:
            break
        selected.append(chunk)
    return selected


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
    verification_enabled: bool = False,
    verification_pool_multiplier: int = 4,
    verification_recovery_results: int = 3,
    verification_max_keypoints: int = 8,
    verification_llm_model: str = DEFAULT_LLM_MODEL,
    verification_llm_temperature: float = 0.0,
    verification_llm_context_chars: int = 16_000,
    year_branching_enabled: bool = False,
    year_branch_candidates: int = 20,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant Chroma chunks for a query.

    Returns dictionaries with id, text, metadata, distance, and score. Chroma
    cosine distances are lower-is-better; score is 1 - distance for convenience.

    When ``verification_enabled`` is true, the first retrieval is expanded into
    a candidate pool, but verification is applied only after the final selected
    chunks have been chosen. Missing company/year evidence receives targeted
    recovery searches. One final Ollama call semantically audits the requested
    concepts in the exact context that an answer model will receive.

    When ``year_branching_enabled`` is true, each explicit year is searched
    independently for ``year_branch_candidates`` chunks. The branch candidates
    are merged, reranked against the original question, and reduced to the
    final ``n_results`` chunks without invoking an LLM verifier.
    """
    seed_everything(seed)
    if not query.strip():
        return []
    if n_results <= 0:
        raise ValueError("n_results must be positive")
    if verification_pool_multiplier <= 0:
        raise ValueError("verification_pool_multiplier must be positive")
    if verification_recovery_results <= 0:
        raise ValueError("verification_recovery_results must be positive")
    if verification_max_keypoints < 0:
        raise ValueError("verification_max_keypoints cannot be negative")
    if verification_llm_context_chars <= 0:
        raise ValueError("verification_llm_context_chars must be positive")
    if year_branch_candidates <= 0:
        raise ValueError("year_branch_candidates must be positive")

    retrieval_started_at = time.perf_counter()
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
    requirements = (
        extract_query_verification_requirements(
            query,
            corpus_path=corpus_path,
            db_dir=db_dir,
            max_keypoints=verification_max_keypoints,
        )
        if verification_enabled or year_branching_enabled
        else None
    )
    initial_request_size = min(
        n_results * verification_pool_multiplier if verification_enabled else n_results,
        collection_count,
    )

    def _query(
        active_where: dict[str, Any] | None,
        *,
        query_text: str = query,
        request_size: int = initial_request_size,
    ) -> dict[str, Any]:
        query_kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": min(request_size, collection_count),
            "include": ["documents", "metadatas", "distances"],
        }
        if active_where is not None:
            query_kwargs["where"] = active_where
        if where_document is not None:
            query_kwargs["where_document"] = where_document
        return collection.query(**query_kwargs)

    def _chunks_from_results(results: dict[str, Any], retrieval_pass: str) -> list[dict[str, Any]]:
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        result_chunks: list[dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            result_chunks.append(
                {
                    "id": chunk_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distance,
                    "score": None if distance is None else 1.0 - float(distance),
                    "retrieval_pass": retrieval_pass,
                }
            )
        return result_chunks

    if year_branching_enabled and requirements and requirements.get("years"):
        branch_years = [int(year) for year in requirements["years"]]
        branch_base_where = _merge_where_filters(
            where,
            _company_where_condition(
                [str(ticker) for ticker in requirements.get("company_tickers", [])]
            ),
        )
        branch_chunks_by_id: dict[str, dict[str, Any]] = {}
        for year in branch_years:
            branch_where = _merge_where_filters(
                branch_base_where,
                _year_where_condition([year]),
            )
            branch_results = _query(
                branch_where,
                request_size=year_branch_candidates,
            )
            branch_ids = branch_results.get("ids", [[]])[0]
            if not branch_ids and auto_where is not None and not strict_metadata_filter:
                # Preserve the original backend's non-strict behavior, while
                # retaining the year requirement for a useful recovery branch.
                branch_results = _query(
                    _merge_where_filters(where, _year_where_condition([year])),
                    request_size=year_branch_candidates,
                )
            for chunk in _chunks_from_results(branch_results, f"year-branch:{year}"):
                chunk_id = str(chunk.get("id") or "")
                if not chunk_id:
                    continue
                existing = branch_chunks_by_id.get(chunk_id)
                if existing is None:
                    chunk["year_branches"] = [year]
                    branch_chunks_by_id[chunk_id] = chunk
                    continue
                existing_years = set(existing.get("year_branches") or [])
                existing_years.add(year)
                existing["year_branches"] = sorted(existing_years)
                existing_score = _optional_float(existing.get("score"))
                incoming_score = _optional_float(chunk.get("score"))
                if incoming_score is not None and (
                    existing_score is None or incoming_score > existing_score
                ):
                    existing.update(
                        {
                            "text": chunk.get("text", ""),
                            "metadata": chunk.get("metadata", {}),
                            "distance": chunk.get("distance"),
                            "score": incoming_score,
                            "retrieval_pass": chunk.get("retrieval_pass"),
                        }
                    )

        merged_candidates = list(branch_chunks_by_id.values())
        merged_candidates.sort(
            key=lambda chunk: (
                -(
                    _optional_float(chunk.get("score"))
                    if _optional_float(chunk.get("score")) is not None
                    else float("-inf")
                ),
                str(chunk.get("id") or ""),
            )
        )
        # Reserve one strong chunk for each requested year before filling the
        # remaining slots by the common original-query similarity score.
        chunks = []
        selected_ids: set[str] = set()
        for year in branch_years:
            for chunk in merged_candidates:
                chunk_id = str(chunk.get("id") or "")
                if chunk_id in selected_ids or year not in (chunk.get("year_branches") or []):
                    continue
                chunks.append(chunk)
                selected_ids.add(chunk_id)
                break
        for chunk in merged_candidates:
            if len(chunks) >= n_results:
                break
            chunk_id = str(chunk.get("id") or "")
            if chunk_id and chunk_id not in selected_ids:
                chunks.append(chunk)
                selected_ids.add(chunk_id)
        chunks = chunks[:n_results]
    else:
        results = _query(effective_where)
        ids = results.get("ids", [[]])[0]
        if auto_where is not None and not ids and not strict_metadata_filter:
            results = _query(where)
        chunks = _chunks_from_results(results, "initial")

    if year_branching_enabled:
        return chunks
    if not verification_enabled:
        return chunks

    if requirements is None:
        raise RuntimeError("Verification requirements were not initialized.")
    initial_final_chunks = _select_chunks_for_evidence(
        chunks,
        n_results=n_results,
        requirements=requirements,
        include_keypoints=False,
    )
    initial_final_verification = verify_retrieved_evidence(
        query,
        initial_final_chunks,
        corpus_path=corpus_path,
        db_dir=db_dir,
        max_keypoints=verification_max_keypoints,
        requirements=requirements,
    )
    candidates = list(chunks)
    recovery_queries: list[dict[str, Any]] = []

    def _run_recovery(
        label: str,
        recovery_query: str,
        recovery_where: dict[str, Any] | None,
    ) -> None:
        recovery_results = _query(
            recovery_where,
            query_text=recovery_query,
            request_size=verification_recovery_results,
        )
        recovered_chunks = _chunks_from_results(recovery_results, f"recovery:{label}")
        candidates.extend(recovered_chunks)
        recovery_queries.append(
            {
                "label": label,
                "query": recovery_query,
                "chunks_returned": len(recovered_chunks),
            }
        )

    initial_coverage = initial_final_verification["coverage"]
    recovery_company_filter = _or_filter(
        [
            {"company_ticker": {"$eq": ticker}}
            for ticker in requirements.get("company_tickers", [])
        ]
    )
    recovery_base_where = _merge_where_filters(where, recovery_company_filter)

    if initial_coverage["company_tickers_missing"]:
        _run_recovery(
            "company",
            f"{query}\n\nFocus on the requested company evidence.",
            recovery_base_where,
        )

    for year in initial_coverage["years_missing"]:
        year_where = _merge_where_filters(
            recovery_base_where,
            _year_where_condition([int(year)]),
        )
        _run_recovery(
            f"year-{year}",
            f"{query}\n\nFocus on evidence for reporting year {year}.",
            year_where,
        )

    selected_chunks = _select_chunks_for_evidence(
        candidates,
        n_results=n_results,
        requirements=requirements,
        include_keypoints=False,
    )
    verification = verify_retrieved_evidence(
        query,
        selected_chunks,
        corpus_path=corpus_path,
        db_dir=db_dir,
        max_keypoints=verification_max_keypoints,
        requirements=requirements,
    )
    lexical_coverage = dict(verification["coverage"])
    llm_keypoint_verification = verify_final_keypoints_with_ollama(
        query,
        selected_chunks,
        keypoints=[str(keypoint) for keypoint in requirements.get("keypoints", [])],
        llm_model=verification_llm_model,
        ollama_url=ollama_url,
        temperature=verification_llm_temperature,
        max_context_chars=verification_llm_context_chars,
        seed=seed,
    )
    verification["coverage"]["keypoints_found"] = llm_keypoint_verification["keypoints_found"]
    verification["coverage"]["keypoints_missing"] = llm_keypoint_verification["keypoints_missing"]
    verification["is_sufficient"] = bool(
        verification["metadata_sufficient"]
        and llm_keypoint_verification["is_sufficient"]
    )
    verification["initial_final_coverage"] = initial_coverage
    verification["lexical_coverage"] = lexical_coverage
    verification["llm_keypoint_verification"] = llm_keypoint_verification
    verification["candidate_chunks"] = len(_deduplicate_chunks(candidates))
    verification["final_chunks"] = len(selected_chunks)
    verification["recovery_queries"] = recovery_queries
    verification["retrieval_seconds"] = time.perf_counter() - retrieval_started_at
    for chunk in selected_chunks:
        chunk["verification"] = verification
    return selected_chunks


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
    json_schema: dict[str, Any] | None = None,
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
        payload["format"] = json_schema or "json"
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


def _build_keyword_verification_prompt(
    query: str,
    keypoints: list[str],
    chunks: list[dict[str, Any]],
    *,
    max_context_chars: int,
) -> tuple[str, set[str]]:
    """Build a bounded, source-labelled prompt for final-context keyword auditing."""
    if max_context_chars <= 0:
        raise ValueError("verification_llm_context_chars must be positive")

    source_ids: set[str] = set()
    blocks: list[str] = []
    remaining = max_context_chars
    chunk_limit = max(500, min(1_400, max_context_chars // max(len(chunks), 1)))
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id or remaining <= 0:
            continue
        metadata = chunk.get("metadata") or {}
        title = str(metadata.get("title") or "")
        text = str(chunk.get("text") or "").strip()
        if not text:
            continue
        text = text[: min(chunk_limit, remaining)].rsplit(" ", 1)[0].rstrip()
        block = f"CHUNK_ID: {chunk_id}\nTITLE: {title}\nTEXT: {text}"
        blocks.append(block)
        source_ids.add(chunk_id)
        remaining -= len(block) + 5

    context = "\n\n---\n\n".join(blocks) or "No retrieved evidence was supplied."
    prompt = (
        "You are an evidence verifier for financial filing retrieval.\n"
        "Assess whether the retrieved chunks, taken together, contain support for each "
        "requested concept in the question. A synonym or a more specific expression may "
        "support a concept (for example, management actions can support leadership), but do "
        "not infer facts not present in the chunks.\n"
        "Return ONLY valid JSON using this schema:\n"
        "{\n"
        '  "keyword_checks": [\n'
        '    {"keyword": "one requested concept", "supported": true, '
        '"supporting_chunk_ids": ["exact CHUNK_ID"], "reason": "short explanation"}\n'
        "  ]\n"
        "}\n"
        "Include exactly one check for each requested concept. Use only CHUNK_ID values that "
        "appear in the evidence. If support is absent, set supported to false and use an empty "
        "supporting_chunk_ids list.\n\n"
        f"QUESTION:\n{query.strip()}\n\n"
        f"REQUESTED CONCEPTS:\n{json.dumps(keypoints, ensure_ascii=False)}\n\n"
        f"RETRIEVED EVIDENCE:\n{context}\n\n"
        "FINAL OUTPUT REQUIREMENT: Do not copy, summarize, or return a retrieved chunk. "
        "Return one JSON object with the single top-level key keyword_checks. The only permitted "
        "keyword values are the exact strings in this final list; do not add concepts from the "
        "evidence: "
        f"{json.dumps(keypoints, ensure_ascii=False)}"
    )
    return prompt, source_ids


def verify_final_keypoints_with_ollama(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    keypoints: list[str],
    llm_model: str = DEFAULT_LLM_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    max_context_chars: int = 16_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Use one LLM call to semantically check concepts in the final selected context."""
    started_at = time.perf_counter()
    requested = [str(keypoint) for keypoint in keypoints if str(keypoint).strip()]
    if not requested:
        return {
            "enabled": True,
            "status": "not_needed",
            "model": llm_model,
            "keypoints_found": [],
            "keypoints_missing": [],
            "checks": [],
            "is_sufficient": True,
            "verification_seconds": time.perf_counter() - started_at,
        }

    prompt, valid_chunk_ids = _build_keyword_verification_prompt(
        query,
        requested,
        chunks,
        max_context_chars=max_context_chars,
    )
    response_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "keyword_checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "enum": requested},
                        "supported": {"type": "boolean"},
                        "supporting_chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "keyword",
                        "supported",
                        "supporting_chunk_ids",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["keyword_checks"],
        "additionalProperties": False,
    }
    try:
        raw_response = _generate_with_ollama(
            prompt,
            model=llm_model,
            ollama_url=ollama_url,
            temperature=temperature,
            seed=seed,
            json_mode=True,
            json_schema=response_schema,
        )
        payload = _extract_json_object(raw_response)
    except Exception as exc:  # noqa: BLE001 - answer retrieval remains usable if verification is unavailable.
        return {
            "enabled": True,
            "status": "error",
            "model": llm_model,
            "error": f"{type(exc).__name__}: {exc}",
            "keypoints_found": [],
            "keypoints_missing": requested,
            "checks": [],
            "is_sufficient": False,
            "verification_seconds": time.perf_counter() - started_at,
        }

    raw_checks = (
        payload.get("keyword_checks")
        or payload.get("keypoint_checks")
        or payload.get("checks")
        or []
    )
    checks_by_keypoint: dict[str, dict[str, Any]] = {}
    requested_by_normalized = {
        _normalize_verification_term(keypoint): keypoint for keypoint in requested
    }
    if isinstance(raw_checks, list):
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            raw_keyword = str(
                item.get("keyword")
                or item.get("keypoint")
                or item.get("concept")
                or item.get("term")
                or ""
            )
            normalized_keyword = _normalize_verification_term(raw_keyword)
            matched_keypoints = [
                requested_by_normalized[normalized_keyword]
            ] if normalized_keyword in requested_by_normalized else []
            if not matched_keypoints:
                raw_keyword_terms = _verification_terms(raw_keyword)
                matched_keypoints = [
                    keypoint
                    for normalized, keypoint in requested_by_normalized.items()
                    if normalized in raw_keyword_terms
                ]
            if not matched_keypoints:
                continue
            raw_source_ids = (
                item.get("supporting_chunk_ids")
                or item.get("chunk_ids")
                or item.get("evidence_chunk_ids")
                or item.get("supporting_chunks")
                or []
            )
            if isinstance(raw_source_ids, (str, int)):
                raw_source_ids = [raw_source_ids]
            if not isinstance(raw_source_ids, list):
                raw_source_ids = []
            source_ids = [
                str(chunk_id)
                for chunk_id in raw_source_ids
                if str(chunk_id) in valid_chunk_ids
            ]
            supported = bool(item.get("supported")) and bool(source_ids)
            for keypoint in matched_keypoints:
                if keypoint in checks_by_keypoint:
                    continue
                checks_by_keypoint[keypoint] = {
                    "keypoint": keypoint,
                    "supported": supported,
                    "supporting_chunk_ids": source_ids if supported else [],
                    "reason": str(item.get("reason") or "").strip(),
                }

    checks = [
        checks_by_keypoint.get(
            keypoint,
            {
                "keypoint": keypoint,
                "supported": False,
                "supporting_chunk_ids": [],
                "reason": "The verifier did not return a usable check for this concept.",
            },
        )
        for keypoint in requested
    ]
    found = [check["keypoint"] for check in checks if check["supported"]]
    missing = [check["keypoint"] for check in checks if not check["supported"]]
    return {
        "enabled": True,
        "status": "ok",
        "model": llm_model,
        "keypoints_found": found,
        "keypoints_missing": missing,
        "checks": checks,
        "is_sufficient": not missing,
        "verification_seconds": time.perf_counter() - started_at,
    }


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
    verification_enabled: bool = False,
    verification_pool_multiplier: int = 4,
    verification_recovery_results: int = 3,
    verification_max_keypoints: int = 8,
    verification_llm_model: str = DEFAULT_LLM_MODEL,
    verification_llm_temperature: float = 0.0,
    verification_llm_context_chars: int = 16_000,
    year_branching_enabled: bool = False,
    year_branch_candidates: int = 20,
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
        verification_enabled=verification_enabled,
        verification_pool_multiplier=verification_pool_multiplier,
        verification_recovery_results=verification_recovery_results,
        verification_max_keypoints=verification_max_keypoints,
        verification_llm_model=verification_llm_model,
        verification_llm_temperature=verification_llm_temperature,
        verification_llm_context_chars=verification_llm_context_chars,
        year_branching_enabled=year_branching_enabled,
        year_branch_candidates=year_branch_candidates,
    )
    result = generate_answer_from_chunks(
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
    if verification_enabled:
        result["retrieval_verification"] = (
            chunks[0].get("verification") if chunks else verify_retrieved_evidence(
                question,
                [],
                corpus_path=corpus_path,
                db_dir=db_dir,
                max_keypoints=verification_max_keypoints,
            )
        )
    return result


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
    parser.add_argument(
        "--verify-retrieval",
        action="store_true",
        help=(
            "Verify the final selected chunks: deterministically check company/year coverage, "
            "then use Ollama to semantically check requested concepts."
        ),
    )
    parser.add_argument(
        "--verification-pool-multiplier",
        type=int,
        default=4,
        help="Initial candidate-pool size as a multiple of --top-k when verification is enabled.",
    )
    parser.add_argument(
        "--verification-recovery-results",
        type=int,
        default=3,
        help="Chunks requested by each focused company/year recovery search.",
    )
    parser.add_argument(
        "--verification-max-keypoints",
        type=int,
        default=8,
        help="Maximum meaningful query concepts checked by the final LLM verifier.",
    )
    parser.add_argument(
        "--verification-llm-model",
        default=None,
        help="Ollama model for the final semantic evidence check. Defaults to --llm-model.",
    )
    parser.add_argument(
        "--verification-llm-context-chars",
        type=int,
        default=16_000,
        help="Maximum final-context characters supplied to the verification LLM.",
    )
    parser.add_argument(
        "--year-branching",
        action="store_true",
        help="Retrieve candidates separately for each explicit query year, then merge and rerank them.",
    )
    parser.add_argument(
        "--year-branch-candidates",
        type=int,
        default=20,
        help="Candidate chunks retrieved per explicit year with --year-branching.",
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
            verification_enabled=args.verify_retrieval,
            verification_pool_multiplier=args.verification_pool_multiplier,
            verification_recovery_results=args.verification_recovery_results,
            verification_max_keypoints=args.verification_max_keypoints,
            verification_llm_model=args.verification_llm_model or args.llm_model,
            verification_llm_context_chars=args.verification_llm_context_chars,
            year_branching_enabled=args.year_branching,
            year_branch_candidates=args.year_branch_candidates,
        )
        verification = chunks[0].get("verification") if chunks else None
        for chunk in chunks:
            printable_chunk = {key: value for key, value in chunk.items() if key != "verification"}
            print(json.dumps(printable_chunk, ensure_ascii=False, indent=2))
            print("\n---\n")
        if verification is not None:
            print("Retrieval verification:")
            print(json.dumps(verification, ensure_ascii=False, indent=2))

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
            verification_enabled=args.verify_retrieval,
            verification_pool_multiplier=args.verification_pool_multiplier,
            verification_recovery_results=args.verification_recovery_results,
            verification_max_keypoints=args.verification_max_keypoints,
            verification_llm_model=args.verification_llm_model or args.llm_model,
            verification_llm_context_chars=args.verification_llm_context_chars,
            year_branching_enabled=args.year_branching,
            year_branch_candidates=args.year_branch_candidates,
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
            "sources": sources,
            "retrieval_verification": result.get("retrieval_verification"),
        }
        with (ROOT_DIR / "data.json").open("w", encoding="utf-8") as f:
            json.dump(datajson, f, ensure_ascii=False, indent=2)
        print(json.dumps(datajson, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
