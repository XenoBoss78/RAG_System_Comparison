from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:
    from .reproducibility import DEFAULT_SEED, seed_everything
except ImportError:
    from reproducibility import DEFAULT_SEED, seed_everything


CRAG_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CRAG_DIR.parent

DEFAULT_SOURCE_PATH = WORKSPACE_ROOT / "Fin-RATE" / "corpus" / "corpus" / "corpus.jsonl"
DEFAULT_INDEX_DIR = CRAG_DIR / "ollama_crag_index"

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_LLM_MODEL)

CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
CONFIG_FILE = "config.json"
STATS_FILE = "build_stats.json"

DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 12
DEFAULT_CORRECTIVE_TOP_K = 8
DEFAULT_MAX_CONTEXT_CHARS = 14000

SUPPORTED_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".html", ".htm"}
SUPPORTED_STRUCTURED_SUFFIXES = {".jsonl", ".json", ".csv", ".tsv"}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE_PATTERN = re.compile(r"\s+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CRAGIndex:
    embeddings: np.ndarray
    chunks: list[dict[str, Any]]
    config: dict[str, Any]
    index_dir: Path


@dataclass(frozen=True)
class RetrievalEvaluation:
    label: str
    confidence: float
    reason: str
    useful_chunk_ranks: list[int]
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "reason": self.reason,
            "useful_chunk_ranks": self.useful_chunk_ranks,
            "raw_response": self.raw_response,
        }


@dataclass(frozen=True)
class CRAGAnswer:
    question: str
    answer: str
    retrieval_evaluation: RetrievalEvaluation
    initial_chunks: list[dict[str, Any]]
    corrected_chunks: list[dict[str, Any]]
    refined_context: list[dict[str, Any]]
    rewritten_queries: list[str]
    verification: dict[str, Any] | None
    trace: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieval_evaluation": self.retrieval_evaluation.to_dict(),
            "initial_chunks": self.initial_chunks,
            "corrected_chunks": self.corrected_chunks,
            "refined_context": self.refined_context,
            "rewritten_queries": self.rewritten_queries,
            "verification": self.verification,
            "trace": self.trace,
        }


class OllamaError(RuntimeError):
    """Raised when the local Ollama server cannot complete a request."""


class OllamaClient:
    """Small HTTP client for a local Ollama server."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_OLLAMA_HOST,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        timeout_seconds: int = 300,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.host = host.rstrip("/")
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.timeout_seconds = timeout_seconds
        self.seed = seed

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.host}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(
                f"Ollama returned HTTP {exc.code} for {endpoint}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                "Could not reach Ollama at "
                f"{self.host}. Start Ollama and make sure the model exists, e.g. "
                f"`ollama pull {self.llm_model}`."
            ) from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned a non-JSON response.") from exc

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        temperature: float = 0.1,
        num_predict: int = 512,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "seed": self.seed,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            response = self._post_json("/api/generate", payload)
        except OllamaError:
            if not json_mode:
                raise
            payload.pop("format", None)
            response = self._post_json("/api/generate", payload)

        return str(response.get("response", "")).strip()

    def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        embedding_model = model or self.embedding_model
        payload = {"model": embedding_model, "input": list(texts)}
        try:
            response = self._post_json("/api/embed", payload)
            embeddings = response.get("embeddings")
            if embeddings is None and "embedding" in response:
                embeddings = [response["embedding"]]
            if not isinstance(embeddings, list):
                raise OllamaError("Ollama /api/embed response did not include embeddings.")
            return embeddings
        except OllamaError as first_error:
            # Older Ollama versions expose /api/embeddings and accept one prompt at a time.
            embeddings: list[list[float]] = []
            try:
                for text in texts:
                    response = self._post_json(
                        "/api/embeddings",
                        {"model": embedding_model, "prompt": text},
                    )
                    embedding = response.get("embedding")
                    if not isinstance(embedding, list):
                        raise OllamaError(
                            "Ollama /api/embeddings response did not include an embedding."
                        )
                    embeddings.append(embedding)
                return embeddings
            except OllamaError as second_error:
                raise OllamaError(
                    "Could not create embeddings with Ollama. If llama3.1:8b is not "
                    "available for embeddings on your Ollama build, install a local "
                    "embedding model such as `ollama pull nomic-embed-text` and pass "
                    "`embedding_model='nomic-embed-text'`."
                ) from second_error or first_error


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(text: str) -> str:
    """Normalize text before chunking and embedding."""
    text = CONTROL_CHAR_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _token_matches(text: str) -> list[re.Match[str]]:
    return list(TOKEN_PATTERN.finditer(text))


def _token_count(text: str) -> int:
    return len(_token_matches(text))


def chunk_text(
    text: str,
    *,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[dict[str, Any]]:
    """Split text into overlapping token windows."""
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens cannot be negative")
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    text = clean_text(text)
    matches = _token_matches(text)
    if not matches:
        return [{"text": text, "token_start": 0, "token_end": 0}] if text else []

    chunks: list[dict[str, Any]] = []
    step = max(1, chunk_size_tokens - chunk_overlap_tokens)
    for token_start in range(0, len(matches), step):
        token_end = min(len(matches), token_start + chunk_size_tokens)
        char_start = matches[token_start].start()
        char_end = matches[token_end - 1].end()
        chunk = text[char_start:char_end].strip()
        if chunk:
            chunks.append(
                {
                    "text": chunk,
                    "token_start": token_start,
                    "token_end": token_end,
                }
            )
        if token_end >= len(matches):
            break
    return chunks


def _source_paths(sources: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(sources, (str, Path)):
        raw_sources = [sources]
    else:
        raw_sources = list(sources)

    paths: list[Path] = []
    for source in raw_sources:
        path = Path(source)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in (
                    SUPPORTED_TEXT_SUFFIXES | SUPPORTED_STRUCTURED_SUFFIXES
                ):
                    paths.append(child)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Source path not found: {path}")
    return paths


def _first_text_field(record: dict[str, Any]) -> str:
    for key in ("text", "content", "contents", "body", "page_content", "document"):
        value = record.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return json.dumps(value, ensure_ascii=False)
    return json.dumps(record, ensure_ascii=False)


def _document_from_record(
    record: dict[str, Any],
    *,
    fallback_id: str,
    source: Path,
) -> SourceDocument:
    doc_id = str(
        record.get("_id")
        or record.get("id")
        or record.get("doc_id")
        or record.get("document_id")
        or fallback_id
    )
    title = record.get("title") or record.get("name") or record.get("heading") or ""
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"text", "content", "contents", "body", "page_content", "document"}
    }
    metadata["source_path"] = str(source)
    return SourceDocument(
        doc_id=doc_id,
        text=clean_text(_first_text_field(record)),
        title=str(title),
        source=str(source),
        metadata=metadata,
    )


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item if isinstance(item, dict) else {"text": item} for item in data]
    if isinstance(data, dict):
        for key in ("documents", "items", "records", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"text": item} for item in value]
        return [data]
    return [{"text": data}]


def _iter_documents_from_file(path: Path) -> Iterable[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    record = {"text": record}
                yield _document_from_record(
                    record,
                    fallback_id=f"{path.stem}-{line_number}",
                    source=path,
                )
        return

    if suffix == ".json":
        for index, record in enumerate(_read_json_records(path), start=1):
            yield _document_from_record(
                record,
                fallback_id=f"{path.stem}-{index}",
                source=path,
            )
        return

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            for row_number, row in enumerate(reader, start=1):
                yield _document_from_record(
                    dict(row),
                    fallback_id=f"{path.stem}-{row_number}",
                    source=path,
                )
        return

    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        text = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", text, flags=re.I)
        text = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    yield SourceDocument(
        doc_id=path.stem,
        text=clean_text(text),
        title=path.stem,
        source=str(path),
        metadata={"source_path": str(path)},
    )


def load_source_documents(sources: str | Path | Sequence[str | Path]) -> list[SourceDocument]:
    """Load source files into normalized SourceDocument records."""
    documents: list[SourceDocument] = []
    for path in _source_paths(sources):
        documents.extend(_iter_documents_from_file(path))
    return [document for document in documents if document.text]


def prepare_chunks(
    documents: Sequence[SourceDocument],
    *,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Attach metadata and chunk loaded documents for retrieval."""
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for chunk_index, chunk in enumerate(
            chunk_text(
                document.text,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        ):
            chunk_id = f"{document.doc_id}::chunk_{chunk_index:04d}"
            chunks.append(
                {
                    "id": chunk_id,
                    "doc_id": document.doc_id,
                    "title": document.title,
                    "source": document.source,
                    "chunk_index": chunk_index,
                    "token_start": chunk["token_start"],
                    "token_end": chunk["token_end"],
                    "text": chunk["text"],
                    "metadata": document.metadata,
                }
            )
            if limit is not None and len(chunks) >= limit:
                return chunks
    return chunks


def _embedding_text(chunk: dict[str, Any]) -> str:
    title = str(chunk.get("title", "")).strip()
    text = str(chunk.get("text", "")).strip()
    return f"{title}\n\n{text}" if title else text


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_crag_index(
    sources: str | Path | Sequence[str | Path] = DEFAULT_SOURCE_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    ollama: OllamaClient | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """
    Ingest source data and persist a local vector index.

    Pipeline steps:
    1. load source documents
    2. clean and normalize text
    3. split into overlapping chunks
    4. attach metadata
    5. create embeddings with Ollama
    6. store chunks, vectors, config, and build stats
    """
    seed_everything(seed)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    client = ollama or OllamaClient(embedding_model=embedding_model, seed=seed)
    client.seed = seed

    started_at = _utc_now_iso()
    start_time = time.perf_counter()

    documents = load_source_documents(sources)
    chunks = prepare_chunks(
        documents,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        limit=limit,
    )
    if not chunks:
        raise ValueError("No chunks were created from the provided sources.")

    vectors: list[list[float]] = []
    for batch in _batched([_embedding_text(chunk) for chunk in chunks], batch_size):
        vectors.extend(client.embed(batch, model=embedding_model))

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: got {len(vectors)} embeddings for {len(chunks)} chunks."
        )

    embedding_matrix = _normalize_matrix(np.asarray(vectors, dtype=np.float32))
    np.save(index_path / EMBEDDINGS_FILE, embedding_matrix)
    _write_jsonl(index_path / CHUNKS_FILE, chunks)

    config = {
        "index_type": "ollama_dense_cosine",
        "embedding_model": embedding_model,
        "ollama_host": client.host,
        "seed": seed,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "default_top_k": DEFAULT_TOP_K,
        "chunks_file": CHUNKS_FILE,
        "embeddings_file": EMBEDDINGS_FILE,
        "created_at_utc": started_at,
    }
    stats = {
        **config,
        "source_paths": [str(path) for path in _source_paths(sources)],
        "index_dir": str(index_path),
        "documents_loaded": len(documents),
        "chunks_indexed": len(chunks),
        "embedding_shape": list(embedding_matrix.shape),
        "tokens_indexed": sum(_token_count(chunk["text"]) for chunk in chunks),
        "build_started_at_utc": started_at,
        "build_finished_at_utc": _utc_now_iso(),
        "build_seconds": time.perf_counter() - start_time,
        "pipeline_steps": [
            "load_source_documents",
            "clean_text",
            "chunk_text",
            "attach_metadata",
            "ollama_embed",
            "persist_index",
        ],
    }
    _write_json(index_path / CONFIG_FILE, config)
    _write_json(index_path / STATS_FILE, stats)
    return stats


def load_crag_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> CRAGIndex:
    index_path = Path(index_dir)
    embeddings_path = index_path / EMBEDDINGS_FILE
    chunks_path = index_path / CHUNKS_FILE
    config_path = index_path / CONFIG_FILE

    missing = [
        path
        for path in (embeddings_path, chunks_path, config_path)
        if not path.exists()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"CRAG index is missing required file(s): {names}")

    embeddings = np.load(embeddings_path)
    chunks = _read_jsonl(chunks_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Index has {embeddings.shape[0]} vectors but {len(chunks)} chunks."
        )
    return CRAGIndex(
        embeddings=embeddings,
        chunks=chunks,
        config=config,
        index_dir=index_path,
    )


def _matches_metadata_filter(
    chunk: dict[str, Any],
    metadata_filter: dict[str, Any] | None,
) -> bool:
    if not metadata_filter:
        return True
    metadata = chunk.get("metadata") or {}
    for key, expected in metadata_filter.items():
        actual = chunk.get(key, metadata.get(key))
        if actual != expected:
            return False
    return True


def retrieve_chunks(
    query: str,
    *,
    index: CRAGIndex | None = None,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    ollama: OllamaClient | None = None,
    embedding_model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    metadata_filter: dict[str, Any] | None = None,
    min_score: float | None = None,
) -> list[dict[str, Any]]:
    """Retrieve chunks with cosine similarity over Ollama embeddings."""
    query = query.strip()
    if not query or top_k <= 0:
        return []

    crag_index = index or load_crag_index(index_dir)
    if crag_index.embeddings.size == 0:
        return []

    model_name = embedding_model or crag_index.config.get(
        "embedding_model", DEFAULT_EMBEDDING_MODEL
    )
    client = ollama or OllamaClient(embedding_model=model_name)

    query_vector = np.asarray(client.embed([query], model=model_name)[0], dtype=np.float32)
    query_vector = _normalize_vector(query_vector)
    if query_vector.shape[0] != crag_index.embeddings.shape[1]:
        raise ValueError(
            f"Query embedding has {query_vector.shape[0]} dimensions but the index "
            f"stores {crag_index.embeddings.shape[1]}. Rebuild the index with the "
            "same embedding model used for querying."
        )

    scores = crag_index.embeddings @ query_vector
    eligible_positions = [
        position
        for position, chunk in enumerate(crag_index.chunks)
        if _matches_metadata_filter(chunk, metadata_filter)
        and (min_score is None or float(scores[position]) >= min_score)
    ]
    if not eligible_positions:
        return []

    positions = np.asarray(eligible_positions, dtype=np.int64)
    top_count = min(top_k, positions.size)
    subset_scores = scores[positions]
    ranked = np.lexsort((positions, -subset_scores))
    top_positions = positions[ranked[:top_count]]

    results: list[dict[str, Any]] = []
    for rank, position in enumerate(top_positions, start=1):
        chunk = dict(crag_index.chunks[int(position)])
        chunk["rank"] = rank
        chunk["score"] = float(scores[int(position)])
        results.append(chunk)
    return results


def _extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")

    candidates = [text]
    for fenced in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S):
        candidates.append(fenced.strip())

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end > array_start:
        candidates.append(text[array_start : array_end + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object or array found")


def _coerce_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return max(0.0, min(1.0, result))


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _canonical_label(label: Any) -> str:
    normalized = str(label or "").strip().lower().replace("-", "_")
    if normalized in {"correct", "good", "high", "relevant", "sufficient"}:
        return "correct"
    if normalized in {"ambiguous", "partial", "mixed", "medium", "uncertain"}:
        return "ambiguous"
    if normalized in {"incorrect", "bad", "low", "irrelevant", "insufficient", "none"}:
        return "incorrect"
    return "ambiguous"


def _format_chunk_preview(chunks: Sequence[dict[str, Any]], *, max_chars: int = 1000) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        title = str(chunk.get("title", "")).strip()
        score = chunk.get("score")
        text = str(chunk.get("text", "")).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0].rstrip()
        heading = f"[{index}]"
        if title:
            heading += f" title={title}"
        if score is not None:
            heading += f" score={float(score):.4f}"
        blocks.append(f"{heading}\n{text}")
    return "\n\n---\n\n".join(blocks)


def evaluate_retrieval(
    question: str,
    chunks: Sequence[dict[str, Any]],
    *,
    ollama: OllamaClient | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_chunks: int = 8,
) -> RetrievalEvaluation:
    """
    CRAG retrieval evaluator.

    Labels:
    - correct: retrieved context can answer the question
    - ambiguous: context is partially useful but incomplete or mixed
    - incorrect: context is irrelevant or not enough to answer
    """
    if not chunks:
        return RetrievalEvaluation(
            label="incorrect",
            confidence=1.0,
            reason="No chunks were retrieved.",
            useful_chunk_ranks=[],
        )

    client = ollama or OllamaClient(llm_model=llm_model)
    preview = _format_chunk_preview(list(chunks)[:max_chunks])
    prompt = (
        "Evaluate the retrieval quality for the question.\n"
        "Return JSON only with this schema:\n"
        "{"
        '"label":"correct|ambiguous|incorrect",'
        '"confidence":0.0,'
        '"useful_chunk_ranks":[1],'
        '"reason":"short reason"'
        "}\n\n"
        "Use label=correct only when the chunks directly support a complete answer. "
        "Use label=ambiguous when some chunks are relevant but incomplete, mixed, or weak. "
        "Use label=incorrect when the chunks are off-topic or cannot support an answer.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Retrieved chunks:\n{preview}"
    )
    raw = client.generate(
        prompt,
        model=llm_model,
        system="You are a careful retrieval-quality evaluator for a CRAG pipeline.",
        temperature=0.0,
        num_predict=300,
        json_mode=True,
    )

    try:
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Evaluator JSON was not an object")
        label = _canonical_label(parsed.get("label"))
        confidence = _coerce_float(parsed.get("confidence"), 0.5)
        useful = _coerce_int_list(
            parsed.get("useful_chunk_ranks")
            or parsed.get("useful_chunks")
            or parsed.get("useful_ids")
        )
        useful = [rank for rank in useful if 1 <= rank <= min(max_chunks, len(chunks))]
        reason = str(parsed.get("reason", "")).strip()
    except ValueError:
        scores = [float(chunk.get("score", 0.0)) for chunk in chunks]
        best = max(scores) if scores else 0.0
        label = "correct" if best >= 0.45 else "ambiguous" if best >= 0.25 else "incorrect"
        confidence = min(1.0, max(0.25, abs(best)))
        useful = list(range(1, min(4, len(chunks)) + 1)) if label != "incorrect" else []
        reason = "Fell back to vector-score heuristic because evaluator JSON was not parseable."

    if label in {"correct", "ambiguous"} and not useful:
        useful = list(range(1, min(4, len(chunks)) + 1))

    return RetrievalEvaluation(
        label=label,
        confidence=confidence,
        reason=reason,
        useful_chunk_ranks=useful,
        raw_response=raw,
    )


def rewrite_queries(
    question: str,
    *,
    ollama: OllamaClient | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    num_queries: int = 3,
) -> list[str]:
    """Ask Ollama to generate alternate local-search queries for corrective retrieval."""
    client = ollama or OllamaClient(llm_model=llm_model)
    prompt = (
        "Generate alternate retrieval queries for the user's question. "
        "Prefer entity names, dates, aliases, and keywords that would find evidence "
        "in a local document index. Return JSON only with this schema: "
        '{"queries":["query 1","query 2"]}.\n\n'
        f"Question: {question.strip()}\n"
        f"Number of queries: {num_queries}"
    )
    raw = client.generate(
        prompt,
        model=llm_model,
        system="You rewrite questions into precise retrieval queries.",
        temperature=0.2,
        num_predict=256,
        json_mode=True,
    )
    try:
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            queries = parsed.get("queries", [])
        else:
            queries = parsed
        if not isinstance(queries, list):
            queries = []
        cleaned = []
        for query in queries:
            text = clean_text(str(query))
            if text and text.lower() != question.strip().lower():
                cleaned.append(text)
        return cleaned[:num_queries]
    except ValueError:
        return []


def _lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = {token.group(0).lower() for token in TOKEN_PATTERN.finditer(query)}
    if not query_tokens:
        return 0.0
    text_tokens = {token.group(0).lower() for token in TOKEN_PATTERN.finditer(text)}
    if not text_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return overlap / math.sqrt(len(query_tokens) * len(text_tokens))


def _split_into_strips(
    chunk: dict[str, Any],
    *,
    sentences_per_strip: int = 3,
) -> list[dict[str, Any]]:
    text = str(chunk.get("text", "")).strip()
    if not text:
        return []
    sentences = [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]
    if not sentences:
        sentences = [text]

    strips: list[dict[str, Any]] = []
    for start in range(0, len(sentences), sentences_per_strip):
        strip_text = " ".join(sentences[start : start + sentences_per_strip]).strip()
        if strip_text:
            strips.append(
                {
                    "text": strip_text,
                    "source_chunk_id": chunk.get("id"),
                    "source_rank": chunk.get("rank"),
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("title"),
                    "source": chunk.get("source"),
                    "score": chunk.get("score", 0.0),
                    "metadata": chunk.get("metadata") or {},
                }
            )
    return strips


def refine_knowledge(
    question: str,
    chunks: Sequence[dict[str, Any]],
    *,
    ollama: OllamaClient | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    max_strips: int = 8,
    candidate_limit: int = 36,
) -> list[dict[str, Any]]:
    """
    Decompose retrieved chunks into knowledge strips and keep the useful pieces.

    This mirrors CRAG's decompose-then-recompose step while using llama3.1:8b
    as the selector. A lexical fallback keeps the pipeline usable if selection
    JSON is malformed.
    """
    strips: list[dict[str, Any]] = []
    for chunk in chunks:
        strips.extend(_split_into_strips(chunk))
    if not strips:
        return []

    for strip_id, strip in enumerate(strips, start=1):
        lexical = _lexical_overlap_score(question, strip["text"])
        retrieval_score = max(0.0, float(strip.get("score") or 0.0))
        strip["strip_id"] = strip_id
        strip["_selection_score"] = lexical + 0.15 * retrieval_score

    candidates = sorted(
        strips,
        key=lambda item: item["_selection_score"],
        reverse=True,
    )[:candidate_limit]

    candidate_blocks: list[str] = []
    for strip in candidates:
        text = strip["text"]
        if len(text) > 700:
            text = text[:700].rsplit(" ", 1)[0].rstrip()
        title = str(strip.get("title") or "")
        heading = f"[{strip['strip_id']}]"
        if title:
            heading += f" title={title}"
        candidate_blocks.append(f"{heading}\n{text}")

    client = ollama or OllamaClient(llm_model=llm_model)
    prompt = (
        "Select the smallest set of knowledge strips that directly help answer "
        "the question. Ignore duplicates and off-topic strips. Return JSON only "
        'as {"selected_strip_ids":[1,2],"reason":"short reason"}.\n\n'
        f"Question:\n{question.strip()}\n\n"
        "Candidate strips:\n"
        + "\n\n---\n\n".join(candidate_blocks)
    )
    raw = client.generate(
        prompt,
        model=llm_model,
        system="You select grounded evidence for a CRAG answer generator.",
        temperature=0.0,
        num_predict=256,
        json_mode=True,
    )

    selected_ids: list[int] = []
    try:
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            selected_ids = _coerce_int_list(
                parsed.get("selected_strip_ids")
                or parsed.get("strip_ids")
                or parsed.get("ids")
            )
        elif isinstance(parsed, list):
            selected_ids = _coerce_int_list(parsed)
    except ValueError:
        selected_ids = []

    candidate_ids = {int(strip["strip_id"]) for strip in candidates}
    selected_ids = [strip_id for strip_id in selected_ids if strip_id in candidate_ids]
    if not selected_ids:
        selected_ids = [int(strip["strip_id"]) for strip in candidates[:max_strips]]

    selected_by_id = {int(strip["strip_id"]): strip for strip in strips}
    refined: list[dict[str, Any]] = []
    for output_rank, strip_id in enumerate(selected_ids[:max_strips], start=1):
        strip = dict(selected_by_id[strip_id])
        strip.pop("_selection_score", None)
        strip["context_rank"] = output_rank
        refined.append(strip)
    return refined


def _dedupe_chunks(chunks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for chunk in chunks:
        key = str(chunk.get("id") or clean_text(str(chunk.get("text", "")))[:500])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(chunk))
    for rank, chunk in enumerate(deduped, start=1):
        chunk["rank"] = rank
    return deduped


def _chunks_from_external_results(
    results: Sequence[dict[str, Any] | str],
    *,
    base_rank: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for offset, item in enumerate(results, start=base_rank):
        if isinstance(item, str):
            chunks.append(
                {
                    "id": f"external::{offset}",
                    "doc_id": f"external::{offset}",
                    "title": "external result",
                    "source": "external_search_fn",
                    "text": clean_text(item),
                    "score": 0.0,
                    "rank": offset,
                    "metadata": {},
                }
            )
        elif isinstance(item, dict):
            chunk = dict(item)
            chunk.setdefault("id", f"external::{offset}")
            chunk.setdefault("doc_id", chunk["id"])
            chunk.setdefault("title", "external result")
            chunk.setdefault("source", "external_search_fn")
            chunk.setdefault("score", 0.0)
            chunk.setdefault("rank", offset)
            chunk.setdefault("metadata", {})
            chunk["text"] = clean_text(str(chunk.get("text", chunk.get("content", ""))))
            if chunk["text"]:
                chunks.append(chunk)
    return chunks


def corrective_retrieval(
    question: str,
    initial_chunks: Sequence[dict[str, Any]],
    evaluation: RetrievalEvaluation,
    *,
    index: CRAGIndex,
    ollama: OllamaClient,
    embedding_model: str | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_CORRECTIVE_TOP_K,
    external_search_fn: Callable[[str], Sequence[dict[str, Any] | str]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Route retrieval according to the CRAG evaluator.

    correct: keep useful local chunks.
    ambiguous: keep useful local chunks and add expanded local retrieval.
    incorrect: rewrite the query and retrieve again locally. If external_search_fn
    is supplied, its results are included as the web-search analogue.
    """
    useful = [
        chunk
        for index_position, chunk in enumerate(initial_chunks, start=1)
        if index_position in evaluation.useful_chunk_ranks
    ]

    if evaluation.label == "correct":
        return _dedupe_chunks(useful or list(initial_chunks)), []

    rewritten = rewrite_queries(
        question,
        ollama=ollama,
        llm_model=llm_model,
        num_queries=3,
    )
    expanded: list[dict[str, Any]] = []
    for query in rewritten:
        expanded.extend(
            retrieve_chunks(
                query,
                index=index,
                ollama=ollama,
                embedding_model=embedding_model,
                top_k=top_k,
            )
        )
        if external_search_fn is not None:
            external = external_search_fn(query)
            expanded.extend(
                _chunks_from_external_results(
                    external,
                    base_rank=len(expanded) + len(useful) + 1,
                )
            )

    if evaluation.label == "ambiguous":
        corrected = list(useful or initial_chunks[: max(1, top_k // 2)]) + expanded
    else:
        corrected = expanded or list(initial_chunks)
    return _dedupe_chunks(corrected), rewritten


def build_answer_prompt(
    question: str,
    refined_context: Sequence[dict[str, Any]],
    *,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    context_blocks: list[str] = []
    chars_used = 0
    for rank, item in enumerate(refined_context, start=1):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        title = str(item.get("title") or "").strip()
        source = str(item.get("source") or item.get("doc_id") or item.get("source_chunk_id") or "")
        header = f"[{rank}]"
        if title:
            header += f" title={title}"
        if source:
            header += f" source={source}"

        block = f"{header}\n{text}"
        remaining = max_context_chars - chars_used
        if remaining <= 0:
            break
        if len(block) > remaining:
            block = block[:remaining].rsplit(" ", 1)[0].rstrip()
        context_blocks.append(block)
        chars_used += len(block)

    context = "\n\n---\n\n".join(context_blocks) or "No usable context was found."
    return (
        "Answer the question using only the context below. "
        "If the context does not contain enough evidence, say that the answer "
        "cannot be determined from the provided context. Cite evidence with "
        "bracketed numbers like [1] or [2]. Keep the answer concise.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Answer:"
    )


def generate_grounded_answer(
    question: str,
    refined_context: Sequence[dict[str, Any]],
    *,
    ollama: OllamaClient | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.1,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    client = ollama or OllamaClient(llm_model=llm_model)
    prompt = build_answer_prompt(
        question,
        refined_context,
        max_context_chars=max_context_chars,
    )
    return client.generate(
        prompt,
        model=llm_model,
        system="You are a grounded RAG answer generator. Do not use unsupported facts.",
        temperature=temperature,
        num_predict=700,
    )


def verify_grounding(
    question: str,
    answer: str,
    refined_context: Sequence[dict[str, Any]],
    *,
    ollama: OllamaClient | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> dict[str, Any]:
    client = ollama or OllamaClient(llm_model=llm_model)
    context = build_answer_prompt(
        question,
        refined_context,
        max_context_chars=9000,
    ).split("Question:", 1)[0]
    prompt = (
        "Check whether the answer is fully supported by the context. "
        "Return JSON only with this schema: "
        '{"supported":true,"unsupported_claims":[],"notes":"short notes"}.\n\n'
        f"{context}\n"
        f"Question:\n{question.strip()}\n\n"
        f"Answer:\n{answer.strip()}"
    )
    raw = client.generate(
        prompt,
        model=llm_model,
        system="You verify answer grounding against retrieved context.",
        temperature=0.0,
        num_predict=300,
        json_mode=True,
    )
    try:
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Verification JSON was not an object")
        parsed["raw_response"] = raw
        return parsed
    except ValueError:
        return {
            "supported": None,
            "unsupported_claims": [],
            "notes": "Could not parse grounding verifier response.",
            "raw_response": raw,
        }


def answer_question(
    question: str,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    ollama: OllamaClient | None = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    corrective_top_k: int = DEFAULT_CORRECTIVE_TOP_K,
    max_refined_strips: int = 8,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    temperature: float = 0.1,
    verify: bool = True,
    external_search_fn: Callable[[str], Sequence[dict[str, Any] | str]] | None = None,
    seed: int = DEFAULT_SEED,
) -> CRAGAnswer:
    """
    Run the query-to-answer CRAG pipeline against a persisted local index.

    Pipeline steps:
    1. preprocess the question
    2. retrieve initial chunks
    3. evaluate retrieval quality with Ollama
    4. route through corrective retrieval when needed
    5. decompose and refine knowledge strips with Ollama
    6. generate a grounded answer with Ollama
    7. optionally verify grounding with Ollama
    """
    seed_everything(seed)
    clean_question = clean_text(question)
    if not clean_question:
        raise ValueError("question cannot be empty")

    client = ollama or OllamaClient(
        llm_model=llm_model,
        embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL,
        seed=seed,
    )
    client.seed = seed
    crag_index = load_crag_index(index_dir)
    model_name = embedding_model or crag_index.config.get(
        "embedding_model", DEFAULT_EMBEDDING_MODEL
    )

    trace: list[dict[str, Any]] = [
        {"step": "preprocess_question", "question": clean_question}
    ]
    initial_chunks = retrieve_chunks(
        clean_question,
        index=crag_index,
        ollama=client,
        embedding_model=model_name,
        top_k=top_k,
    )
    trace.append({"step": "initial_retrieval", "chunks": len(initial_chunks)})

    evaluation = evaluate_retrieval(
        clean_question,
        initial_chunks,
        ollama=client,
        llm_model=llm_model,
    )
    trace.append(
        {
            "step": "retrieval_evaluation",
            "label": evaluation.label,
            "confidence": evaluation.confidence,
        }
    )

    corrected_chunks, rewritten = corrective_retrieval(
        clean_question,
        initial_chunks,
        evaluation,
        index=crag_index,
        ollama=client,
        embedding_model=model_name,
        llm_model=llm_model,
        top_k=corrective_top_k,
        external_search_fn=external_search_fn,
    )
    trace.append(
        {
            "step": "corrective_retrieval",
            "chunks": len(corrected_chunks),
            "rewritten_queries": rewritten,
        }
    )

    refined_context = refine_knowledge(
        clean_question,
        corrected_chunks,
        ollama=client,
        llm_model=llm_model,
        max_strips=max_refined_strips,
    )
    trace.append({"step": "knowledge_refinement", "strips": len(refined_context)})

    answer = generate_grounded_answer(
        clean_question,
        refined_context,
        ollama=client,
        llm_model=llm_model,
        temperature=temperature,
        max_context_chars=max_context_chars,
    )
    trace.append({"step": "answer_generation", "model": llm_model})

    verification = None
    if verify:
        verification = verify_grounding(
            clean_question,
            answer,
            refined_context,
            ollama=client,
            llm_model=llm_model,
        )
        trace.append(
            {
                "step": "grounding_verification",
                "supported": verification.get("supported"),
            }
        )

    return CRAGAnswer(
        question=clean_question,
        answer=answer,
        retrieval_evaluation=evaluation,
        initial_chunks=initial_chunks,
        corrected_chunks=corrected_chunks,
        refined_context=refined_context,
        rewritten_queries=rewritten,
        verification=verification,
        trace=trace,
    )


class OllamaCRAGPipeline:
    """Convenience wrapper for ingestion and question answering."""

    def __init__(
        self,
        *,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.ollama = OllamaClient(
            host=ollama_host,
            llm_model=llm_model,
            embedding_model=embedding_model,
            seed=seed,
        )
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.seed = seed

    def ingest(
        self,
        sources: str | Path | Sequence[str | Path] = DEFAULT_SOURCE_PATH,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.setdefault("seed", self.seed)
        return build_crag_index(
            sources=sources,
            index_dir=self.index_dir,
            ollama=self.ollama,
            embedding_model=self.embedding_model,
            **kwargs,
        )

    def ask(self, question: str, **kwargs: Any) -> CRAGAnswer:
        kwargs.setdefault("seed", self.seed)
        return answer_question(
            question,
            index_dir=self.index_dir,
            ollama=self.ollama,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model,
            **kwargs,
        )

    def retrieve(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        index = load_crag_index(self.index_dir)
        return retrieve_chunks(
            query,
            index=index,
            ollama=self.ollama,
            embedding_model=self.embedding_model,
            **kwargs,
        )


def ingest_data(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Alias for build_crag_index."""
    return build_crag_index(*args, **kwargs)


def query_to_answer_pipeline(*args: Any, **kwargs: Any) -> CRAGAnswer:
    """Alias for answer_question."""
    return answer_question(*args, **kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Ollama implementation of a CRAG-style RAG pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build a local CRAG index.")
    ingest_parser.add_argument(
        "--source",
        action="append",
        type=Path,
        default=None,
        help="File or directory to ingest. Can be provided multiple times.",
    )
    ingest_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ingest_parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    ingest_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ingest_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ingest_parser.add_argument("--chunk-size-tokens", type=int, default=DEFAULT_CHUNK_SIZE_TOKENS)
    ingest_parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_TOKENS,
    )
    ingest_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional chunk limit for a quick smoke-test index.",
    )
    ingest_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve chunks for a query.")
    retrieve_parser.add_argument("query")
    retrieve_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    retrieve_parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    retrieve_parser.add_argument("--embedding-model", default=None)
    retrieve_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    retrieve_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    ask_parser = subparsers.add_parser("ask", help="Run the full CRAG answer pipeline.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ask_parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    ask_parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    ask_parser.add_argument("--embedding-model", default=None)
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ask_parser.add_argument("--corrective-top-k", type=int, default=DEFAULT_CORRECTIVE_TOP_K)
    ask_parser.add_argument("--max-refined-strips", type=int, default=8)
    ask_parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    ask_parser.add_argument("--temperature", type=float, default=0.1)
    ask_parser.add_argument("--no-verify", action="store_true")
    ask_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    seed_everything(args.seed)

    if args.command == "ingest":
        client = OllamaClient(
            host=args.ollama_host,
            embedding_model=args.embedding_model,
            seed=args.seed,
        )
        sources = args.source or [DEFAULT_SOURCE_PATH]
        stats = build_crag_index(
            sources=sources,
            index_dir=args.index_dir,
            ollama=client,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            chunk_size_tokens=args.chunk_size_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            limit=args.limit,
            seed=args.seed,
        )
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    if args.command == "retrieve":
        index = load_crag_index(args.index_dir)
        model_name = args.embedding_model or index.config.get(
            "embedding_model", DEFAULT_EMBEDDING_MODEL
        )
        client = OllamaClient(
            host=args.ollama_host,
            embedding_model=model_name,
            seed=args.seed,
        )
        chunks = retrieve_chunks(
            args.query,
            index=index,
            ollama=client,
            embedding_model=model_name,
            top_k=args.top_k,
        )
        print(json.dumps(chunks, indent=2, ensure_ascii=False))
        return

    if args.command == "ask":
        client = OllamaClient(
            host=args.ollama_host,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model or DEFAULT_EMBEDDING_MODEL,
            seed=args.seed,
        )
        result = answer_question(
            args.question,
            index_dir=args.index_dir,
            ollama=client,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            corrective_top_k=args.corrective_top_k,
            max_refined_strips=args.max_refined_strips,
            max_context_chars=args.max_context_chars,
            temperature=args.temperature,
            verify=not args.no_verify,
            seed=args.seed,
        )
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
