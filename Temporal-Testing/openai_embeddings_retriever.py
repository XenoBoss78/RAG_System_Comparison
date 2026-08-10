from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from .reproducibility import DEFAULT_SEED, seed_everything
except ImportError:
    from reproducibility import DEFAULT_SEED, seed_everything


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "Fin-RATE" / "corpus" / "corpus" / "corpus.jsonl"
LEGACY_VECTOR_DB_DIR = PROJECT_ROOT / "Fin-RATE" / "vector_db"
DEFAULT_OPENAI_VECTOR_DB_DIR = PROJECT_ROOT / "Fin-RATE" / "openai_vector_db"
DEFAULT_INDEX_DIR = DEFAULT_OPENAI_VECTOR_DB_DIR

EMBEDDINGS_FILE = "embeddings.npy"
DOCUMENTS_FILE = "chunks.jsonl"
CONFIG_FILE = "config.json"
STATS_FILE = "build_stats.json"

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_TOP_K = 15
DEFAULT_BATCH_SIZE = 64
DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_CHUNK_OVERLAP_TOKENS = 64

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")


@dataclass(frozen=True)
class RetrieverIndex:
    embeddings: np.ndarray
    documents: list[dict[str, Any]]
    config: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_openai_client(client: Any | None = None) -> Any:
    if client is not None:
        return client

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The openai package is required. Install CRAG requirements or run "
            "`pip install openai`."
        ) from exc

    return OpenAI()


def _token_windows(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> Iterable[tuple[str, int, int]]:
    if overlap_tokens >= max_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    matches = list(TOKEN_PATTERN.finditer(text))
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


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def _ensure_not_legacy_vector_db(index_dir: Path) -> None:
    if _same_path(index_dir, LEGACY_VECTOR_DB_DIR):
        raise ValueError(
            f"Refusing to write OpenAI embeddings into the existing sparse vector DB: "
            f"{LEGACY_VECTOR_DB_DIR}. Use {DEFAULT_OPENAI_VECTOR_DB_DIR} instead."
        )


def _embedding_request_kwargs(
    texts: Sequence[str],
    model: str,
    dimensions: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": list(texts),
        "encoding_format": "float",
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    return kwargs


def _embed_texts(
    texts: Sequence[str],
    *,
    client: Any | None,
    model: str,
    dimensions: int | None,
) -> list[np.ndarray]:
    if not texts:
        return []

    openai_client = _load_openai_client(client)
    response = openai_client.embeddings.create(
        **_embedding_request_kwargs(texts, model, dimensions)
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return [
        np.asarray(item.embedding, dtype=np.float32)
        for item in ordered
    ]


def build_index(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    *,
    client: Any | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Embed the Fin-RATE corpus and persist a cosine-search index.

    The index stores retrieved document chunks only. No answer generation is
    performed here or in the query path.
    """
    seed_everything(seed)
    corpus_path = Path(corpus_path)
    index_dir = Path(index_dir)
    embedding_model = model or os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when provided")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    _ensure_not_legacy_vector_db(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now_iso()
    start_time = time.perf_counter()
    documents_seen = 0
    chunks_indexed = 0
    pending_texts: list[str] = []
    documents: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []

    def flush_pending() -> None:
        nonlocal pending_texts
        vectors.extend(
            _embed_texts(
                pending_texts,
                client=client,
                model=embedding_model,
                dimensions=dimensions,
            )
        )
        pending_texts = []

    stop_after_limit = False
    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            documents_seen += 1
            doc_id = record.get("_id", f"line_{line_number}")
            title = record.get("title", "")
            text = record.get("text", "")

            for chunk_index, (chunk_text, token_start, token_end) in enumerate(
                _token_windows(text, chunk_size_tokens, chunk_overlap_tokens)
            ):
                indexed_text = f"{title}\n\n{chunk_text}" if title else chunk_text
                documents.append(
                    {
                        "chunk_id": chunks_indexed,
                        "doc_id": doc_id,
                        "title": title,
                        "chunk_index": chunk_index,
                        "token_start": token_start,
                        "token_end": token_end,
                        "text": chunk_text,
                    }
                )
                pending_texts.append(indexed_text)
                chunks_indexed += 1

                if len(pending_texts) >= batch_size:
                    flush_pending()
                if limit is not None and chunks_indexed >= limit:
                    stop_after_limit = True
                    break

            if stop_after_limit:
                break

    if pending_texts:
        flush_pending()

    if len(vectors) != len(documents):
        raise RuntimeError(
            f"Embedding count mismatch: received {len(vectors)} vectors for "
            f"{len(documents)} documents."
        )

    embedding_matrix = (
        _normalize_matrix(np.vstack(vectors))
        if vectors
        else np.empty((0, 0), dtype=np.float32)
    )
    np.save(index_dir / EMBEDDINGS_FILE, embedding_matrix)
    _write_jsonl(index_dir / DOCUMENTS_FILE, documents)

    build_seconds = time.perf_counter() - start_time
    config = {
        "corpus_path": str(corpus_path),
        "database_type": "openai_dense_embedding_vector_db",
        "embedding_model": embedding_model,
        "dimensions": dimensions,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "default_top_k": DEFAULT_TOP_K,
        "documents_file": DOCUMENTS_FILE,
        "embeddings_file": EMBEDDINGS_FILE,
        "distance": "cosine_similarity",
        "seed": seed,
    }
    stats = {
        **config,
        "database_dir": str(index_dir),
        "index_dir": str(index_dir),
        "documents_seen": documents_seen,
        "chunks_indexed": chunks_indexed,
        "embedding_shape": list(embedding_matrix.shape),
        "build_started_at_utc": started_at,
        "build_finished_at_utc": _utc_now_iso(),
        "build_seconds": build_seconds,
    }

    _write_json(index_dir / CONFIG_FILE, config)
    _write_json(index_dir / STATS_FILE, stats)
    return stats


def load_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> RetrieverIndex:
    index_dir = Path(index_dir)
    embeddings_path = index_dir / EMBEDDINGS_FILE
    documents_path = index_dir / DOCUMENTS_FILE
    config_path = index_dir / CONFIG_FILE

    missing = [
        path
        for path in (embeddings_path, documents_path, config_path)
        if not path.exists()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"OpenAI embedding index is missing required file(s): {names}. "
            "Run the build command first."
        )

    embeddings = np.load(embeddings_path)
    documents = _read_jsonl(documents_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if embeddings.shape[0] != len(documents):
        raise ValueError(
            f"Index has {embeddings.shape[0]} vectors but {len(documents)} documents."
        )
    return RetrieverIndex(embeddings=embeddings, documents=documents, config=config)


def retrieve_documents(
    query: str,
    *,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    client: Any | None = None,
    top_k: int | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Return the top retrieved Fin-RATE document chunks for a query."""
    seed_everything(seed)
    if not query.strip():
        return []

    index = load_index(index_dir)
    if index.embeddings.size == 0:
        return []

    embedding_model = model or index.config.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    embedding_dimensions = (
        dimensions
        if dimensions is not None
        else index.config.get("dimensions")
    )
    requested_top_k = top_k or int(index.config.get("default_top_k", DEFAULT_TOP_K))
    if requested_top_k < 1:
        return []

    query_vector = _embed_texts(
        [query],
        client=client,
        model=embedding_model,
        dimensions=embedding_dimensions,
    )[0]
    query_vector = _normalize_vector(query_vector)
    if query_vector.shape[0] != index.embeddings.shape[1]:
        raise ValueError(
            f"Query embedding has {query_vector.shape[0]} dimensions, but the "
            f"vector DB stores {index.embeddings.shape[1]}. Rebuild the DB with "
            "the same model/dimensions used for querying."
        )
    scores = index.embeddings @ query_vector

    top_k_count = min(requested_top_k, len(index.documents))
    positions = np.arange(len(scores), dtype=np.int64)
    top_positions = positions[np.lexsort((positions, -scores))[:top_k_count]]

    results: list[dict[str, Any]] = []
    for position in top_positions:
        document = dict(index.documents[int(position)])
        document["score"] = float(scores[int(position)])
        results.append(document)
    return results


def retrieval_pipeline(
    query: str,
    *,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    return retrieve_documents(query, seed=seed)


def retrieve_relevant_chunks(
    query: str,
    *,
    seed: int = DEFAULT_SEED,
) -> list[str]:
    return [document["text"] for document in retrieve_documents(query, seed=seed)]


class OpenAIEmbeddingRetriever:
    """Small wrapper for building and querying a local OpenAI embeddings index."""

    def __init__(
        self,
        *,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        client: Any | None = None,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.client = client
        self.seed = seed

    def build(
        self,
        corpus_path: str | Path = DEFAULT_CORPUS_PATH,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.setdefault("seed", self.seed)
        return build_index(
            corpus_path=corpus_path,
            index_dir=self.index_dir,
            client=self.client,
            **kwargs,
        )

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[dict[str, Any]]:
        return retrieve_documents(
            query,
            index_dir=self.index_dir,
            client=self.client,
            top_k=top_k,
            seed=self.seed,
        )

    def retrieve_texts(self, query: str, *, top_k: int | None = None) -> list[str]:
        return [document["text"] for document in self.retrieve(query, top_k=top_k)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or query an OpenAI-embedding retriever for Fin-RATE."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Embed the corpus and save a vector DB.")
    build_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    build_parser.add_argument(
        "--db-dir",
        "--index-dir",
        dest="index_dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Where to write the OpenAI vector DB.",
    )
    build_parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    build_parser.add_argument("--dimensions", type=int, default=None)
    build_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    build_parser.add_argument("--chunk-size-tokens", type=int, default=DEFAULT_CHUNK_SIZE_TOKENS)
    build_parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_TOKENS,
    )
    build_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional chunk limit for smoke tests before embedding the full corpus.",
    )
    build_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    query_parser = subparsers.add_parser("query", help="Return retrieved documents as JSON.")
    query_parser.add_argument("query")
    query_parser.add_argument(
        "--db-dir",
        "--index-dir",
        dest="index_dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="OpenAI vector DB directory to query.",
    )
    query_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    query_parser.add_argument("--model", default=None)
    query_parser.add_argument("--dimensions", type=int, default=None)
    query_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    seed_everything(args.seed)
    if args.command == "build":
        stats = build_index(
            corpus_path=args.corpus,
            index_dir=args.index_dir,
            model=args.model,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
            chunk_size_tokens=args.chunk_size_tokens,
            chunk_overlap_tokens=args.chunk_overlap_tokens,
            limit=args.limit,
            seed=args.seed,
        )
        print(json.dumps(stats, indent=2))
    elif args.command == "query":
        results = retrieve_documents(
            args.query,
            index_dir=args.index_dir,
            top_k=args.top_k,
            model=args.model,
            dimensions=args.dimensions,
            seed=args.seed,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
