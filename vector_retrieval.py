from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = ROOT_DIR / "Fin-RATE" / "corpus" / "corpus" / "corpus.jsonl"
DEFAULT_DB_DIR = ROOT_DIR / "Fin-RATE" / "vector_db"

VECTOR_FILE = "vector_index.npz"
CHUNKS_FILE = "chunks.jsonl"
CONFIG_FILE = "config.json"
STATS_FILE = "build_stats.json"

TOKENIZER_NAME = "regex-tokenizer"
N_FEATURES = 2**18
CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
RETRIEVAL_TOP_K = 15

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")


@dataclass(frozen=True)
class VectorDatabase:
    # The database is stored as an inverted index: each feature points to the
    # chunk IDs and normalized weights where that feature appears.
    feature_ids: np.ndarray
    chunk_ids: np.ndarray
    weights: np.ndarray
    feature_offsets: dict[int, tuple[int, int]]
    chunks: list[dict[str, Any]]
    config: dict[str, Any]


_VECTOR_DB_CACHE: VectorDatabase | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_hash(value: str, n_features: int) -> int:
    # Hash each token into a fixed feature space so no vocabulary file is needed.
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_features


def _analyze(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _encode_vector(text: str, n_features: int) -> dict[int, float]:
    # Build a sparse, L2-normalized term-frequency vector for cosine scoring.
    token_counts = Counter(_analyze(text))
    if not token_counts:
        return {}

    feature_counts: Counter[int] = Counter()
    for token, count in token_counts.items():
        feature_counts[_stable_hash(token, n_features)] += count

    weighted = {
        feature_id: 1.0 + math.log(count)
        for feature_id, count in feature_counts.items()
    }
    norm = math.sqrt(sum(weight * weight for weight in weighted.values()))
    if not norm:
        return {}
    return {
        feature_id: weight / norm
        for feature_id, weight in weighted.items()
    }


def _fallback_token_windows(text: str, max_tokens: int, overlap_tokens: int) -> Iterable[tuple[str, int, int]]:
    # Split long filings into overlapping token windows so retrieval can return
    # focused chunks instead of entire SEC filing sections.
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


def _count_tokens(text: str) -> int:
    return len(_analyze(text))


def _iter_chunks(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> Iterable[tuple[str, int, int]]:
    yield from _fallback_token_windows(text, max_tokens, overlap_tokens)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_vector_database(
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    db_dir: str | Path = DEFAULT_DB_DIR,
    *,
    n_features: int = N_FEATURES,
    chunk_size_tokens: int = CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> dict[str, Any]:
    """Build and persist a sparse vector database for the Fin-RATE corpus."""
    corpus_path = Path(corpus_path)
    db_dir = Path(db_dir)

    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    db_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = db_dir / CHUNKS_FILE
    vectors_path = db_dir / VECTOR_FILE

    started_at = _utc_now_iso()
    start_time = time.perf_counter()

    feature_ids = array("I")
    chunk_ids = array("I")
    weights = array("f")
    documents_seen = 0
    chunks_indexed = 0
    tokens_processed = 0

    with corpus_path.open("r", encoding="utf-8") as corpus_file, chunks_path.open(
        "w", encoding="utf-8"
    ) as chunks_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            documents_seen += 1

            doc_id = record.get("_id", f"line_{line_number}")
            title = record.get("title", "")
            text = record.get("text", "")
            full_text = f"{title}\n\n{text}" if title else text
            tokens_processed += _count_tokens(full_text)

            for chunk_index, (chunk_text, token_start, token_end) in enumerate(
                _iter_chunks(text, chunk_size_tokens, chunk_overlap_tokens)
            ):
                # Index the title together with the chunk text so company,
                # filing, and section names can influence retrieval.
                indexed_text = f"{title}\n\n{chunk_text}" if title else chunk_text
                vector = _encode_vector(indexed_text, n_features)
                for feature_id, weight in vector.items():
                    feature_ids.append(feature_id)
                    chunk_ids.append(chunks_indexed)
                    weights.append(weight)

                chunk_record = {
                    "chunk_id": chunks_indexed,
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_index": chunk_index,
                    "token_start": token_start,
                    "token_end": token_end,
                    "text": chunk_text,
                }
                chunks_file.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                chunks_indexed += 1

    feature_array = np.frombuffer(feature_ids, dtype=np.uint32).copy()
    chunk_array = np.frombuffer(chunk_ids, dtype=np.uint32).copy()
    weight_array = np.frombuffer(weights, dtype=np.float32).copy()
    if feature_array.size:
        # Sorting by feature ID makes lookup cheap when reconstructing postings.
        order = np.argsort(feature_array, kind="stable")
        feature_array = feature_array[order]
        chunk_array = chunk_array[order]
        weight_array = weight_array[order]

    np.savez_compressed(
        vectors_path,
        feature_ids=feature_array,
        chunk_ids=chunk_array,
        weights=weight_array,
    )

    build_seconds = time.perf_counter() - start_time
    config = {
        "n_features": n_features,
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "tokenizer": TOKENIZER_NAME,
        "vectorizer": "stable hashed term-frequency vectors with l2 normalization",
        "index": "compressed sparse inverted vector index",
    }
    stats = {
        "corpus_path": str(corpus_path),
        "database_dir": str(db_dir),
        "documents_seen": documents_seen,
        "chunks_indexed": chunks_indexed,
        "tokens_processed": tokens_processed,
        "external_embedding_api_tokens": 0,
        "build_started_at_utc": started_at,
        "build_finished_at_utc": _utc_now_iso(),
        "build_seconds": build_seconds,
        "vector_shape": [chunks_indexed, n_features],
        "nonzero_vector_entries": int(feature_array.size),
        **config,
    }

    _write_json(db_dir / CONFIG_FILE, config)
    _write_json(db_dir / STATS_FILE, stats)
    return stats


def load_vector_database(db_dir: str | Path = DEFAULT_DB_DIR) -> VectorDatabase:
    db_dir = Path(db_dir)
    vectors_path = db_dir / VECTOR_FILE
    chunks_path = db_dir / CHUNKS_FILE
    config_path = db_dir / CONFIG_FILE

    missing = [path for path in (vectors_path, chunks_path, config_path) if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Vector database is missing required file(s): {names}")

    index = np.load(vectors_path)
    feature_ids = index["feature_ids"]
    chunk_ids = index["chunk_ids"]
    weights = index["weights"]

    feature_offsets: dict[int, tuple[int, int]] = {}
    if feature_ids.size:
        # Record where each feature's postings start and end in the flat arrays.
        change_points = np.flatnonzero(np.diff(feature_ids)) + 1
        starts = np.r_[0, change_points]
        ends = np.r_[change_points, feature_ids.size]
        for start, end in zip(starts, ends):
            feature_offsets[int(feature_ids[start])] = (int(start), int(end))

    chunks: list[dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as chunks_file:
        for line in chunks_file:
            if line.strip():
                chunks.append(json.loads(line))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return VectorDatabase(
        feature_ids=feature_ids,
        chunk_ids=chunk_ids,
        weights=weights,
        feature_offsets=feature_offsets,
        chunks=chunks,
        config=config,
    )


def _get_cached_vector_database() -> VectorDatabase:
    global _VECTOR_DB_CACHE
    if _VECTOR_DB_CACHE is None:
        _VECTOR_DB_CACHE = load_vector_database()
    return _VECTOR_DB_CACHE


def retrieval_pipeline(query: str) -> list[str]:
    """Return only the most relevant chunk texts for a query."""
    database = _get_cached_vector_database()
    query_vector = _encode_vector(query, int(database.config["n_features"]))
    if not query_vector or not database.chunks:
        return []

    scores_array = np.zeros(len(database.chunks), dtype=np.float32)
    for feature_id, query_weight in query_vector.items():
        # Accumulate dot products only for chunks sharing query features.
        offsets = database.feature_offsets.get(feature_id)
        if offsets is None:
            continue
        start, end = offsets
        np.add.at(
            scores_array,
            database.chunk_ids[start:end],
            database.weights[start:end] * query_weight,
        )

    positive = np.flatnonzero(scores_array > 0)
    if positive.size == 0:
        return []

    top_k = min(RETRIEVAL_TOP_K, positive.size)
    # Select the best matching chunks and return only their text, as requested.
    top_positions = positive[np.argpartition(scores_array[positive], -top_k)[-top_k:]]
    top_positions = top_positions[np.argsort(scores_array[top_positions])[::-1]]
    #TESTING FOR DOC ID, CHANGE TO CHUNK TEXT WHEN ANSWERING
    #return [database.chunks[int(index)]["text"] for index in top_positions]
    #print("THIGNS")
    return [database.chunks[int(index)]["doc_id"] for index in top_positions]


def retrieve_relevant_chunks(query: str) -> list[str]:
    return retrieval_pipeline(query)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the Fin-RATE vector database.")
    parser.add_argument("--build", action="store_true", help="Build the vector database.")
    parser.add_argument("--query", type=str, help="Query an already-built vector database.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    args = parser.parse_args()

    if args.build:
        stats = build_vector_database(corpus_path=args.corpus, db_dir=args.db_dir)
        print(json.dumps(stats, indent=2))
    if args.query:
        global _VECTOR_DB_CACHE
        _VECTOR_DB_CACHE = load_vector_database(args.db_dir)
        for chunk in retrieval_pipeline(args.query):
            print(chunk)
            print("\n---\n")


if __name__ == "__main__":
    _main()
