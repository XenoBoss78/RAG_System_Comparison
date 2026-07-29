from __future__ import annotations

import chromadb
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS_PATH = ROOT_DIR / "Fin-RATE" / "corpus" / "corpus" / "corpus.jsonl"
DEFAULT_DB_DIR = ROOT_DIR / "Fin-RATE" / "chroma_db"
DEFAULT_COLLECTION_NAME = "fin_rate"
DEFAULT_EMBEDDING_BACKEND = "default"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BGE_MODEL = "BAAI/bge-m3"
DEFAULT_LLM_BACKEND = "ollama"
DEFAULT_LLM_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
BATCH_SIZE = 8
MAX_CONTEXT_CHARS = 14000

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_$%./'&-]*")

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
) -> Any:
    """
    Create a Chroma embedding function.

    Use backend="sentence-transformers" with device="cuda" to run embeddings
    on an NVIDIA GPU when CUDA-enabled PyTorch is installed.
    Use backend="bge" to load BGE through Sentence Transformers. If model_name
    is not supplied, BAAI/bge-m3 is used. Short BGE aliases include "small",
    "base", "large", and "m3".
    """
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

    raise ValueError(
        "Unsupported embedding backend. Use 'default', 'sentence-transformers', or 'bge'."
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
    reset_collection: bool = True,
) -> dict[str, Any]:
    """
    Build a fully local persistent Chroma database from the Fin-RATE corpus.jsonl.

    The corpus is expected to be JSONL with fields like _id, title, and text.
    Passing a custom local embedding_function lets you use Sentence Transformers,
    Ollama embeddings, or another local embedding backend.
    """
    chromadb = _require_chromadb()
    corpus_path = Path(corpus_path)
    db_dir = Path(db_dir)

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
    documents_seen = 0
    chunks_indexed = 0
    tokens_processed = 0
    batch: dict[str, list[Any]] = {"ids": [], "documents": [], "metadatas": []}

    with corpus_path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            documents_seen += 1

            doc_id = _record_id(record, line_number)
            title = _record_title(record)
            text = _record_text(record)
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

                batch["ids"].append(chunk_id)
                batch["documents"].append(document_text)
                batch["metadatas"].append(metadata)
                chunks_indexed += 1

                if len(batch["ids"]) >= batch_size:
                    _flush_batch(collection, batch)

    _flush_batch(collection, batch)

    stats = {
        "corpus_path": str(corpus_path),
        "database_dir": str(db_dir),
        "collection_name": collection_name,
        "documents_seen": documents_seen,
        "chunks_indexed": chunks_indexed,
        "collection_count": collection.count(),
        "chunk_size_tokens": chunk_size_tokens,
        "chunk_overlap_tokens": chunk_overlap_tokens,
        "embedding_function": type(embedder).__name__,
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
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant Chroma chunks for a query.

    Returns dictionaries with id, text, metadata, distance, and score. Chroma
    cosine distances are lower-is-better; score is 1 - distance for convenience.
    """
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

    query_kwargs: dict[str, Any] = {
        "query_texts": [query],
        "n_results": min(n_results, collection_count),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where
    if where_document is not None:
        query_kwargs["where_document"] = where_document

    results = collection.query(**query_kwargs)
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
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Make sure Ollama is running locally and "
            f"the model is available: ollama pull {model}"
        ) from exc

    return str(result.get("response", "")).strip()


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
) -> dict[str, Any]:
    """
    Generate an answer from retrieved chunks.

    Pass llm=<callable> to use any model client. If no callable is passed, the
    default backend calls a local Ollama model.
    """
    prompt = build_answer_prompt(
        question,
        chunks,
        max_context_chars=max_context_chars,
    )

    with open("example.txt", "w") as file:
        file.write(prompt)

    if llm is not None:
        answer = llm(prompt)
    elif llm_backend == "ollama":
        answer = _generate_with_ollama(
            prompt,
            model=llm_model,
            ollama_url=ollama_url,
            temperature=temperature,
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
) -> dict[str, Any]:
    """Retrieve relevant chunks and generate a grounded answer."""
    chunks = retrieve_relevant_chunks(
        question,
        db_dir=db_dir,
        collection_name=collection_name,
        n_results=n_results,
        embedding_function=embedding_function,
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
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=5)
    parser.add_argument(
        "--embedding-backend",
        choices=["default", "sentence-transformers", "bge"],
        default=DEFAULT_EMBEDDING_BACKEND,
        help="Embedding backend to use for build/query.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Sentence Transformers model name. With --embedding-backend bge, "
            "defaults to BAAI/bge-m3 and also accepts aliases: small, base, "
            "large, m3."
        ),
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
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument("--no-reset", action="store_true", help="Do not delete an existing collection.")
    args = parser.parse_args()

    embedding_function = create_embedding_function(
        backend=args.embedding_backend,
        model_name=args.embedding_model,
        device=args.device,
    )

    if args.build:
        stats = build_chroma_database(
            corpus_path=args.corpus,
            db_dir=args.db_dir,
            collection_name=args.collection,
            embedding_function=embedding_function,
            reset_collection=not args.no_reset,
        )
        print(json.dumps(stats, indent=2))

    if args.query:
        chunks = retrieve_relevant_chunks(
            args.query,
            db_dir=args.db_dir,
            collection_name=args.collection,
            n_results=args.top_k,
            embedding_function=embedding_function,
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
        )
        sources = [
            {
                "rank": index,
                "id": chunk.get("id"),
                "doc_id": (chunk.get("metadata") or {}).get("doc_id"),
                "title": (chunk.get("metadata") or {}).get("title"),
                "score": chunk.get("score"),
            }
            for index, chunk in enumerate(result["chunks"], start=1)
        ]
        datajson = {
            "question": result["question"],
            "answer": result["answer"],
            "sources": sources,}
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(datajson, f, ensure_ascii=False, indent=2)
        print(json.dumps(datajson, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
