"""OpenAI-embedding retrieval utilities for the Fin-RATE corpus."""

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_INDEX_DIR",
    "DEFAULT_OPENAI_VECTOR_DB_DIR",
    "LEGACY_VECTOR_DB_DIR",
    "OpenAIEmbeddingRetriever",
    "build_index",
    "retrieval_pipeline",
    "retrieve_documents",
    "retrieve_relevant_chunks",
    "seed_everything",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if __package__:
        from . import openai_embeddings_retriever
    else:
        import openai_embeddings_retriever

    value = getattr(openai_embeddings_retriever, name)
    globals()[name] = value
    return value
