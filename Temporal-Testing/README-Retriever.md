# OpenAI Embedding Retriever

This folder contains a basic retrieval-only RAG module for the Fin-RATE corpus.
It embeds `Fin-RATE/corpus/corpus/corpus.jsonl` with OpenAI embeddings, stores a
local cosine-search vector DB, and returns the matching document chunks. It does
not generate answers from those documents.

The existing sparse vector DB at `Fin-RATE/vector_db` is left untouched. The
OpenAI embedding DB is written separately to `Fin-RATE/openai_vector_db`.

## Setup

Set an OpenAI API key before building or querying:

```powershell
$env:OPENAI_API_KEY = "your_api_key"
```

## Build The Index

From the `Temporal-Testing` directory:

```powershell
python openai_embeddings_retriever.py build
```

For a quick smoke test before embedding the full corpus:

```powershell
python openai_embeddings_retriever.py build --limit 100
```

By default the index is written to `Fin-RATE/openai_vector_db`. You can choose a
different location with `--db-dir`, but the command refuses to write into the
existing `Fin-RATE/vector_db` directory.

## Query

```powershell
python openai_embeddings_retriever.py query "What did Valero report for Q4 2021?"
```

The query command prints JSON document chunks with fields such as `doc_id`,
`title`, `chunk_index`, `text`, and `score`. It returns retrieved documents only,
not model-written answers.

## Python Usage

```python
from openai_embeddings_retriever import retrieve_documents, retrieve_relevant_chunks

documents = retrieve_documents("What did Valero report for Q4 2021?", top_k=5)
chunk_texts = retrieve_relevant_chunks("What did Valero report for Q4 2021?")
```
