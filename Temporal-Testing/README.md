# TemporalLedgerMemory
Faadil Shaikh-Internship work

## Setup

Use Python 3.12 and install the pinned dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

The requirements expect the cloned `engram` repository to remain beside this
directory at `../engram`. The reproducibility checks used Engram commit
`ed0e53a8b1626158282afe3d0ec252d52fe17749` and Fin-RATE commit
`284b1b4f6cd1e58574f1f1d17c29b35c6aa72cce`.

## Reproducibility

All command-line workflows default to seed `42`. Pass `--seed <integer>` to
repeat a different seeded run. Python callers can pass the same value with the
`seed=` keyword. The notebook defines `SEED` once and uses it for sampling and
retrieval.

Local random-number generation and Ollama sampling are seeded. Exact results
can still vary when model versions, hardware, remote APIs, or the source data
change, which is why the direct Python dependencies are pinned above. Runtime
timestamps, elapsed-time statistics, and UUIDs generated internally by Engram
are expected to differ even when the semantic results are the same.

## Temporal Engram ingestion

`Engram_Temporal.py` stores every chunk as a dated episode and persists a
temporal document profile alongside the raw content and embeddings. The profile
contains the source filing date, the actual ingestion timestamp, SEC document
type, ticker/company, explicitly stated reporting period(s), and retrieval
keywords. It also saves a compact summary and its embedding, so both raw and
summary retrieval remain available after reopening the store.

The default extractor is deterministic and never invents a reporting period.
For company-name resolution, less regular period wording, and more useful
keywords, pass a local Ollama model with `--llm-document-metadata`:

```powershell
python Engram_Temporal.py --build --limit 200 --reset `
  --ollama-model llama3.1:8b `
  --ollama-embed-model nomic-embed-text `
  --llm-document-metadata `
  --llm-summaries
```

The structured metadata is saved in every episode's `metadata` field, the raw
and summary vectors are saved in `embedding` and `summary_embedding`, and graph
edges connect documents to their company, type, ingestion time, reporting
periods, and keywords. `--metadata-llm-text-chars` controls how much source
text is sent to Ollama for each metadata extraction (default: 6,000 characters).

### Filing-level summaries

Each build now also writes one `artifact_type: "filing_summary"` episode for
every filing. It contains the filing's company/form/date/period metadata, a
summary embedding, source document IDs, and the exact raw chunk IDs it covers.
The graph connects this stable summary node back to its filing. By default the
summary is a deterministic evidence digest, so it adds negligible build time.

For richer, cross-section summaries, enable a single LLM call per filing:

```powershell
python Engram_Temporal.py --build --reset `
  --ollama-model qwen3:4b `
  --llm-filing-summaries `
  --no-summaries
```

`--no-summaries` above skips the much more expensive one-LLM-call-per-chunk
summaries; filing-level summaries still provide a compact cross-filing retrieval
layer. Use `--no-filing-summaries` only when that layer is not wanted.

Every build also appends per-document ingestion telemetry to
`ingestion_timings.jsonl` in the store. Each `record_type: "document"` line
records durations for parsing, metadata extraction, chunking, episode
ingestion/embedding, episode metadata persistence, graph metadata creation, and
the total for that source document. The final `record_type: "build_summary"`
line records total build time, batch-only work (relationships, fact extraction,
summary indexing, and saving), and the average duration of every document step.
Use `--ingestion-timings-file <path>` to put this analysis file elsewhere.

The store is checkpointed after raw chunk/graph ingestion and again after
summary indexing. If a late stage fails, rerun the same build **without**
`--reset`: existing document IDs are skipped, incomplete chunk summaries are
finished, and filing-level summaries are then created. The timing JSONL records
each successful checkpoint as `record_type: "checkpoint"`; use `--reset` only
when deliberately starting the target store over.

### Build an LT-QA document subset

The LT-QA selection cell in `TemporalMemoryRetrieval.ipynb` saves two files for
every seeded sample: `ltqa_subset_doc_ids.json`, containing the connected
corpus-document IDs, and `ltqa_subset_questions.json`, containing the complete
selected LT-QA records. The latter preserves each `q_id`, question, reference
answer, key points, and gold `doc_ids`, so it can be used later for a
reproducible evaluation of the same subset.

Then build a new, separate store from exactly those IDs. The command below uses
LLM summaries for every raw chunk, LLM-enriched document metadata, and LLM
filing-level summaries. It leaves the full database untouched:

```powershell
python Engram_Temporal.py --build --reset `
  --document-ids-file .\ltqa_subset_doc_ids.json `
  --data-dir ..\Fin-RATE\engram_data_ltqa_subset `
  --namespace fin-rate-ltqa-subset `
  --ollama-model qwen3:4b `
  --ollama-embed-model embeddinggemma:latest `
  --llm-document-metadata `
  --llm-summaries `
  --llm-filing-summaries
```

`--document-ids-file` accepts the JSON list written by the notebook, a JSON
object with a `document_ids` list, or a plain text file with one ID per line.
The build result and final ingestion-timing record report which requested IDs
were found and which, if any, were missing.

### Build matching Chroma subset databases

Use the same `ltqa_subset_doc_ids.json` file to build comparison databases.
Give each database its own directory and collection name so it cannot replace a
full-corpus collection. The plain Chroma baseline stores raw embedded chunks:

```powershell
python ChromaSetup.py --build `
  --document-ids-file .\ltqa_subset_doc_ids.json `
  --db-dir ..\Fin-RATE\chroma_db_ltqa_subset `
  --collection fin_rate_ltqa_subset `
  --embedding-backend ollama `
  --embedding-model embeddinggemma:latest `
  --ollama-embedding-batch-size 32 `
  --batch-size 256
```

The metadata-aware Chroma variant can use the local `embeddinggemma` model and
persists its title/company/year metadata alongside the same raw chunks:

```powershell
python ChromaSetupMetaData.py --build `
  --document-ids-file .\ltqa_subset_doc_ids.json `
  --db-dir ..\Fin-RATE\chroma_metadata_db_ltqa_subset `
  --collection fin_rate_ltqa_subset `
  --embedding-backend ollama `
  --embedding-model embeddinggemma:latest `
  --ollama-embedding-batch-size 32
```

Once the notebook has written `ltqa_subset_doc_ids.json`, Git Bash or WSL users
can run both commands sequentially with `bash ./dbmaker.sh`.
From PowerShell, use `./dbmaker.ps1`; from Command Prompt, use
`powershell -NoProfile -ExecutionPolicy Bypass -File .\dbmaker.ps1`.

Both builders report requested, matched, and missing IDs in their
`build_stats.json` file. Chroma ingestion does not generate LLM summaries; it
only chunks and embeds the selected raw documents. The LLM is optional later,
when generating an answer or enriching a query filter.

Evaluate the same selected questions against both subset databases with:

```powershell
python benchmark_retrievers.py `
  --systems chroma,chroma_metadata `
  --qa-file .\ltqa_subset_questions.json `
  --chroma-db-dir ..\Fin-RATE\chroma_db_ltqa_subset `
  --chroma-metadata-db-dir ..\Fin-RATE\chroma_metadata_db_ltqa_subset `
  --chroma-collection fin_rate_ltqa_subset `
  --embedding-backend ollama `
  --embedding-model embeddinggemma:latest `
  --warmup 0
```

## Metadata-filtered RAG queries

`Engram_Temporal.py --query` first extracts the company and every explicit
four-digit year from the question. It uses the stored company, filing-date, and
reporting-period metadata to narrow candidates before retrieval. A specific
question searches raw chunks directly. A broad question (for example a
comparison, change, trend, or multi-year question) additionally searches the
filtered filing summaries, then expands their linked source chunks and ranks
those as exact evidence. A year matches either the filing date or any reporting
period stored for the document, so a 2022 filing that reports 2021 results
remains eligible for a 2021 question.

Relative phrases are resolved during query intake. The parser supports `past`,
`last`, or `previous` *N* years, quarters, or months, plus `this`, `current`,
and `latest` year/quarter/month. It anchors the window to the latest explicit
reporting-period endpoint available for the selected company in the store
(falling back to a filing date, then UTC today only if the store has no date).
For example, if Valero's latest stored reporting period ends in 2024, “within
the past 3 years” becomes explicit years `2022`, `2023`, and `2024`, with a
`2022-01-01` to `2024-12-31` time window. The resolved expression, reference,
years, and dates are returned in the query's `filters` payload.

```powershell
python Engram_Temporal.py --query "What financial results did Valero report for 2021?" `
  --store-dir ..\Fin-RATE\engram_data\fin-rate--444fdb2b328ad636 `
  --ollama-model llama3.1:8b `
  --llm-query-filters
```

The answer output shows the extracted filters, filing summaries used for
navigation, and the retrieved raw source documents. Use
`--no-filing-summary-retrieval` to force raw-only retrieval, or adjust
`--filing-summary-top-k` and `--evidence-per-filing-summary` for broader/narrower
coverage.
Each query also appends a JSON record to `query_timings.jsonl` in the store with
separate durations for store loading, query decomposition, metadata filtering,
query embedding, raw retrieval, filing-summary retrieval, evidence expansion,
context assembly, answer generation, and the total. Use
`--query-timings-file <path>` to write the analysis log elsewhere.

## Retrieval Benchmark

Use `benchmark_retrievers.py` to compare retrieval speed and document-level
accuracy on Fin-RATE LT-QA. It writes a summary JSON file and a per-query JSONL
trace under `Fin-RATE/retrieval_benchmarks`.

Smoke test Engram against an existing store:

```powershell
python benchmark_retrievers.py `
  --systems engram `
  --engram-store-dir ..\Fin-RATE\engram_data_deep_smoke\fin-rate--444fdb2b328ad636 `
  --limit 10 `
  --warmup 0
```

Compare Chroma, metadata-reranked Chroma, and Engram once dependencies and
indexes are present:

```powershell
python benchmark_retrievers.py `
  --systems chroma,chroma_metadata,engram `
  --limit 100 `
  --runs 3 `
  --top-ks 5,10,15
```

For full LT-QA, omit `--limit`. The metric is source-document retrieval:
`recall@k`, `hit@k`, `precision@k`, `MRR`, and latency are computed from the
ranked retrieved `doc_id`s against each QA pair's gold `doc_ids`.

When `chroma_metadata` is selected, the benchmark now calls its complete
metadata-aware retrieval path: it extracts company and year constraints,
filters the Chroma candidates, then applies the title-overlap rerank. The
default `--metadata-filter-mode heuristic` is local and deterministic. Use
`--metadata-filter-mode ollama --metadata-llm-model qwen3:4b` for LLM query
decomposition, or `--strict-metadata-filter` to report no result rather than
falling back to unfiltered retrieval when the constraints match nothing.
