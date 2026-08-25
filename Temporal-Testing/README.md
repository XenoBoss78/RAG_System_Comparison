# Temporal Fin-RATE RAG Experiments

This directory contains the Fin-RATE LT-QA experiments for Chroma, metadata
Chroma, Engram Base, and Temporal Engram. The sections below include detailed
implementation notes. The **current workflow** here is the recommended way to
set up a clean workspace, build a reproducible subset, benchmark every system,
and generate answer-quality samples.

## Current workflow

All commands in this section are PowerShell commands run from
`Inception\Temporal-Testing`. In a notebook, use the same arguments as separate
items in a `subprocess.run([...], check=True)` list; do not copy PowerShell
backticks into a notebook cell.

### Workspace and Python setup

The expected layout is:

```text
Inception/
├── Fin-RATE/
│   ├── corpus/corpus/corpus.jsonl
│   └── qa/LT-QA.json
├── engram/
└── Temporal-Testing/
```

Use Python 3.12. A virtual environment is recommended but not required:

```powershell
Set-Location C:\path\to\Inception\Temporal-Testing
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the sibling `../engram` checkout in editable mode,
so keep `engram` beside `Temporal-Testing`.

The default configuration uses Ollama. Install/start Ollama, then pull the
models needed by the workflows you will run:

```powershell
ollama pull embeddinggemma:latest
ollama pull qwen3:4b
ollama pull qwen3:4b-instruct
ollama list
```

- `embeddinggemma:latest` is the default embedding model for Chroma and
  Temporal Engram.
- `qwen3:4b` is used by optional Temporal ingestion, query-time company
  resolution, and graph planning.
- `qwen3:4b-instruct` is recommended for grounded answer-generation tests.

Ollama normally listens at `http://127.0.0.1:11434`. If `ollama list` cannot
connect, start the Ollama application or run `ollama serve` in another terminal.

All scripts default to `--seed 42`. Change the seed only when deliberately
creating a different question/document sample. LLM text can still vary across
model versions and hardware.

### Systems and the stores they share

The query-time variants below reuse a physical database; they do not require a
separate ingestion build.

| Benchmark system | Physical database required | Query-time behavior |
|---|---|---|
| `chroma` | Plain Chroma | Raw vector chunk retrieval. |
| `chroma_metadata` | Metadata Chroma | Company/year filtering, vector retrieval, and title-token boost. |
| `chroma_metadata_multilayer` | Metadata Chroma | Multilayer temporal filtering. |
| `chroma_metadata_verified` | Metadata Chroma | Company/year coverage selection, targeted recovery, then LLM evidence audit. |
| `chroma_metadata_yearly` | Metadata Chroma | Retrieves candidates per requested year and protects year coverage. |
| `chroma_metadata_nsq` | Metadata Chroma | Retrieves the parent and saved subqueries, then merges branches with RRF. |
| `engram_base` | Engram Base | Regular Engram retrieval (`engram` is an alias). |
| `engram_temporal` | Temporal Engram | Company/time filtering and raw evidence; broad questions can navigate through filing summaries. |
| `engram_temporal_nsq` | Temporal Engram | Temporal retrieval per subquery, then reciprocal-rank fusion. |
| `engram_temporal_graph` | Temporal Engram | Metadata-filtered vector retrieval guided by graph connections. |

Consequently, a complete build creates only four physical stores: `chroma`,
`chroma_metadata`, `engram_base`, and `engram_temporal`.

### Build a seeded LT-QA subset and all required stores

`build_ltqa_subset_databases.py` is the recommended builder. It samples the
requested number of LT-QA records, takes the union of their golden `doc_ids`,
optionally adds random non-gold documents, saves the question/document lists,
and builds only the needed physical stores.

Inspect the deterministic selection and plan first:

```powershell
python .\build_ltqa_subset_databases.py `
  --questions 100 `
  --extra-documents 0 `
  --systems all `
  --subset-name ltqa-q100-extra0-seed42 `
  --dry-run
```

Build the four physical stores with deterministic Temporal metadata and filing
digests:

```powershell
python .\build_ltqa_subset_databases.py `
  --questions 100 `
  --extra-documents 0 `
  --systems all `
  --subset-name ltqa-q100-extra0-seed42 `
  --embedding-backend ollama `
  --embedding-model embeddinggemma:latest `
  --engram-embed-model embeddinggemma:latest
```

For the full Temporal Engram build—LLM document metadata, per-chunk summaries,
and filing-level summaries—add these options. This can be much slower because
it invokes an LLM during ingestion.

```powershell
python .\build_ltqa_subset_databases.py `
  --questions 100 `
  --extra-documents 0 `
  --systems all `
  --subset-name ltqa-q100-extra0-seed42 `
  --embedding-backend ollama `
  --embedding-model embeddinggemma:latest `
  --engram-embed-model embeddinggemma:latest `
  --temporal-detail full `
  --temporal-llm-model qwen3:4b
```

The builder writes a question file, document-ID file, selection JSON, and
manifest under `Temporal-Testing/subsets`. It writes the stores and a
`build_manifest.json` under:

```text
Fin-RATE/ltqa_subset_stores/ltqa-q100-extra0-seed42/
├── build_manifest.json
├── chroma/
├── chroma_metadata/
├── engram_base/
└── engram_temporal/
```

The manifest is the source of truth for paths, collection name, namespaces, and
the invoked build commands. For the example above the collection is
`fin_rate_ltqa_q100_extra0_seed42`; the namespaces are
`fin-rate-ltqa-q100-extra0-seed42-base` and
`fin-rate-ltqa-q100-extra0-seed42-temporal`.

To create a harder retrieval set, add distractor documents. For example, build
only Metadata Chroma and Temporal Engram for 150 LT-QA questions plus 75
non-gold documents:

```powershell
python .\build_ltqa_subset_databases.py `
  --questions 150 `
  --extra-documents 75 `
  --systems chroma_metadata,engram_temporal `
  --subset-name ltqa-q150-extra75-seed42 `
  --temporal-detail full
```

The builder protects existing subset names. Use a new `--subset-name` for a new
experiment. Use `--overwrite` only when you deliberately want to reset the
selected physical stores.

Temporal Engram writes `ingestion_timings.jsonl` in its store. It records every
document's ingestion time, individual ingestion-stage timings, checkpoints,
total build time, and average document-stage times. If an ingestion run stops
late, rerun the corresponding `Engram_Temporal.py --build` command **without**
`--reset` after Ollama is healthy: already ingested document IDs are skipped and
the remaining summary stages can finish.

### Benchmark the non-NSQ systems

`benchmark_retrievers.py` writes a `*_summary.json` and per-query
`*_details.jsonl` under `Fin-RATE/retrieval_benchmarks`. Its metrics are
document-level retrieval metrics: `recall@k`, `hit@k`, `precision@k`, `MRR`, and
latency. They do not directly measure answer quality.

For the example subset, set these PowerShell variables once:

```powershell
$subset = "ltqa-q100-extra0-seed42"
$qaFile = ".\subsets\$($subset)_questions.json"
$storeRoot = "..\Fin-RATE\ltqa_subset_stores\$subset"
$collection = "fin_rate_ltqa_q100_extra0_seed42"
$baseNamespace = "fin-rate-ltqa-q100-extra0-seed42-base"
$temporalNamespace = "fin-rate-ltqa-q100-extra0-seed42-temporal"
```

Then benchmark the standard, metadata, temporal, and graph variants:

```powershell
python .\benchmark_retrievers.py `
  --systems chroma,chroma_metadata,chroma_metadata_multilayer,chroma_metadata_verified,chroma_metadata_yearly,engram_base,engram_temporal,engram_temporal_graph `
  --qa-file $qaFile `
  --chroma-db-dir "$storeRoot\chroma" `
  --chroma-metadata-db-dir "$storeRoot\chroma_metadata" `
  --chroma-collection $collection `
  --engram-base-store-dir "$storeRoot\engram_base" `
  --engram-base-namespace $baseNamespace `
  --engram-temporal-store-dir "$storeRoot\engram_temporal" `
  --engram-temporal-namespace $temporalNamespace `
  --embedding-backend ollama `
  --embedding-model embeddinggemma:latest `
  --engram-ollama-embed-model embeddinggemma:latest `
  --metadata-filter-mode heuristic `
  --year-branch-candidates 20 `
  --temporal-graph-llm-model qwen3:4b `
  --temporal-graph-max-hops 2 `
  --temporal-graph-seed-k 12 `
  --temporal-graph-candidate-k 40 `
  --n-results 15 `
  --top-ks 5,10,15 `
  --runs 1 `
  --warmup 1 `
  --output-prefix ltqa_q100_standard_systems
```

`--metadata-filter-mode heuristic` is local and deterministic. To test LLM
query decomposition for Metadata Chroma, use
`--metadata-filter-mode ollama --metadata-llm-model qwen3:4b`. The normal
non-strict behavior retries unfiltered retrieval when a metadata predicate has
no matches; `--strict-metadata-filter` disables that recovery.

### Benchmark NSQ systems

NSQ requires a QA file containing precomputed subqueries for the same question
IDs and golden `doc_ids`. Use the NSQ question file rather than the original
question file. The original question stays as an anchor branch, while every
subquery contributes candidates to reciprocal-rank fusion.

```powershell
python .\benchmark_retrievers.py `
  --systems chroma_metadata_nsq,engram_temporal_nsq `
  --qa-file .\ltqa_subset_questions_nsq.json `
  --chroma-metadata-db-dir "$storeRoot\chroma_metadata" `
  --chroma-collection $collection `
  --engram-temporal-store-dir "$storeRoot\engram_temporal" `
  --engram-temporal-namespace $temporalNamespace `
  --embedding-backend ollama `
  --embedding-model embeddinggemma:latest `
  --engram-ollama-embed-model embeddinggemma:latest `
  --metadata-filter-mode heuristic `
  --nsq-branch-candidates 15 `
  --nsq-rrf-k 60 `
  --nsq-temporal-company-model qwen3:4b-instruct `
  --n-results 15 `
  --top-ks 5,10,15 `
  --runs 1 `
  --warmup 1 `
  --output-prefix ltqa_q100_nsq
```

Evaluate NSQ separately from an original-question run. Its rewritten query text
can raise document recall without improving downstream answer quality, so use
the same question file, seed, and gold document IDs for any direct comparison.

### Generate answers from saved benchmark details

`answer_saved_benchmark_runs.py` never reruns retrieval. It reads already saved
`*_details.jsonl` rows, rebuilds the saved evidence from the corpus, and sends
only that evidence to an Ollama answer model. This is the preferred way to
compare answer quality after a benchmark has completed.

Replay a 20-question Yearly sample:

```powershell
python .\answer_saved_benchmark_runs.py `
  --details-file ..\Fin-RATE\retrieval_benchmarks\ltqa_subset_chroma_metadata_yearly_details.jsonl `
  --systems chroma_metadata_yearly `
  --sample-size 20 `
  --seed 42 `
  --answer-model qwen3:4b-instruct `
  --output-prefix ltqa_20_ans_Yearly
```

Replay a saved NSQ run:

```powershell
python .\answer_saved_benchmark_runs.py `
  --details-file ..\Fin-RATE\retrieval_benchmarks\ltqa_subset_nsq_details.jsonl `
  --systems chroma_metadata_nsq,engram_temporal_nsq `
  --sample-size 20 `
  --seed 42 `
  --answer-model qwen3:4b-instruct `
  --output-prefix ltqa_nsq_20_replayed_answers
```

To compare systems saved in different benchmark runs, repeat `--details-file`:

```powershell
python .\answer_saved_benchmark_runs.py `
  --details-file ..\Fin-RATE\retrieval_benchmarks\ltqa_subset_chroma_metadata_yearly_details.jsonl `
  --details-file ..\Fin-RATE\retrieval_benchmarks\ltqa_subset_chroma_metadata_verified_details.jsonl `
  --systems chroma_metadata_yearly,chroma_metadata_verified `
  --sample-size 20 `
  --seed 42 `
  --answer-model qwen3:4b-instruct `
  --output-prefix ltqa_20_yearly_verified_answers
```

The replay tool writes `<prefix>_answers.jsonl` and `<prefix>_summary.json` to
`Fin-RATE/retrieval_benchmarks`. Chroma rows are reconstructed from their saved
chunk IDs. Engram rows often lack an exact source chunk ID, so the default
fallback uses the saved source document and marks this in the output. Use
`--missing-chunk-policy skip` for a strict chunk-only comparison, or `--resume`
to continue an interrupted answer run.

### Direct Temporal Engram query

Use the paths and namespace in `build_manifest.json`:

```powershell
python .\Engram_Temporal.py --query "What did Valero report for 2021 and 2022?" `
  --store-dir "$storeRoot\engram_temporal" `
  --namespace $temporalNamespace `
  --ollama-embed-model embeddinggemma:latest `
  --ollama-model qwen3:4b
```

Temporal Engram records query-stage timings in `query_timings.jsonl` in its
store. For a complete list of options, run:

```powershell
python .\build_ltqa_subset_databases.py --help
python .\benchmark_retrievers.py --help
python .\answer_saved_benchmark_runs.py --help
python .\Engram_Temporal.py --help
```

### Common problems

- **Could not reach Ollama**: start Ollama and confirm the named model exists
  with `ollama list`.
- **Embedding dimension mismatch**: reopen a store with the same embedding
  model used to build it. A model with another vector dimension requires a new
  store.
- **A long Temporal build stops during summaries**: fix Ollama, then rerun the
  matching direct Temporal build command without `--reset` so checkpointed raw
  ingestion is retained.
- **Unexpected answer quality despite strong recall**: inspect both the
  benchmark `*_details.jsonl` and replayed `*_answers.jsonl`. A metric can count
  a golden filing as retrieved even when the answer model receives the wrong
  chunk from that filing, or when a fixed context budget truncates it.

---

## Legacy implementation notes

The detailed sections below remain useful for direct builds and lower-level
Temporal Engram options. Prefer the current workflow above for new experiments.

## Setup

Use Python 3.12 and install the listed dependencies from this directory:

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
change, which is why the dependency ranges are recorded above. Runtime
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
  --ollama-model qwen3:4b `
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
  --ollama-model qwen3:4b `
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
