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
