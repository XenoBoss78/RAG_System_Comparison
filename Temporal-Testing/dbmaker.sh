#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
document_ids_file="$script_dir/ltqa_subset_doc_ids.json"

if [[ ! -f "$document_ids_file" ]]; then
  echo "Missing $document_ids_file. Run the LT-QA selection cell in TemporalMemoryRetrieval.ipynb first." >&2
  exit 1
fi

python "$script_dir/ChromaSetup.py" --build \
  --document-ids-file "$document_ids_file" \
  --db-dir "$script_dir/../Fin-RATE/chroma_db_ltqa_subset" \
  --collection fin_rate_ltqa_subset \
  --embedding-backend ollama \
  --embedding-model embeddinggemma:latest \
  --ollama-embedding-batch-size 32 \
  --batch-size 256

python "$script_dir/ChromaSetupMetaData.py" --build \
  --document-ids-file "$document_ids_file" \
  --db-dir "$script_dir/../Fin-RATE/chroma_metadata_db_ltqa_subset" \
  --collection fin_rate_ltqa_subset \
  --embedding-backend ollama \
  --embedding-model embeddinggemma:latest \
  --ollama-embedding-batch-size 32
