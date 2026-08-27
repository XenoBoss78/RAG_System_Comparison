[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$documentIdsFile = Join-Path $scriptDir "ltqa_subset_doc_ids.json"

if (-not (Test-Path -LiteralPath $documentIdsFile -PathType Leaf)) {
    throw "Missing $documentIdsFile. Run the LT-QA selection cell in TemporalMemoryRetrieval.ipynb first."
}

function Invoke-PythonBuild {
    param([string[]]$Arguments)

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Database build failed with exit code $LASTEXITCODE."
    }
}

Invoke-PythonBuild @(
    (Join-Path $scriptDir "ChromaSetup.py"),
    "--build",
    "--document-ids-file", $documentIdsFile,
    "--db-dir", (Join-Path $scriptDir "..\Fin-RATE\chroma_db_ltqa_subset"),
    "--collection", "fin_rate_ltqa_subset",
    "--embedding-backend", "ollama",
    "--embedding-model", "embeddinggemma:latest",
    "--ollama-embedding-batch-size", "32",
    "--batch-size", "256"
)

Invoke-PythonBuild @(
    (Join-Path $scriptDir "ChromaSetupMetaData.py"),
    "--build",
    "--document-ids-file", $documentIdsFile,
    "--db-dir", (Join-Path $scriptDir "..\Fin-RATE\chroma_metadata_db_ltqa_subset"),
    "--collection", "fin_rate_ltqa_subset",
    "--embedding-backend", "ollama",
    "--embedding-model", "embeddinggemma:latest",
    "--ollama-embedding-batch-size", "32"
)
