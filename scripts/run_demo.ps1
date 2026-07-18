param(
    [string]$OutputRoot = "demo_output",
    [int]$RandomTrials = 100
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$edgeCsv = Join-Path $repoRoot "examples\bridge20\edges.csv"
$leakyOutput = Join-Path $repoRoot "$OutputRoot\leaky"
$topologyOutput = Join-Path $repoRoot "$OutputRoot\topology_only"

Push-Location $repoRoot
try {
    python -m cascaderank.audit --dataset CSV --edge-csv $edgeCsv --mode leaky `
        --random-trials $RandomTrials --output-dir $leakyOutput
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -m cascaderank.verify --manifest "$leakyOutput\audit_manifest.json" `
        --edge-csv $edgeCsv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -m cascaderank.audit --dataset CSV --edge-csv $edgeCsv `
        --mode topology-only --random-trials $RandomTrials --output-dir $topologyOutput
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -m cascaderank.verify --manifest "$topologyOutput\audit_manifest.json" `
        --edge-csv $edgeCsv
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
