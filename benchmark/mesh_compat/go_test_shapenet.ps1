<#
.SYNOPSIS
  One-click ShapeNetPart mesh semantic test: export cache -> run ours -> live
  metrics + GT/ours renders (every connected component shown).

.DESCRIPTION
  Requires TWO datasets on disk:
    -PartRoot : ShapeNetPart "_normal" root  (<synset>/<model>.txt + train_test_split/)
    -CoreRoot : ShapeNetCore root            (<synset>/<model>/...obj, v1 or v2)

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File `
    E:\Proj\AgentSematic\benchmark\mesh_compat\go_test_shapenet.ps1 `
    -PartRoot "E:\data\shapenetcore_partanno_segmentation_benchmark_v0_normal" `
    -CoreRoot "E:\data\ShapeNetCore.v2" -N 5 -Categories "chair,table"
#>
param(
    [Parameter(Mandatory = $true)][string]$PartRoot,
    [Parameter(Mandatory = $true)][string]$CoreRoot,
    [int]$N = 5,
    [string]$Split = "test",
    [string]$Categories = "",
    [string]$ShapeIds = "",
    [string]$Provider = "qwen",
    [int]$MaxDepth = 4,
    [int]$ParallelVlm = 3,
    [switch]$ForceExport,
    [switch]$SkipEval
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$AgentRoot = "E:\Proj\AgentSematic"
$MraRoot   = "E:\Proj\MeshReductionAgent"
$PyMra     = "D:\scoop\apps\miniconda3\current\envs\MeshReduction\python.exe"
$PyAgent   = "D:\scoop\apps\miniconda3\current\envs\agentsematic\python.exe"

$McDir  = Join-Path $AgentRoot "benchmark\mesh_compat"
$ResDir = Join-Path $AgentRoot "benchmark\results"
$Cache  = Join-Path $ResDir "mesh_cache_shapenet"
$Work   = Join-Path $Cache "_ours_test"
$Out    = Join-Path $ResDir "preds_ours_shapenet.jsonl"
$Report = Join-Path $ResDir "compare_mesh_shapenet.json"
$Vocab  = Join-Path $MraRoot "configs\vocab\shapenetpart.json"

New-Item -ItemType Directory -Force -Path $ResDir | Out-Null

function Step($m) { Write-Host "`n========== $m ==========" -ForegroundColor Cyan }

if (-not (Test-Path $PartRoot)) { Write-Host "PartRoot not found: $PartRoot" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $CoreRoot)) { Write-Host "CoreRoot not found: $CoreRoot" -ForegroundColor Red; exit 1 }

# ---- Stage 1: export cache (if missing / too few) ----
$idx = Join-Path $Cache "index.json"
$needExport = $ForceExport -or -not (Test-Path $idx)
if (-not $needExport -and (Test-Path $idx)) {
    $cacheJson = Get-Content $idx -Raw | ConvertFrom-Json
    $cachedCount = @($cacheJson.PSObject.Properties).Count
    if ($cachedCount -lt $N) {
        Write-Host "cache has $cachedCount shape(s), need $N -> re-export" -ForegroundColor Yellow
        $needExport = $true
    } else {
        Write-Host "cache has $cachedCount shape(s) (>= $N), reusing" -ForegroundColor DarkGray
    }
}
if ($needExport) {
    Step "STAGE 1  export ShapeNetPart mesh+GT cache (agentsematic env)"
    $exportArgs = @(
        (Join-Path $McDir "export_cache_shapenet.py"),
        "--cache", $Cache,
        "--part-root", $PartRoot,
        "--core-root", $CoreRoot,
        "--vocab", $Vocab,
        "--split", $Split,
        "--n", "$N"
    )
    if ($Categories) { $exportArgs += @("--categories", $Categories) }
    if ($ShapeIds)   { $exportArgs += @("--shape-ids", $ShapeIds) }
    & $PyAgent @exportArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "export failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1 }
} else {
    Step "STAGE 1  reuse cache at $Cache (add -ForceExport to rebuild)"
}

# ---- Stage 2: ours + GT/ours renders + live metrics ----
Step "STAGE 2  AgentSematic ours + visual outputs (agentsematic env)"
$oursArgs = @(
    (Join-Path $McDir "run_ours.py"),
    "--cache", $Cache,
    "--out", $Out,
    "--work-dir", $Work,
    "--provider", $Provider,
    "--max-depth", "$MaxDepth",
    "--parallel-vlm", "$ParallelVlm",
    "--limit", "$N"
)
if ($ShapeIds) { $oursArgs += @("--shape-ids", $ShapeIds) }
& $PyAgent -u @oursArgs
if ($LASTEXITCODE -ne 0) { Write-Host "run_ours exit=$LASTEXITCODE" -ForegroundColor Yellow }

# ---- Stage 3: unified metrics (optional) ----
if (-not $SkipEval -and (Test-Path $Out)) {
    Step "STAGE 3  unified metrics (MeshReduction env)"
    & $PyMra (Join-Path $MraRoot "scripts\eval_unified.py") `
        --predictions $Out `
        --methods ours_agent `
        --vocab $Vocab `
        --out $Report `
        --empty-union-policy shapenet
}

Step "DONE"
Write-Host ""
Write-Host "JSONL      : $Out"
Write-Host "work dir   : $Work"
Write-Host "report     : $Report  (skip with -SkipEval)"
Write-Host ""
Write-Host "Per-shape renders under:" -ForegroundColor Green
Write-Host "  $Work\<sid>\mesh_all_components.png"
Write-Host "  $Work\<sid>\shape_gt_all_parts.png"
Write-Host "  $Work\<sid>\shape_ours_all_parts.png"
Write-Host "  $Work\<sid>\<part>\gt_highlight.png  /  ours_highlight.png"
Write-Host ""
Write-Host "Open output folder:" -ForegroundColor Green
Write-Host "  explorer `"$Work`""
