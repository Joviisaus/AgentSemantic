<#
.SYNOPSIS
  One-click mesh semantic test: export cache (if needed) -> run ours -> live metrics + GT/ours renders.

.DESCRIPTION
  Runs AgentSematic on cached 3DCoMPaT meshes and writes:
    - per-shape: mesh_all_components.png, shape_gt_all_parts.png, shape_ours_all_parts.png
    - per-part:  gt_highlight.png, ours_highlight.png + VLM semantic in the log
  All renders include every face-adjacency connected component on the full mesh.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File `
    E:\Proj\AgentSematic\benchmark\mesh_compat\go_test_ours.ps1

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File `
    E:\Proj\AgentSematic\benchmark\mesh_compat\go_test_ours.ps1 -ShapeIds "0a_034" -N 1

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File `
    E:\Proj\AgentSematic\benchmark\mesh_compat\go_test_ours.ps1 -N 5 -ForceExport
#>
param(
    [int]$N = 5,
    [string]$Split = "valid",
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
$Cache  = Join-Path $ResDir "mesh_cache"
$Work   = Join-Path $Cache "_ours_test"
$Out    = Join-Path $ResDir "preds_ours_visual_test.jsonl"
$Report = Join-Path $ResDir "compare_mesh_visual_test.json"
$Vocab  = Join-Path $MraRoot "configs\vocab\3dcompat_coarse.json"

New-Item -ItemType Directory -Force -Path $ResDir | Out-Null

function Step($m) { Write-Host "`n========== $m ==========" -ForegroundColor Cyan }

# ---- Stage 1: export cache (if missing or too few shapes) ----
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
    Step "STAGE 1  export mesh+GT cache (MeshReduction env)"
    $exportArgs = @(
        (Join-Path $McDir "export_cache.py"),
        "--cache", $Cache,
        "--split", $Split,
        "--n", "$N"
    )
    if ($ShapeIds) { $exportArgs += @("--shape-ids", $ShapeIds) }
    & $PyMra @exportArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "export failed ($LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
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
if ($LASTEXITCODE -ne 0) {
    Write-Host "run_ours exit=$LASTEXITCODE" -ForegroundColor Yellow
}

# ---- Stage 3: unified metrics table (optional) ----
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
Write-Host "  $Work\<shape_id>\mesh_all_components.png"
Write-Host "  $Work\<shape_id>\shape_gt_all_parts.png"
Write-Host "  $Work\<shape_id>\shape_ours_all_parts.png"
Write-Host "  $Work\<shape_id>\<part>\gt_highlight.png"
Write-Host "  $Work\<shape_id>\<part>\ours_highlight.png"
Write-Host ""
Write-Host "Open output folder:" -ForegroundColor Green
Write-Host "  explorer `"$Work`""
