<#
.SYNOPSIS
  One-click mesh-native comparison of AgentSematic ("ours") vs the referenced
  LLM_Spectral_Search method on 3DCoMPaT++, using the unified metrics
  (insmIoU / catmIoU / OA / mAcc / boundary / pointing-game / AP).

.DESCRIPTION
  Stage 1  export shared mesh+GT cache from the 3DCoMPaT ZIP   (MeshReduction env)
  Stage 2a run AgentSematic pipeline on the cache -> JSONL      (agentsematic env)
  Stage 2b run LLM_Spectral_Search on the SAME cache -> JSONL   (MeshReduction env, optional)
  Stage 3  eval_unified scores both JSONLs into one table       (MeshReduction env)

  Both methods read byte-identical mesh + GT + face-index, so the metrics are a
  strictly fair head-to-head.

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File `
    E:\Proj\AgentSematic\benchmark\mesh_compat\go_compare.ps1 -N 5
#>
param(
    [int]$N = 5,
    [string]$Split = "valid",
    [string]$Provider = "qwen",
    [int]$MaxDepth = 4,
    [int]$ParallelVlm = 3,
    [int]$NumModes = 5,
    [switch]$WithSpectral,
    [switch]$ForceExport,
    [string]$ShapeIds = ""
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$AgentRoot = "E:\Proj\AgentSematic"
$MraRoot   = "E:\Proj\MeshReductionAgent"
$PyMra     = "D:\scoop\apps\miniconda3\current\envs\MeshReduction\python.exe"
$PyAgent   = "D:\scoop\apps\miniconda3\current\envs\agentsematic\python.exe"

$McDir   = Join-Path $AgentRoot "benchmark\mesh_compat"
$ResDir  = Join-Path $AgentRoot "benchmark\results"
$Cache   = Join-Path $ResDir   "mesh_cache"
$OursJsonl     = Join-Path $ResDir "preds_ours_agent_3dcompat.jsonl"
$SpectralJsonl = Join-Path $ResDir "preds_spectral_3dcompat.jsonl"
$Vocab   = Join-Path $MraRoot "configs\vocab\3dcompat_coarse.json"
$Report  = Join-Path $ResDir "compare_mesh_3dcompat.json"

New-Item -ItemType Directory -Force -Path $ResDir | Out-Null

function Step($m) { Write-Host "`n========== $m ==========" -ForegroundColor Cyan }

# ---- Stage 1: export shared cache ----
$idx = Join-Path $Cache "index.json"
if ($ForceExport -or -not (Test-Path $idx)) {
    Step "STAGE 1  export mesh+GT cache (MeshReduction env)"
    $exportArgs = @((Join-Path $McDir "export_cache.py"),
        "--cache", $Cache, "--split", $Split, "--n", "$N")
    if ($ShapeIds) { $exportArgs += @("--shape-ids", $ShapeIds) }
    & $PyMra @exportArgs
    if ($LASTEXITCODE -ne 0) { Write-Host "export failed ($LASTEXITCODE)" -ForegroundColor Red; exit 1 }
} else {
    Step "STAGE 1  reuse existing cache at $Cache (use -ForceExport to rebuild)"
}

# ---- Stage 2a: AgentSematic "ours" ----
Step "STAGE 2a  AgentSematic pipeline -> ours JSONL (agentsematic env)"
& $PyAgent (Join-Path $McDir "run_ours.py") `
    --cache $Cache --out $OursJsonl `
    --provider $Provider --max-depth $MaxDepth --parallel-vlm $ParallelVlm
if ($LASTEXITCODE -ne 0) { Write-Host "ours run exit=$LASTEXITCODE (continuing to eval)" -ForegroundColor Yellow }

# ---- Stage 2b: LLM_Spectral_Search (optional) ----
$predArgs    = @($OursJsonl)
$methodNames = @("ours_agent")
if ($WithSpectral) {
    Step "STAGE 2b  LLM_Spectral_Search on SAME cache -> spectral JSONL (MeshReduction env)"
    & $PyMra (Join-Path $McDir "run_spectral.py") `
        --cache $Cache --out $SpectralJsonl `
        --provider $Provider --num-modes $NumModes
    if (Test-Path $SpectralJsonl) {
        $predArgs    += $SpectralJsonl
        $methodNames += "spectral"
    } else {
        Write-Host "spectral JSONL not produced; scoring ours only" -ForegroundColor Yellow
    }
}

# ---- Stage 3: unified evaluation (combined table) ----
Step "STAGE 3  unified metrics (MeshReduction env)"
& $PyMra (Join-Path $MraRoot "scripts\eval_unified.py") `
    --predictions @predArgs `
    --methods @methodNames `
    --vocab $Vocab `
    --out $Report `
    --empty-union-policy shapenet

Step "DONE"
Write-Host "ours JSONL : $OursJsonl"
if ($WithSpectral) { Write-Host "spectral   : $SpectralJsonl" }
Write-Host "report     : $Report"
