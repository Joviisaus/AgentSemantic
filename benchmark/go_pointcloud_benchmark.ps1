<#
.SYNOPSIS
  One-click point-cloud benchmark: export GT -> run ours_pointcloud -> mIoU -> HTML report.

.DESCRIPTION
  Defaults to the CoMPaT-200 / CoSMo3D comparison path. It keeps ShapeNet-Part as
  an explicit supplementary option via -Dataset shapenetpart.

  Main outputs:
    - benchmark\data\gt\*.npz and benchmark\data\subsets\*.txt
    - benchmark\results\ours_pointcloud\*.npy
    - benchmark\results\summary_<subset>.json / .md
    - benchmark\results\rendered_report\report.html

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File `
    E:\Proj\AgentSematic\benchmark\go_pointcloud_benchmark.ps1

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File `
    E:\Proj\AgentSematic\benchmark\go_pointcloud_benchmark.ps1 -Subset subset -Limit 10 -Provider qwen

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File `
    E:\Proj\AgentSematic\benchmark\go_pointcloud_benchmark.ps1 -Dataset shapenetpart -ShapeNetPartRoot E:\data\shapenetpart
#>
param(
    [ValidateSet("compat200", "shapenetpart")]
    [string]$Dataset = "compat200",
    [string]$Subset = "tiny",
    [int]$Limit = 0,

    [string]$Provider = "qwen",
    [string]$Model = "",
    [string]$Config = "",
    [string]$EnvFile = "",

    [int]$MaxDepth = 6,
    [int]$MaxSeeds = 4,
    [int]$MinBlockSize = 20,
    [int]$ParallelVlm = 1,

    [switch]$NoNeo,
    [string]$NeoCkpt = "",
    [string]$NeoPython = "",
    [int]$NeoK = 0,
    [string]$NeoDevice = "cpu",
    [int]$NeoDensifyTarget = 20000,

    [string[]]$EvalMethods = @(),
    [string]$AgentEnv = "agentsematic",
    [string]$ExportEnv = "cosmo3d",
    [string]$ShapeNetPartRoot = "E:\data\shapenetpart",
    [string]$ShapeNetPartVocab = "",

    [switch]$Overwrite,
    [switch]$ForceExport,
    [switch]$SkipExport,
    [switch]$SkipRun,
    [switch]$SkipEval,
    [switch]$SkipReport,
    [switch]$NoViewSelect,
    [switch]$OpenReport
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$BenchRoot = $PSScriptRoot
$AgentRoot = Split-Path $BenchRoot -Parent
Set-Location $AgentRoot

function Step([string]$Message) {
    Write-Host ""
    Write-Host "========== $Message ==========" -ForegroundColor Cyan
}

function RunCmd([string]$Title, [string[]]$ArgsList) {
    Step $Title
    Write-Host ("conda " + ($ArgsList -join " ")) -ForegroundColor DarkGray
    & conda @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "$Title failed with exit code $LASTEXITCODE"
    }
}

function Add-IfValue([System.Collections.ArrayList]$List, [string]$Name, [string]$Value) {
    if ($Value) {
        [void]$List.Add($Name)
        [void]$List.Add($Value)
    }
}

$isShapeNetPart = $Dataset -eq "shapenetpart"
if ($isShapeNetPart) {
    $DataDir = Join-Path $BenchRoot "data_shapenetpart"
    $ResultsDir = Join-Path $BenchRoot "results_shapenetpart"
    $ExportModule = "benchmark.export_shapenetpart_gt"
    $ExportCondaEnv = $ExportEnv
    if ($EvalMethods.Count -eq 0) { $EvalMethods = @("ours_pointcloud") }
} else {
    $DataDir = Join-Path $BenchRoot "data"
    $ResultsDir = Join-Path $BenchRoot "results"
    $ExportModule = "benchmark.export_gt"
    $ExportCondaEnv = $ExportEnv
    if ($EvalMethods.Count -eq 0) { $EvalMethods = @("ours_pointcloud", "cosmo3d") }
}

$Manifest = Join-Path $DataDir "manifest.json"
$SubsetFile = Join-Path (Join-Path $DataDir "subsets") "$Subset.txt"
$SummaryJson = Join-Path $ResultsDir "summary_$Subset.json"
$SummaryMd = Join-Path $ResultsDir "summary_$Subset.md"
$ReportDir = Join-Path $ResultsDir "rendered_report"
$ReportHtml = Join-Path $ReportDir "report.html"

Write-Host "AgentSematic point-cloud benchmark" -ForegroundColor Green
Write-Host "  root       : $AgentRoot"
Write-Host "  dataset    : $Dataset"
Write-Host "  subset     : $Subset"
Write-Host "  eval       : $($EvalMethods -join ', ')"
Write-Host "  results    : $ResultsDir"

$needExport = $ForceExport -or (-not (Test-Path $Manifest)) -or (-not (Test-Path $SubsetFile))
if ($SkipExport) {
    Step "STAGE 1  skip GT export"
} elseif ($needExport) {
    if ($isShapeNetPart -and -not (Test-Path $ShapeNetPartRoot)) {
        throw @"
ShapeNet-Part root not found:
  $ShapeNetPartRoot

Pass the real *_normal dataset directory, for example:
  -Dataset shapenetpart -ShapeNetPartRoot E:\data\shapenetpart

Expected layout:
  <root>\test0.h5
  <root>\test1.h5

or:
  <root>\02691156\<model_id>.txt
  <root>\train_test_split\shuffled_test_file_list.json
"@
    }
    $exportArgs = [System.Collections.ArrayList]@("run", "-n", $ExportCondaEnv, "python", "-m", $ExportModule, "--build-subsets")
    if ($isShapeNetPart) {
        [void]$exportArgs.Add("--root")
        [void]$exportArgs.Add($ShapeNetPartRoot)
        Add-IfValue $exportArgs "--vocab" $ShapeNetPartVocab
    }
    RunCmd "STAGE 1  export GT and subsets" ([string[]]$exportArgs)
} else {
    Step "STAGE 1  reuse exported GT"
    Write-Host "manifest : $Manifest"
    Write-Host "subset   : $SubsetFile"
    Write-Host "use -ForceExport to rebuild"
}

if ($SkipRun) {
    Step "STAGE 2  skip ours_pointcloud"
} else {
    $runArgs = [System.Collections.ArrayList]@(
        "run", "-n", $AgentEnv, "python", "-m", "benchmark.run_method",
        "--dataset", $Dataset,
        "--method", "ours_pointcloud",
        "--subset", $Subset,
        "--provider", $Provider,
        "--max-depth", "$MaxDepth",
        "--max-seeds", "$MaxSeeds",
        "--min-block-size", "$MinBlockSize",
        "--parallel-vlm", "$ParallelVlm"
    )
    if ($Limit -gt 0) {
        [void]$runArgs.Add("--limit")
        [void]$runArgs.Add("$Limit")
    }
    if ($Overwrite) { [void]$runArgs.Add("--overwrite") }
    if ($NoViewSelect) { [void]$runArgs.Add("--no-view-select") }
    if ($NoNeo) { [void]$runArgs.Add("--no-neo") }
    Add-IfValue $runArgs "--model" $Model
    Add-IfValue $runArgs "--config" $Config
    Add-IfValue $runArgs "--env-file" $EnvFile
    Add-IfValue $runArgs "--neo-ckpt" $NeoCkpt
    Add-IfValue $runArgs "--neo-python" $NeoPython
    if ($NeoK -gt 0) {
        [void]$runArgs.Add("--neo-k")
        [void]$runArgs.Add("$NeoK")
    }
    Add-IfValue $runArgs "--neo-device" $NeoDevice
    if ($NeoDensifyTarget -gt 0) {
        [void]$runArgs.Add("--neo-densify-target")
        [void]$runArgs.Add("$NeoDensifyTarget")
    }
    RunCmd "STAGE 2  run ours_pointcloud" ([string[]]$runArgs)
}

if ($SkipEval) {
    Step "STAGE 3  skip mIoU evaluation"
} else {
    $evalArgs = [System.Collections.ArrayList]@(
        "run", "-n", $AgentEnv, "python", "-m", "benchmark.evaluate",
        "--dataset", $Dataset,
        "--subset", $Subset,
        "--methods"
    )
    foreach ($method in $EvalMethods) { [void]$evalArgs.Add($method) }
    RunCmd "STAGE 3  evaluate mIoU and point accuracy" ([string[]]$evalArgs)
}

if ($SkipReport) {
    Step "STAGE 4  skip HTML report"
} else {
    $reportArgs = [System.Collections.ArrayList]@(
        "run", "-n", $AgentEnv, "python", "-m", "benchmark.render_report",
        "--dataset", $Dataset,
        "--subset", $Subset,
        "--output-dir", $ReportDir,
        "--methods"
    )
    foreach ($method in $EvalMethods) { [void]$reportArgs.Add($method) }
    RunCmd "STAGE 4  render HTML report" ([string[]]$reportArgs)
}

Step "DONE"
Write-Host "summary json : $SummaryJson" -ForegroundColor Green
Write-Host "summary md   : $SummaryMd" -ForegroundColor Green
Write-Host "html report  : $ReportHtml" -ForegroundColor Green
Write-Host "predictions  : $(Join-Path $ResultsDir 'ours_pointcloud')" -ForegroundColor Green

if ($OpenReport -and (Test-Path $ReportHtml)) {
    Start-Process $ReportHtml
}

