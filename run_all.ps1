<#
  run_all.ps1 -- the benchmark of the paper
  "How Much Space Is Enough for Short-Term Satellite Irradiance Forecasting?"

  MATLAB fits and scores every model. Python trains the U-Net and the LSTM when
  MATLAB asks for them, and never computes a score. This script checks the
  tools, runs the MATLAB steps one after the other and keeps a log of each.

  Modes
    tables  rebuild the paper's two result tables from the archives kept in
            paper_results\results. No model is fitted.
    light   every model of the paper, with one random draw per ELM and one
            payload per code. The parameters below set it.
    full    the campaign as the paper ran it. The light parameters are ignored.

  Examples
    .\run_all.ps1
    .\run_all.ps1 -Mode light
    .\run_all.ps1 -Mode light -Only bench,blend
    .\run_all.ps1 -Mode light -Codes dct,pca -Payload 49
    .\run_all.ps1 -Mode light -Python C:\Python312\python.exe
#>
param(
  [ValidateSet('tables', 'light', 'full')]
  [string]   $Mode     = 'tables',
  [string]   $Matlab   = 'matlab',
  [string]   $Python   = 'python',

  # light mode; the paper's value is in brackets
  [int]      $Draws    = 1,                  # random hidden layers per ELM arm, the median one is kept [50]
  [string[]] $Payload  = @('196'),           # numbers kept per map by every code [49, 196, 392]
  [string[]] $Codes    = @('radon', 'dct', 'wavelet', 'pca', 'randproj', 'subsample', 'autoencoder'),
  [string[]] $UnetBase = @('8'),             # U-Net widths [8, 16, 24]
  [string[]] $LstmReps = @('field', 'dct'),  # what the LSTM reads [field, radon, dct, pca]
  [string[]] $Patch    = @('1', '3'),        # patch sides of the neighborhood sweep [1, 3, 5, 7]

  [string[]] $Only     = @('all')            # any of: all, bench, blend, patch, unet, lstm
)

$ErrorActionPreference = 'Stop'

# "powershell -File" hands over "dct,pca" as one string, so every list is split here
function Split-List($values) {
  @($values | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
$Payload  = @(Split-List $Payload  | ForEach-Object { [int]$_ })
$UnetBase = @(Split-List $UnetBase | ForEach-Object { [int]$_ })
$Patch    = @(Split-List $Patch    | ForEach-Object { [int]$_ })
$Codes    = Split-List $Codes
$LstmReps = Split-List $LstmReps
$Only     = Split-List $Only
foreach ($s in $Only) {
  if ($s -notin 'all', 'bench', 'blend', 'patch', 'unet', 'lstm') {
    throw "unknown step '$s': use all, bench, blend, patch, unet or lstm"
  }
}
$root = $PSScriptRoot
$logs = Join-Path $root 'results\logs'
New-Item -ItemType Directory -Force $logs | Out-Null

function Test-Tool($name, $exe) {
  if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
    throw "$name not found: '$exe'. Put it on the PATH or pass -$name with the full path."
  }
}

function MatlabCells($items) { "{'" + ($items -join "', '") + "'}" }
function MatlabRow($numbers) { "[" + ($numbers -join ' ') + "]" }

function Invoke-Matlab($step, $command, $folder) {
  $log = Join-Path $logs "$step.log"
  Write-Host ""
  Write-Host "=== $step ===   matlab -batch `"$command`""
  $t0 = Get-Date
  Push-Location $folder
  try {
    & $Matlab -batch $command | Tee-Object -FilePath $log
    $code = $LASTEXITCODE
  } finally {
    Pop-Location
  }
  $line = "{0:yyyy-MM-dd HH:mm}  {1,-6}  {2,8:N1} min  exit {3}" -f $t0, $step, ((Get-Date) - $t0).TotalMinutes, $code
  # a busy log file (antivirus, indexer, an open editor) must not stop the run
  try {
    Add-Content -Path (Join-Path $logs 'run_log.txt') -Value $line -ErrorAction Stop
  } catch {
    Write-Warning "could not write run_log.txt: $($_.Exception.Message)"
  }
  Write-Host $line
  if ($code -ne 0) { throw "step '$step' failed, see $log" }
}

Test-Tool 'Matlab' $Matlab

# ------------------------------------------------------------------ tables
if ($Mode -eq 'tables') {
  $folder = Join-Path $root 'paper_results'
  Invoke-Matlab 'tables' "addpath('$root'); hms_results_tables(fullfile(pwd, 'tables'))" $folder
  Write-Host ""
  Write-Host "The paper's tables are in $(Join-Path $folder 'tables')"
  return
}

# ------------------------------------------------------------ light or full
if ($Mode -eq 'full') {
  $Draws    = 50
  $Payload  = @(49, 196, 392)
  $Codes    = @('radon', 'dct', 'wavelet', 'pca', 'randproj', 'subsample', 'autoencoder')
  $UnetBase = @(8, 16, 24)
  $LstmReps = @('field', 'radon', 'dct', 'pca')
  $Patch    = @(1, 3, 5, 7)
}

if ($Only -contains 'all') { $steps = @('bench', 'blend', 'patch', 'unet', 'lstm') } else { $steps = $Only }

foreach ($f in 'Basic\GHI_HC3.mat', 'Basic\geopoint.mat', 'results\hyper_frozen.mat', 'results\scale_ref.mat') {
  if (-not (Test-Path (Join-Path $root $f))) { throw "missing input file: $f" }
}

if ($steps -contains 'unet' -or $steps -contains 'lstm') {
  Test-Tool 'Python' $Python
  & $Python -c "import numpy, scipy, h5py, torch"
  if ($LASTEXITCODE -ne 0) {
    throw "Python needs numpy, scipy, h5py and torch: $Python -m pip install -r python\requirements.txt"
  }
  # MATLAB calls the interpreter by the name 'python', so the chosen one goes first on the PATH
  $env:Path = (Split-Path (Get-Command $Python).Source) + ';' + $env:Path
}

$previous = Get-ChildItem (Join-Path $root 'results') -Filter '*.mat' |
            Where-Object { $_.Name -notin 'hyper_frozen.mat', 'scale_ref.mat' }
if ($previous) {
  Write-Host "results\ already holds $($previous.Name -join ', ')."
  Write-Host "The bench, patch, unet and lstm steps resume and skip the arms these files hold. Move them away for a fresh run."
}

$commands = [ordered]@{
  bench = "hms_forecast_bench(fullfile(pwd, 'results', 'bench.mat'), $Draws, $(MatlabCells $Codes), $(MatlabRow $Payload))"
  blend = "hms_blend_arm(fullfile(pwd, 'results', 'blend_bench.mat'))"
  patch = "hms_neighbourhood($(MatlabRow $Patch), fullfile(pwd, 'results', 'neighbourhood.mat'))"
  unet  = "hms_unet_arm(fullfile(pwd, 'results', 'unet_bench.mat'), $(MatlabRow $UnetBase))"
  lstm  = "hms_lstm_arm($(MatlabCells $LstmReps), $(MatlabRow $Payload), fullfile(pwd, 'results', 'lstm_bench.mat'))"
}

$start = Get-Date
foreach ($s in $commands.Keys) {
  if ($steps -contains $s) { Invoke-Matlab $s $commands[$s] $root }
}
Invoke-Matlab 'tables' "hms_results_tables(fullfile(pwd, 'results', 'tables'))" $root

Write-Host ""
Write-Host ("Done in {0:N1} h. Tables in results\tables, logs in results\logs." -f ((Get-Date) - $start).TotalHours)
