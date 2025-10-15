param(
  [string]$AliasesCsv = "src/research/RQ2/data/aliases_review.csv",
  [string]$CentralVersion = "v2_alignment_cctr",
  [string]$FlVersion = "v2_alignment_flctr",
  [int]$Epochs = 5,
  [int]$Rounds = 20,
  [int]$LocalEpochs = 1,
  [double]$DpClip = 0.0,
  [double]$DpNoise = 0.0
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $AliasesCsv)) {
  Write-Error "Aliases CSV not found: $AliasesCsv"
  exit 1
}

New-Item -ItemType Directory -Force src/research/RQ2/logs | Out-Null
$ts = Get-Date -Format "yyyyMMdd_HHmmss"

function Run-Step {
  param([string]$Cmd, [string]$Log)
  Write-Host "[RUN] $Cmd"
  # Use cmd.exe /c to preserve quoting reliably
  cmd.exe /c $Cmd 2>&1 | Tee-Object -FilePath $Log
}

Write-Host "[STEP] Centralized contrastive training..."
Run-Step "python -m src.research.RQ2.encoders.train_contrastive --version $CentralVersion --aliases_csv $AliasesCsv --epochs $Epochs" "src/research/RQ2/logs/centralized_contrastive_train_${ts}.log"

Write-Host "[STEP] Centralized intrinsic eval..."
Run-Step "python -m src.research.RQ2.eval.run_intrinsic_eval --version $CentralVersion" "src/research/RQ2/logs/intrinsic_eval_cctr_${ts}.log"

Write-Host "[STEP] Centralized build aligned features..."
Run-Step "python -m src.research.RQ2.pipelines.build_aligned_features --version $CentralVersion --notes \"Centralized contrastive encoder\"" "src/research/RQ2/logs/build_features_cctr_${ts}.log"

Write-Host "[STEP] FL contrastive training..."
$flCmd = "python -m src.research.RQ2.federation.train_fl_contrastive --version $FlVersion --aliases_csv $AliasesCsv --rounds $Rounds --local_epochs $LocalEpochs"
if ($DpClip -gt 0 -or $DpNoise -gt 0) {
  $flCmd = "$flCmd --dp_clip $DpClip --dp_noise $DpNoise"
}
Run-Step $flCmd "src/research/RQ2/logs/fl_contrastive_train_${ts}.log"

Write-Host "[STEP] FL intrinsic eval..."
Run-Step "python -m src.research.RQ2.eval.run_intrinsic_eval --version $FlVersion" "src/research/RQ2/logs/intrinsic_eval_flctr_${ts}.log"

Write-Host "[STEP] FL build aligned features..."
Run-Step "python -m src.research.RQ2.pipelines.build_aligned_features --version $FlVersion --notes \"FL contrastive encoder\"" "src/research/RQ2/logs/build_features_flctr_${ts}.log"

Write-Host "[STEP] Aggregate results..."
Run-Step "python -m src.research.RQ2.eval.aggregate_results --versions $CentralVersion $FlVersion --out src/research/RQ2/logs/agg_${ts}.json" "src/research/RQ2/logs/aggregate_${ts}.log"

Write-Host "[OK] Completed. Logs in src/research/RQ2/logs and outputs under data/processed/$CentralVersion and data/processed/$FlVersion."

