# Phase 2 Step 4 combo run — detached, resumable. Tests whether the two de-biasing
# winners stack: beta=0 (no decision term) + W=1 (no national upweight).
$PY   = 'C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe'
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_combo.log'
Set-Location $ROOT
"=== COMBO START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'combo-beta0-w1' '--beta' '0' '--w' '1' '--seeds' '5' '--epochs' '150' '--notes' 'combo: pure-Poisson (beta0) + no-upweight (W1) - do the de-biasing levers stack?' *>> $LOG
"=== COMBO EXIT=$LASTEXITCODE ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
