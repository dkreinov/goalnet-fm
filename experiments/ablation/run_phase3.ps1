# Phase 3 ablation — detached, resumable. Uses the ADOPTED Phase-2 config (--beta 0 --w 1) so the
# momentum feature effect is not confounded. Single direct call (no array loop — PowerShell flattens
# single-element nested arrays). Resumable via per-seed caches.
$PY   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_phase3.log'
Set-Location $ROOT
"=== PHASE3 ctx-momentum START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'ctx-momentum' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '5' '--epochs' '150' '--ctx-extra' 'ctx_momentum.npz' '--notes' 'Elo-momentum + form-trajectory bundle vs beta0-W1' *>> $LOG
"=== PHASE3 ctx-momentum EXIT=$LASTEXITCODE ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
