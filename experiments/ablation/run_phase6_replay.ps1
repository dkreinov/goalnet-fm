# Phase 6 - WC2026 walk-forward replay, all 4 candidates x {frozen,finetune}, seeds=3. Detached.
$PY   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_phase6_replay.log'
Set-Location $ROOT
"=== PHASE6 replay START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/replay_wc.py' '--seeds' '3' '--epochs' '150' *>> $LOG
"=== PHASE6 replay EXIT=$LASTEXITCODE DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
