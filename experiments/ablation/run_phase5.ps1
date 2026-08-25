# Phase 5 Arm X — cross-team attention exploratory (seeds=3). Detached, resumable (per-seed caches).
$PY   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_phase5.log'
Set-Location $ROOT
"=== PHASE5 arch-cross22-s3 START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'arch-cross22-s3' '--arch' 'cross22' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '3' '--epochs' '150' '--notes' 'Phase 5 Arm X exploratory: 22-token joint transformer (cross-team attention), s3, vs combo-beta0-w1' *>> $LOG
"=== PHASE5 arch-cross22-s3 EXIT=$LASTEXITCODE ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
