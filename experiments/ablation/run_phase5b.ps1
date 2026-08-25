# Phase 5 chained: Step 5 ctx-pm-s3 (Arm P1) then Step 3 fallback arch-latecross-s3. Detached, resumable.
$PY   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_phase5b.log'
Set-Location $ROOT
"=== PHASE5B ctx-pm-s3 START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'ctx-pm-s3' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '3' '--epochs' '150' '--ctx-extra' 'ctx_pm.npz' '--notes' 'Phase 5 Arm P1 cheap: team-aggregate shrunk net-of-club plus-minus diff + coverage as ctx, s3' *>> $LOG
"=== PHASE5B ctx-pm-s3 EXIT=$LASTEXITCODE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
"=== PHASE5B arch-latecross-s3 START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'arch-latecross-s3' '--arch' 'latecross' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '3' '--epochs' '150' '--notes' 'Phase 5 Arm X fallback: per-team encoder + one late cross-attention block, s3 (cross22-s3 was below baseline)' *>> $LOG
"=== PHASE5B arch-latecross-s3 EXIT=$LASTEXITCODE ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
