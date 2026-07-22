# Phase 5 Step 6 — Arm P2: plus-minus as per-player channels (A 62->64), goalnet arch, s3.
$PY   = 'C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe'
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_phase5c.log'
Set-Location $ROOT
"=== PHASE5C pm-channel-s3 START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'pm-channel-s3' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '3' '--epochs' '150' '--pm-channel' 'players_pm.npz' '--notes' 'Phase 5 Arm P2: per-slot shrunk net-of-club plus-minus [pm,has_pm] appended to X (A 62->64), goalnet, s3' *>> $LOG
"=== PHASE5C pm-channel-s3 EXIT=$LASTEXITCODE ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
