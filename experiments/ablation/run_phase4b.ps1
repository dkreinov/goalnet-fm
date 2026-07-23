# Phase 4 remaining runs: B2 anchor sweep + Arm S stage. Sequential, detached, resumable.
$PY   = 'C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe'
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_phase4b.log'
Set-Location $ROOT
"=== PHASE4B START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'anchor-kl01' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '5' '--epochs' '150' '--market-anchor' '0.1' '--notes' 'B2: KL market anchor w=0.1' *>> $LOG
"### anchor-kl01 EXIT=$LASTEXITCODE ###" | Out-File -FilePath $LOG -Append -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'anchor-kl03' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '5' '--epochs' '150' '--market-anchor' '0.3' '--notes' 'B2: KL market anchor w=0.3' *>> $LOG
"### anchor-kl03 EXIT=$LASTEXITCODE ###" | Out-File -FilePath $LOG -Append -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'ctx-stage' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '5' '--epochs' '150' '--ctx-extra' 'ctx_stage.npz' '--notes' 'Arm S: knockout/stage flag (collect-then-test debt)' *>> $LOG
"### ctx-stage EXIT=$LASTEXITCODE ###" | Out-File -FilePath $LOG -Append -Encoding utf8
"=== PHASE4B ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
