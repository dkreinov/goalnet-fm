# Phase 4 Arm A — odds as ctx feature. Detached, resumable (per-seed caches). Direct call (no array).
$PY   = 'C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe'
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_phase4.log'
Set-Location $ROOT
"=== PHASE4 ctx-odds START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/run_ablation.py' '--name' 'ctx-odds' '--beta' '0' '--w' '1' '--split' 'pooled' '--seeds' '5' '--epochs' '150' '--ctx-extra' 'ctx_odds.npz' '--notes' 'Arm A: Shin-devigged closing 1X2 (38k club from DB + 738 natl scraped) vs beta0-W1' *>> $LOG
"=== PHASE4 ctx-odds EXIT=$LASTEXITCODE ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
