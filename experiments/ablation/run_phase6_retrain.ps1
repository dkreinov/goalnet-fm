# Phase 6 Step 5 - production retrain: goalnet v2 = beta0/W1 + ctx-odds feature, 5-seed ensemble.
# v1 already archived to models/archive/goalnet_v1_20260723.pt. Overwrites data/goalnet.pt.
$PY   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_goalnet_v2_retrain.log'
Set-Location $ROOT
"=== GOALNET V2 RETRAIN START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'src/train_goals.py' '--full' '--odds' '--ensemble' '5' *>> $LOG
"=== GOALNET V2 RETRAIN EXIT=$LASTEXITCODE DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
