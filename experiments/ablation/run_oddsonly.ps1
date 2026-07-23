# Experiment: how much do FM player grades add over odds? Train ctx+odds ONLY (players zeroed), beta0, 3-seed.
$PY   = 'C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe'
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_oddsonly_exp.log'
Set-Location $ROOT
"=== ODDS-ONLY EXP START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'src/train_goals.py' '--full' '--odds' '--no-players' '--beta' '0' '--ensemble' '3' '--out' 'models/exp/goalnet_oddsonly_s3.pt' *>> $LOG
"=== ODDS-ONLY EXP EXIT=$LASTEXITCODE DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
