# Experiment: does beta>0 + odds beat calibrated beta0+odds ON COMPETITION POINTS?
# Trains beta in {0,1,3} + odds at 3-seed (apples-to-apples) to models/exp/. Production data/goalnet.pt untouched.
$PY   = 'C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe'
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_beta_odds_exp.log'
Set-Location $ROOT
"=== BETA+ODDS EXP START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
foreach ($b in 0,1,3) {
  "--- training beta=$b + odds (3-seed) $(Get-Date -Format o) ---" | Out-File -FilePath $LOG -Append -Encoding utf8
  & $PY 'src/train_goals.py' '--full' '--odds' '--beta' "$b" '--ensemble' '3' '--out' "models/exp/goalnet_b${b}_odds_s3.pt" *>> $LOG
  "--- beta=$b EXIT=$LASTEXITCODE $(Get-Date -Format o) ---" | Out-File -FilePath $LOG -Append -Encoding utf8
}
"=== BETA+ODDS EXP DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
