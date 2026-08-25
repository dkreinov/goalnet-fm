# Multi-tournament leakage-free backtest: FM+odds vs odds-only, train<2024-06, eval Euro24+Copa24+NL24-25.
$PY   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_backtest_tournaments.log'
Set-Location $ROOT
"=== BACKTEST START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
& $PY 'experiments/ablation/backtest_tournaments.py' *>> $LOG
"=== BACKTEST EXIT=$LASTEXITCODE DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
