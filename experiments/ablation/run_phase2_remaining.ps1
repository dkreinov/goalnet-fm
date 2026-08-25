# Phase 2 remaining ablation runs — detached, resumable, session-independent.
# Runs the decay sweep (hl 2/4/8) then the W recheck (1/40), sequentially.
# Each run auto-resumes from per-seed caches (rates/<name>.s<k>.npz); a config that
# already has a registry row is refused (exit 1) and skipped. Safe to re-run anytime.
#
# Launch (either works):
#   powershell -ExecutionPolicy Bypass -File D:\Programming\claude\FM\experiments\ablation\run_phase2_remaining.ps1
#   Start-Process -WindowStyle Hidden powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','D:\Programming\claude\FM\experiments\ablation\run_phase2_remaining.ps1'

$ErrorActionPreference = 'Continue'
$PY   = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_phase2.log'
Set-Location $ROOT

# name, extra-args (as array)
$runs = @(
  @('decay-hl2', @('--decay-halflife','2','--notes','time-decay half-life 2y')),
  @('decay-hl4', @('--decay-halflife','4','--notes','time-decay half-life 4y')),
  @('decay-hl8', @('--decay-halflife','8','--notes','time-decay half-life 8y')),
  @('beta3-w1',  @('--w','1','--notes','no national upweight; null test')),
  @('beta3-w40', @('--w','40','--notes','heavy national upweight'))
)

"=== PHASE2 REMAINING START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
foreach ($r in $runs) {
  $name = $r[0]; $extra = $r[1]
  "### $name START $(Get-Date -Format o) ###" | Out-File -FilePath $LOG -Append -Encoding utf8
  $argv = @('experiments/ablation/run_ablation.py','--name',$name,'--seeds','5','--epochs','150') + $extra
  & $PY @argv *>> $LOG
  "### $name EXIT=$LASTEXITCODE $(Get-Date -Format o) ###" | Out-File -FilePath $LOG -Append -Encoding utf8
}
"=== PHASE2 REMAINING ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
