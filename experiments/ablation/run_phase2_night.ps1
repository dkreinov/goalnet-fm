# Phase 2 overnight queue — detached, resumable, session-independent.
# Runs the combo + 3 extension validations sequentially (one training at a time; 4-core box).
# Each run auto-resumes from per-seed caches; a config that already has a registry row is skipped.
#
#   1. combo-beta0-w1        beta0 + W1, pooled 5-seed   (the adopted-config candidate; resumes)
#   2. combo-beta0-w1-decay8 + decay hl8, pooled 5-seed  (does decay add on top of the debiased core?)
#   3. combo-beta0-w1-canon  beta0 + W1, CANONICAL 5-seed(continuity numbers for the adopted config)
#   4. combo-beta0-w1-s10    beta0 + W1, pooled 10-seed  (robustness: does the 5-seed estimate hold?)

$ErrorActionPreference = 'Continue'
$PY   = 'C:\Users\youruser\AppData\Local\Programs\Python\Python312\python.exe'
$ROOT = 'D:\Programming\claude\FM'
$LOG  = Join-Path $ROOT 'data\_ablation_night.log'
Set-Location $ROOT

$runs = @(
  @('combo-beta0-w1',        @('--beta','0','--w','1','--split','pooled','--seeds','5','--notes','combo: beta0 + W1 (adopted-config candidate)')),
  @('combo-beta0-w1-decay8', @('--beta','0','--w','1','--decay-halflife','8','--split','pooled','--seeds','5','--notes','3-lever: beta0 + W1 + decay hl8')),
  @('combo-beta0-w1-canon',  @('--beta','0','--w','1','--split','canonical','--seeds','5','--notes','beta0 + W1 canonical continuity')),
  @('combo-beta0-w1-s10',    @('--beta','0','--w','1','--split','pooled','--seeds','10','--notes','beta0 + W1 robustness 10-seed'))
)

"=== PHASE2 NIGHT START $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Encoding utf8
foreach ($r in $runs) {
  $name = $r[0]; $extra = $r[1]
  "### $name START $(Get-Date -Format o) ###" | Out-File -FilePath $LOG -Append -Encoding utf8
  $argv = @('experiments/ablation/run_ablation.py','--name',$name,'--epochs','150') + $extra
  & $PY @argv *>> $LOG
  "### $name EXIT=$LASTEXITCODE $(Get-Date -Format o) ###" | Out-File -FilePath $LOG -Append -Encoding utf8
}
"=== PHASE2 NIGHT ALL DONE $(Get-Date -Format o) ===" | Out-File -FilePath $LOG -Append -Encoding utf8
