# Future work / known issues

## BUG: scrape_clubs under-collects (club_reputation feature only ~17% populated)
Discovered 2026-06-17 via `src/audit_simple.py`.

- `club_attribute` holds only **69 distinct clubs (EPL + Japan)** out of ~620 expected; only **199 of 2,183**
  `(club, fm_version)` pairs overlap with `player_snapshot`, so `build_dataset`'s reputation/facility join hits ~17%.
- `scrape_clubs.py` printed "20 clubs" + "saved 20" for all 246 league×edition blocks, but they collapsed to 69 distinct
  `club_id`s — i.e. the club pages for ~29 leagues (LaLiga/Serie A/Bundesliga/…) did **not** parse into distinct clubs.
  Only English + Japanese club pages produced real names. Suspect: `club_urls()` enumeration returned a stale/duplicate
  club set for non-English leagues, OR `parse_club` name extraction fails on those pages, OR the parallel club-page
  fetch (workers=6, min_delay=0) got rate-limited so most pages errored.
- Effect: `{side}_club_reputation` + facilities features are mostly NULL. `{side}_squad_value_total` is also only ~69%
  (the club-bridge in build_dataset is imperfect — dominant grade-club for a side sometimes has no squad aggregate).
- FIX (when prioritized): debug `scrape_clubs.club_urls()`/`parse_club` per non-English league (print enumerated club
  names per league); ensure distinct club_ids; re-scrape; then verify `audit_simple.py` shows reputation >~80%.

## INVESTIGATE: identity (DOB) records vs grade records disconnected for common names
Discovered 2026-06-17 while tracing the top unmatched starter "João Pedro" (490 starts, ESPN clubs Watford/Cagliari/Grêmio).
- `source_identity` (source 'fm-uid') has 43 entries named "João Pedro" (name+DOB from the kaggle identity extract).
- Those 43 fm-uids map to **0** internal player_ids via `player_source_id` — i.e. the identity records used for DOB
  disambiguation in build_xwalk are NOT joined to the actual GRADE player rows for common names.
- Consequence: build_xwalk sees N same-name candidates but they carry no grade-club (uid_to_gradepids empty), so the
  club-season squad disambiguation can't pick the right one (only 6 upgraded globally). ESPN DOB for these is often wrong
  (João Pedro = 2008-03-18, impossible) so DOB can't rescue it either.
- The real João Pedro's grades DO exist (snapshots at Watford FM22 / Brighton FM24 / Chelsea FM26 — his real career),
  just not reachable from the 43 identity uids. NOTE: several throwaway diagnostic queries this session had unpack/order
  bugs — VERIFY this cleanly before acting (count distinct player_ids per same-name uid set; confirm grade-player rows
  for "joao pedro" and whether ANY identity uid joins to them).
- Likely high-leverage: common Brazilian/Portuguese/Spanish names dominate the unmatched bucket; repairing identity<->grade
  linkage for them (e.g. link by fm-uid == fminside/kaggle player-id, or by name+club+season trajectory) could lift the
  connected-games % more than any scraping. Threshold-tier (gold/silver 10-11+impute) remains the fallback.

## Other deferred
- DD/MM vs MM/DD DOB swap suspected for a small number of players (e.g. tosaint ricketts ESPN 1987-08-06 vs FM 1987-06-08).
  Low volume; check if it's a TM-parse format issue (`scrape_tm_squads`/`scrape_tm_residual` parse DD/MM/YYYY).
- Managers: fminside has no managers; needs Transfermarkt (manager-per-club) — task #15.
