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

## Other deferred
- DD/MM vs MM/DD DOB swap suspected for a small number of players (e.g. tosaint ricketts ESPN 1987-08-06 vs FM 1987-06-08).
  Low volume; check if it's a TM-parse format issue (`scrape_tm_squads`/`scrape_tm_residual` parse DD/MM/YYYY).
- Managers: fminside has no managers; needs Transfermarkt (manager-per-club) — task #15.
