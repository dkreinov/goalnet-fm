# Plan 2 — Targeted FM-grade scrape for the truly-absent starters (the ~13k-appearance data gap)

Status: QUEUED (run AFTER Plan 1, the club-anchored fuzzy/alias matcher in build_xwalk).
Repo: D:\Programming\claude\FM. Single DB writer at a time (WAL). Keep test_xwalk 8/8. Back up data/fm.db first.

## What this targets
After the un-merge + ESPN-side fix + DOB-gate fix + fuzzy/alias matcher, the residual missing starters split
(diag_missing.py / the NOT_IN_FM probe) into:
- in-FM under a variant name -> handled by Plan 1 (fuzzy/alias, no scraping).
- **truly absent: ~1,180 players / ~12,895 starter-appearances** with NO token overlap to any FM grade name
  (Sam Edmundson, Mexer, Ngonda Muzinga, Jón Böðvarsson, Cheyenne Dunkley...). These are mostly lower-league
  English (Champ/L1/L2/National) + Scandinavian players we never grade-scraped, OR genuinely not in FM.

## Approach (targeted, NOT a full re-enumeration)
1. Build the work-list: re-run the NOT_IN_FM 'NO token overlap' classifier; for each player collect ESPN
   name + clubs + seasons + nationality (from source_identity espn). Persist to a json work-list.
2. fminside has a FREE-TEXT 'name' filter + 'nationality' filter (FILTER_DEFAULTS in scrape_fminside).
   For each absent player, POST update_filter.php with name+nationality, enumerate the result player pages,
   fetch each (fetch.get cache), parse_player. (Player pages are URL-driven/cacheable; only the search POST
   is live — same per-IP session caveat, so run SINGLE worker, serial.)
3. MATCH guard before saving: accept a candidate only if (norm name close) AND (club matches one of the
   player's ESPN clubs for that edition, via club_id/alias) OR DOB matches. Reject otherwise — never save a
   wrong-person grade (that re-introduces the bug we just fixed). Save via db.player_id(grade_uid=True) +
   save_snapshot, so identity stays FM-UID-keyed.
4. Log misses (players fminside genuinely lacks) to FUTURE_WORK so we stop re-trying them.

## Validation
- new fminside snapshots saved > 0; spot-check 5 recovered players link in build_xwalk.
- re-run build_xwalk -> roster_match -> build_dataset --no-flag -> test_xwalk (MUST stay 8/8) -> report_missing.
- report the readiness delta; expect SMALL (this is the genuine-absent tail, ~13k apps spread thin).

## Caveats / why this is last
- Lower yield than Plan 1: each absent player unblocks a match only if ALL of that match's other 21 starters
  are already graded. Many absent players are in lower leagues that are quality-EXCLUDED from training anyway.
- Network-bound + fminside fragility (43s pages, 504s); a name search may return many homonyms -> the club/DOB
  guard is essential. Some players are simply not in FM (retired pre-FM-edition, very low tier) -> unrecoverable.
- If yield is tiny on a sample (first ~100 players), STOP and just document the residual as a hard floor.

Delete this file when done or when the sample proves yield is not worth the scrape.
