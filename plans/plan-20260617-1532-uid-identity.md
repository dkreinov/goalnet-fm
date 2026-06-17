# Plan — Fix grade-player name-merge: key grade-players by FM-UID (the real lever for the 59% gap)

Plan file: `plans/plan-20260617-1532-uid-identity.md`
Execution mode: **autonomous**. Model: Opus / **Extra high** (data-model surgery + correctness). Branch: master.
Repo: D:\Programming\claude\FM (Python pipeline; NOT ballanz_squad — ignore TDD/E2E/wiki rules; use runnable validations).

## READ FIRST (this session has no memory of the investigation)
Also read: `C:\Users\youruser\.claude\projects\D--Programming-claude-FM\memory\fm-ratings-project.md` and `FUTURE_WORK.md`.

## The bug (root cause — CLEANLY VERIFIED 2026-06-17)
`src/db.py player_id()` merges different FM players that share a normalized name into ONE internal `player_id`.
Evidence: internal player_id 126 ("João Pedro") has grade snapshots at **4 different clubs (Watford / Brighton /
Real Valladolid / Chelsea)** — different real people pooled onto one row. Grade sources (fminside, kaggle, futek) ALL
key by the **FM game-UID** as `player_source_id.source_player_id` (verified: Salah = 98028755 across fminside/futek/kaggle;
ESPN uses its own id 173896). So the FM-UID is a reliable per-real-player key, but the name-merge overrides it.

## Why this is THE lever for the not-ready gap
- Match-readiness is stuck at 59% (47,739/81,357); 54% of not-ready target games miss exactly ONE starter. The missing
  starters are dominated by COMMON NAMES (João Pedro = 43 FM namesakes, James Wilson, Bruno×5, etc.). They are 'ambiguous'
  in player_xwalk because name->many candidates and DOB is often wrong/missing.
- The club SHOULD disambiguate ("our João Pedro played at Watford -> the FM João Pedro at Watford"), but it CAN'T because
  all the same-name grades are merged onto one player_id spanning many clubs. Un-merging by FM-UID gives each real player
  their own grade-player with their own club(s), so build_xwalk's club-season squad disambiguation (already implemented,
  method 'name+squad') will finally fire for common names. Expected to lift connected-games % more than any scraping.
- Player-level coverage is already 96.2%; top-5 leagues fully linked; we've exhausted scraping/DOB approaches. This is the
  remaining real-data lever.

## Frozen contracts / guards
- `src/test_xwalk.py` (8 authenticity tests) MUST stay green — the false-merge gate. Run after every xwalk rebuild.
- ESPN<->FM linking belongs in build_xwalk (name+DOB+club -> FM-UID); grade-player identity should be the FM-UID, NOT name.
- Cross-source name-merge for ESPN (which has no FM-UID) is the LEGITIMATE linking path and must keep working via xwalk.
  The bug is WITHIN grade sources: two distinct FM-UIDs (same source) sharing a name must be DISTINCT players.
- Disk cache at `data/cache` holds fminside player pages (URL-keyed by FM-UID) -> re-loading grades is cheap/offline.
- Single DB writer at a time (WAL; concurrent writers caused "database is locked" before). Batched scraper has _begin/_commit guards.
- Baseline to beat: training-ready 59%; ambiguous 2,472; linked-with-grades 42,217; test_xwalk 8/8.

## Plan Review Summary
- Passes: 3. Concerns: (1) snapshots already saved carry player_id=merged, NOT the FM-UID directly -> the un-merge must
  re-key by FM-UID, easiest by RE-LOADING grades from cache with a fixed db.player_id (Step 3), not by guessing which
  merged snapshot belongs to which UID. (2) Must not break ESPN cross-source linking. (3) DB reset risk -> do it on a COPY
  / with a clear migration, never blow away ESPN/match data. Keep a backup of fm.db before re-keying.
- Confidence: ready; this is surgery so go slow, validate each step.

---

## Step 1: Verify + size the merge (read-only)
Files: create src/diag_merge.py.
Goal: confirm root cause + quantify. (a) # internal grade-players whose snapshots span >=2 distinct clubs in the SAME
season (over-merge); (b) # internal grade-players carrying >=2 distinct FM-UIDs (player_source_id grade sources);
(c) of the ~5,437 unmatched/ambiguous target starters, how many have a same-name FM-UID that, IF un-merged, would have a
single grade-club matching the ESPN player's club (the recoverable estimate). Validation: prints all three; recoverable>0.

## Step 2: Read db.player_id + the grade loaders; design the re-key
Files: read src/db.py (player_id, save_snapshot, player_source_id usage), src/scrape_fminside.py (_save_parsed),
src/kaggle_load.py, src/scrape_futek.py. Goal: pinpoint where name-merge happens and design: grade-player identity =
(grade FM-UID); ESPN stays its own player linked via xwalk. Decide migration-in-place vs reload-from-cache.
Validation: written design note in the plan (no code yet) naming the exact change + chosen mechanism.

## Step 3: Implement FM-UID-keyed grade identity + reload grades
Files: src/db.py (and/or a new src/reload_grades.py). BACK UP data/fm.db first (copy to data/fm.db.bak).
Goal: make db.player_id (or a dedicated grade-player resolver) key grade snapshots by (source, FM-UID) so each distinct
FM-UID is a distinct grade-player; re-load fminside (from cache) + kaggle + futek so snapshots re-key correctly. Do NOT
touch ESPN players / match_player / matches. Single writer.
Validation: player_id 126 no longer spans 4 clubs; "joão pedro" now resolves to MULTIPLE distinct grade-players each at
one club; row counts sane; ESPN players untouched (match_player count unchanged).

## Step 4: Rebuild xwalk + dataset + measure
Files: none. Run build_xwalk -> roster_match -> build_dataset --no-flag -> test_xwalk -> report_missing -> report_players.
Goal: the club-season squad disambiguation ('name+squad') now fires for common names. Validation: test_xwalk ALL PASS;
ambiguous count DOWN materially; training-ready UP vs 59%; spot-check João Pedro @ Watford links to the Watford FM-UID.

## Step 5: Commit + report
Commit db.py / reload script + measure before/after. If training-ready rose, great; if not, report honestly (the merge
may be downstream of a different issue). rm this plan file on success.

---
Delete this file (`plans/plan-20260617-1532-uid-identity.md`) after Step 5 validation passes. Until then it is the anchor.
If Step 1 shows the recoverable estimate is small (<~1,000 starters), STOP and report — the merge may not be the main lever
and the gold/silver threshold tier (10/11 + impute) is the pragmatic alternative.
