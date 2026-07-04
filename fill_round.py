"""Pre-fill picks for a whole round NOW (safety net so no game is missed if the cron misses its 60-80min
window). Reuses auto_bet's tested model_pick (round-aware strategy, orientation, fallback XIs) + api submit.
auto_bet later UPGRADES these to confirmed-lineup picks in the window. Dry by default; --submit writes.
Usage: python fill_round.py r16 [--submit]
"""
import sys, json, os
import auto_bet as ab

rnd = sys.argv[1] if len(sys.argv) > 1 else "r16"
SUBMIT = "--submit" in sys.argv
bearer, uid = ab.get_access()
fx = ab.api(ab.BASE + "/rest/v1/fixtures?select=id,home_team,away_team,round,status,kickoff_utc&order=kickoff_utc.asc", bearer=bearer)
games = [f for f in fx if f["round"] == rnd and f["status"] == "scheduled"]
mine = {p["fixture_id"] for p in ab.api(ab.BASE + f"/rest/v1/picks?select=fixture_id&user_id=eq.{uid}", bearer=bearer)}
lu = json.load(open(os.path.join(ab.WC, "lineups.json"), encoding="utf-8"))
results = json.load(open(os.path.join(ab.WC, "results.json"), encoding="utf-8"))
rosters = ab.load_rosters()

print(f"=== {rnd.upper()}: {len(games)} scheduled, {len(mine & {g['id'] for g in games})} already picked ===")
print(f"mode: {'SUBMIT' if SUBMIT else 'DRY (show only)'}\n")
wrote = skipped = nopred = 0
for f in games:
    H, A = f["home_team"], f["away_team"]; ko = f["kickoff_utc"][:16]
    if f["id"] in mine:
        print(f"  {H}-{A:<4} {ko}  already picked — skip"); skipped += 1; continue
    mp = ab.model_pick(f, lu, results, rosters)
    if not mp:
        print(f"  {H}-{A:<4} {ko}  NO PREDICTION (missing lineups/fallback)"); nopred += 1; continue
    hs, as_, tag = mp
    line = f"  {H} {hs}-{as_} {A}   [{tag}, KO {ko}]"
    if SUBMIT:
        ab.api(ab.BASE + "/rest/v1/picks", "POST", [{"user_id": uid, "fixture_id": f["id"], "home_score": hs, "away_score": as_}], bearer=bearer)
        chk = ab.api(ab.BASE + f"/rest/v1/picks?select=home_score,away_score&fixture_id=eq.{f['id']}&user_id=eq.{uid}", bearer=bearer)
        ok = chk and chk[0]["home_score"] == hs and chk[0]["away_score"] == as_
        print(line + ("  WROTE+verified" if ok else "  WRITE FAILED")); wrote += ok
    else:
        print(line + "  [DRY]")
print(f"\n{'wrote' if SUBMIT else 'would write'}: {wrote if SUBMIT else len(games)-skipped-nopred} | already: {skipped} | no-pred: {nopred}")
