"""Join scraped BetExplorer odds to our matches, Shin de-vig, emit Phase-4 feature bundles.

Reads data/natl_odds_raw.csv (from scrape_betexplorer.py), joins to the match table by
(date +/-1 day, normalized home/away names, both orientations), removes the bookmaker margin with
Shin's method, and writes:
  data/ctx_odds.npz  {mids, feats:[pH, pD, pA, ln_overround, has_odds]}  (de-vigged closing 1X2)
  data/ctx_stage.npz {mids, feats:[knockout, has_stage]}                 (ET/pen knockout marker)
Only matched matches are stored; unmatched training matches auto-get zeros+mask via run_ablation's
--ctx-extra loader. Read-only. Usage: python src/build_odds_feat.py
"""
import csv
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "natl_odds_raw.csv"

# BetExplorer spelling -> our DB spelling (only real senior-team mismatches; youth/clubs drop naturally)
ALIAS = {
    "Czech Republic": "Czechia", "USA": "United States", "Turkey": "Türkiye",
    "Ireland": "Republic of Ireland", "D.R. Congo": "Congo DR", "DR Congo": "Congo DR",
    "Trinidad & Tobago": "Trinidad and Tobago", "Saint Kitts and Nevis": "St. Kitts and Nevis",
    "Saint Vincent and the Grenadines": "St. Vincent and the Grenadines",
    "Korea Republic": "South Korea", "Korea DPR": "North Korea", "South Korea": "South Korea",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina", "Bosnia & Herzegovina": "Bosnia-Herzegovina",
    "Cape Verde Islands": "Cape Verde", "Curacao": "Curaçao", "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran", "Côte d'Ivoire": "Ivory Coast", "Ivory Coast": "Ivory Coast",
}


def _nn(name):
    return db.norm(ALIAS.get(name, name))


def shin_devig(oh, od, oa):
    """Shin (1993) margin removal for 3 outcomes -> (pH,pD,pA), overround O. Falls back to
    proportional normalization if the solver misbehaves."""
    q = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    O = q.sum()
    if O <= 1.0:
        p = q / O
        return p, O

    def summ(z):
        return np.sum((np.sqrt(z * z + 4 * (1 - z) * q * q / O) - z) / (2 * (1 - z)))

    lo, hi = 1e-9, 0.4
    if summ(lo) < 1 or summ(hi) > 1:          # not bracketed -> proportional
        p = q / O
        return p, O
    for _ in range(80):
        z = 0.5 * (lo + hi)
        if summ(z) > 1:
            lo = z
        else:
            hi = z
    z = 0.5 * (lo + hi)
    p = (np.sqrt(z * z + 4 * (1 - z) * q * q / O) - z) / (2 * (1 - z))
    p = p / p.sum()
    return p, O


def main():
    con = db.connect(); con.row_factory = None; c = con.cursor()
    cid2name = {r[0]: r[1] for r in c.execute("SELECT club_id,name FROM club")}
    # CLUB odds straight from the DB (football-data.co.uk columns; prefer market-average, else B365).
    # This is where the model LEARNS to use odds (~38k examples); scraped national odds transfer onto it.
    club_odds = {}
    for mid, b3h, b3d, b3a, avh, avd, ava in c.execute(
        "SELECT match_id, b365h, b365d, b365a, avgh, avgd, avga FROM match "
        "WHERE competition_id NOT IN (9,10,11,12,13,14,15) AND home_goals IS NOT NULL "
        "AND (avgh IS NOT NULL OR b365h IS NOT NULL)"):
        oh, od, oa = (avh, avd, ava) if avh else (b3h, b3d, b3a)
        if oh and od and oa and oh > 1 and od > 1 and oa > 1:
            club_odds[mid] = (float(oh), float(od), float(oa))
    # DB national matches indexed by (isodate, normhome, normaway) -> match_id (both orientations)
    idx = {}
    natl_mids = set()
    for mid, d, hc, ac in c.execute(
        "SELECT match_id, match_date, home_club_id, away_club_id FROM match "
        "WHERE competition_id IN (9,10,11,12,13,14,15) AND home_goals IS NOT NULL"):
        natl_mids.add(mid)
        iso = d[:10]
        nh, na = db.norm(cid2name.get(hc, "")), db.norm(cid2name.get(ac, ""))
        idx[(iso, nh, na)] = (mid, False)   # False = same orientation
        idx[(iso, na, nh)] = (mid, True)    # True  = swapped (swap pH/pA)
    con.close()
    # training mids (align the bundle to these; missing ones auto-zero in the loader)
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    train_mids = set(int(m) for m in z["mids"])

    odds_feat, stage_feat = {}, {}
    # seed the odds features with DB club odds (training-set matches only; Shin de-vig same as scraped)
    n_club = 0
    for mid, (oh, od, oa) in club_odds.items():
        if mid in train_mids:
            p, O = shin_devig(oh, od, oa)
            odds_feat[mid] = [p[0], p[1], p[2], np.log(O), 1.0]
            n_club += 1
    print(f"  club odds from DB: {n_club} training matches", flush=True)
    n_csv = n_join = 0
    unmatched = defaultdict(int)
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        n_csv += 1
        try:
            dd, mm, yy = r["date"].split(".")
            d0 = date(int(yy), int(mm), int(dd))
        except ValueError:
            continue
        nh, na = _nn(r["home"]), _nn(r["away"])
        hit = None
        for dz in (0, -1, 1):
            iso = (d0 + timedelta(days=dz)).isoformat()
            if (iso, nh, na) in idx:
                hit = idx[(iso, nh, na)]; break
        if not hit:
            unmatched[r["home"]] += 1; unmatched[r["away"]] += 1
            continue
        mid, swapped = hit
        if mid not in train_mids:
            continue                                 # matched a real match but not in training set
        p, O = shin_devig(float(r["odd_h"]), float(r["odd_d"]), float(r["odd_a"]))
        pH, pD, pA = (p[2], p[1], p[0]) if swapped else (p[0], p[1], p[2])
        odds_feat[mid] = [pH, pD, pA, np.log(O), 1.0]
        stage_feat[mid] = [float(r["knockout"]), 1.0]
        n_join += 1

    def save(name, feat, cols):
        mids = np.array(sorted(feat), dtype=np.int64)
        arr = np.array([feat[int(m)] for m in mids], np.float32)
        out = ROOT / "data" / name
        np.savez_compressed(out, mids=mids, feats=arr)
        print(f"  wrote {out.name}: {len(mids)} matches x {arr.shape[1]} feats {cols}", flush=True)

    save("ctx_odds.npz", odds_feat, "[pH,pD,pA,ln_overround,has_odds]")
    save("ctx_stage.npz", stage_feat, "[knockout,has_stage]")
    nat_join = sum(1 for m in odds_feat if m in natl_mids)
    nat_train = len(train_mids & natl_mids)
    print(f"\ncsv rows={n_csv}  total odds feats={len(odds_feat)} "
          f"(club {len(odds_feat)-nat_join} + natl {nat_join}/{nat_train} = {nat_join/nat_train*100:.1f}% natl coverage)",
          flush=True)
    ko = sum(v[0] for v in stage_feat.values())
    print(f"knockout matches among joined: {int(ko)}", flush=True)
    print("\ntop unmatched CSV names (aliases/non-DB teams):", flush=True)
    for nm, ct in sorted(unmatched.items(), key=lambda kv: -kv[1])[:20]:
        print(f"    {ct:4d}  {nm}", flush=True)


if __name__ == "__main__":
    main()
