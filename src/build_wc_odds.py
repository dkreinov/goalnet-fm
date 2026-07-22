"""Join scraped OddsPortal WC2026 odds to the 104-game replay slate; Shin de-vig; emit wc_odds.npz.

Reads data/oddsportal_raw.csv (from scrape_oddsportal.py, 2026-window rows) and aligns each match to
a wc_inputs.npz slate key (CODE-CODE, e.g. MEX-RSA) by (timestamp, team codes, both orientations).
De-vigs the closing 1X2 with Shin's method (identical to build_odds_feat.py) and writes, keyed to the
slate order so the replay's --wc-extra loader can consume it:
  data/wc_odds.npz  {keys:[CODE-CODE...], feats:[pH,pD,pA,ln_overround,has_odds]}  (home orientation)

The odds array is oriented to each CSV row's home/away; when the slate key's home differs, pH/pA swap.
Read-only w.r.t. the DB. Usage: python src/build_wc_odds.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db
from build_odds_feat import shin_devig                       # reuse the frozen Phase-4 de-vig

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "oddsportal_raw.csv"
WC_INPUTS = ROOT / "experiments" / "ablation" / "wc_inputs.npz"
TEAMS = Path(r"D:\Programming\claude\worldcup\team_db\teams")
OUT = ROOT / "data" / "wc_odds.npz"

# OddsPortal senior-team spelling -> worldcup team_db name (only mismatches; verified against the 104).
NAME_FIX = {
    "USA": "United States", "South Korea": "Korea Republic", "North Korea": "Korea DPR",
    "Ivory Coast": "Cote d'Ivoire", "Czech Republic": "Czechia", "Turkey": "Türkiye",
    "Republic of Ireland": "Ireland", "Cape Verde": "Cabo Verde", "DR Congo": "Congo DR",
    "D.R. Congo": "Congo DR", "Bosnia & Herzegovina": "Bosnia and Herzegovina", "Iran": "IR Iran",
}


def _name2code():
    """worldcup team_db name (normalized) -> 3-letter code."""
    out = {}
    for f in TEAMS.glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]
        out[db.norm(t["name"])] = t["code"]
    return out


def main():
    name2code = _name2code()

    def code_of(op_name):
        fixed = NAME_FIX.get(op_name, op_name)
        return name2code.get(db.norm(fixed))

    # slate keys + kickoffs (align output to slate order)
    w = np.load(WC_INPUTS, allow_pickle=True)
    slate_keys = [str(k) for k in w["keys"]]
    slate_kick = {str(k): int(t) for k, t in zip(w["keys"], w["kickoff"])}
    slate_home = {k: k.split("-")[0] for k in slate_keys}     # slate orientation

    # index CSV 2026-window rows by (timestamp, frozenset(codes)) -> (home_code, pH,pD,pA, lnO)
    by_pair = {}
    n_csv = n_unmapped = 0
    unmapped = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        ts = int(r["timestamp"])
        if not (1780000000 < ts < 1785000000):
            continue
        n_csv += 1
        hc, ac = code_of(r["home"]), code_of(r["away"])
        if not hc or not ac:
            n_unmapped += 1
            for nm in (r["home"], r["away"]):
                if not code_of(nm):
                    unmapped[nm] = unmapped.get(nm, 0) + 1
            continue
        p, O = shin_devig(float(r["odd_h"]), float(r["odd_d"]), float(r["odd_a"]))
        by_pair[frozenset((hc, ac))] = (hc, [p[0], p[1], p[2], float(np.log(O))], ts)

    # emit in slate order, oriented to slate home
    feats, n_join, ts_mismatch = [], 0, 0
    for k in slate_keys:
        pair = frozenset(k.split("-"))
        rec = by_pair.get(pair)
        if rec is None:
            feats.append([0.0, 0.0, 0.0, 0.0, 0.0])           # no odds -> mask 0
            continue
        csv_home, p, ts = rec
        if abs(ts - slate_kick[k]) > 3 * 86400:               # sanity: same fixture, not a rematch
            ts_mismatch += 1
        pH, pD, pA = (p[0], p[1], p[2]) if csv_home == slate_home[k] else (p[2], p[1], p[0])
        feats.append([pH, pD, pA, p[3], 1.0])
        n_join += 1

    arr = np.array(feats, np.float32)
    np.savez_compressed(OUT, keys=np.array(slate_keys), feats=arr)
    cov = n_join / len(slate_keys) * 100
    print(f"csv 2026 rows={n_csv} (unmapped {n_unmapped})", flush=True)
    print(f"wrote {OUT.name}: {len(slate_keys)} slate games, {n_join} with odds "
          f"({cov:.1f}% coverage), {arr.shape[1]} feats [pH,pD,pA,ln_overround,has_odds]", flush=True)
    if ts_mismatch:
        print(f"  WARNING: {ts_mismatch} joined games have >3d kickoff gap (check orientation)", flush=True)
    if unmapped:
        print("  unmapped OddsPortal names:", dict(sorted(unmapped.items(), key=lambda kv: -kv[1])), flush=True)


if __name__ == "__main__":
    main()
