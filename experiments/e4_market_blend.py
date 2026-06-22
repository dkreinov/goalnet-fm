"""E4 — market blend for WC games. The bookmaker is the strongest single WC predictor and (unlike club games)
its odds aren't in the DB, so we use the separately-collected data/wc_odds.csv. De-vig the 1X2 odds -> implied
P(H/D/A); fit market Poisson rates (lh,la) whose double-Poisson 1X2 matches the implied probs; build a
market score grid; blend P=(1-w)*model + w*market and sweep w. A/B the EV-pick on the played WC games.
NOTE: weight is swept in-sample on the played WC slate (no held-out slice has odds) -> read the SHAPE (does
any market info help, and how much), not a single tuned number.
Usage: python D:/Programming/claude/FM/experiments/e4_market_blend.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_harness as H
import train_goals as tg, db
WC = Path(r"D:\Programming\claude\worldcup\team_db")
SLUG2NAME = {"usa": "United States", "d-r-congo": "Congo DR", "ivory-coast": "Côte d'Ivoire",
             "cape-verde": "Cabo Verde", "south-korea": "Korea Republic", "czech-republic": "Czechia",
             "bosnia-herzegovina": "Bosnia and Herzegovina", "saudi-arabia": "Saudi Arabia",
             "new-zealand": "New Zealand", "south-africa": "South Africa", "turkey": "Türkiye", "iran": "IR Iran"}


def norm2code():
    import json
    m = {}
    for f in (WC / "teams").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))["team"]; m[db.norm(t["name"])] = t["code"]
    return m


def code_of(slug, n2c):
    nm = SLUG2NAME.get(slug)
    if nm: return n2c.get(db.norm(nm))
    return n2c.get(db.norm(slug.replace("-", " ")))


def fit_market_rates(pH, pD, pA, rho=0.0):
    """Grid-search lh,la so the double-Poisson 1X2 best matches implied (pH,pD,pA)."""
    best, br = 1e9, (1.3, 1.1)
    for lh in np.arange(0.2, 3.6, 0.1):
        for la in np.arange(0.2, 3.6, 0.1):
            ho = tg.hda_from_P(tg.score_matrix(lh, la, rho))
            e = (ho[0]-pH)**2 + (ho[1]-pD)**2 + (ho[2]-pA)**2
            if e < best: best, br = e, (lh, la)
    return br


def main():
    c = H.load_cache(); w = c["wc"]; n2c = norm2code()
    keys = list(w["keys"])
    # parse odds: line = "slug-a-slug-b|hs|as|oH|oD|oA"; slug pair may contain hyphens, so try every split
    rows = []
    for line in open("data/wc_odds.csv", encoding="utf-8"):
        line = line.strip()
        if not line: continue
        f = line.split("|")
        slugpair, hs, as_, oH, oD, oA = f[0], int(f[1]), int(f[2]), float(f[3]), float(f[4]), float(f[5])
        rows.append((slugpair, oH, oD, oA))
    # slugpair like "tunisia-japan" or "ivory-coast..." — try every split point, accept where both map to codes
    market = {}
    for slugpair, oH, oD, oA in rows:
        toks = slugpair.split("-"); ca = cb = None
        for k in range(1, len(toks)):
            a = "-".join(toks[:k]); b = "-".join(toks[k:])
            if code_of(a, n2c) and code_of(b, n2c): ca, cb = code_of(a, n2c), code_of(b, n2c); break
        if not ca: continue
        s = 1/oH + 1/oD + 1/oA; pH, pD, pA = (1/oH)/s, (1/oD)/s, (1/oA)/s
        market[(ca, cb)] = (pH, pD, pA)
    print(f"odds parsed: {len(market)} games; cache games: {len(keys)}", flush=True)
    # build matched lists oriented to the cache key (hc-ac)
    midx, mgrid_rates = [], {}
    matched = 0
    for i, key in enumerate(keys):
        hc, ac = key.split("-")
        if (hc, ac) in market: pH, pD, pA = market[(hc, ac)]
        elif (ac, hc) in market: pA, pD, pH = market[(ac, hc)]   # flip orientation
        else: continue
        lh, la = fit_market_rates(pH, pD, pA)
        mgrid_rates[i] = (lh, la); midx.append(i); matched += 1
    print(f"matched {matched} WC games with odds", flush=True)
    acts = {i: (int(w["hs"][i]), int(w["as_"][i])) for i in range(len(keys))}

    def slate(weight):
        tot = ex = 0
        for i in midx:
            model = H.ens_grid(w["lh"][i], w["la"][i], 0.0)
            lh, la = mgrid_rates[i]; mk = tg.score_matrix(lh, la, 0.0)
            P = (1 - weight) * model + weight * mk; P = P / P.sum()
            pk = tg.ev_pick(P); pts, lab = tg.grade(pk, *acts[i]); tot += pts; ex += lab == "exact"
        return tot, ex
    # also pure-market and model-only baselines on the matched subset
    print(f"=== market blend on {matched} matched WC games (in-sample weight sweep) ===", flush=True)
    for weight in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        tot, ex = slate(weight)
        tag = "model" if weight == 0 else ("market" if weight == 1 else "blend")
        print(f"  w={weight:.1f} {tag:6s} pts={tot}/{matched} exact={ex}", flush=True)


if __name__ == "__main__":
    main()
