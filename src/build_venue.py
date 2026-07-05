"""Step 1 of the per-team home-advantage A/B: per-match venue / true-home features aligned to players_imp mids.
A team's home stadium is its MODAL home venue (98% of clubs play 98% of home games at one venue, so per-team
≈ per-stadium). Per match: [true_home (home team at its modal venue), neutral (venue is neither team's modal),
venue_known] + home_team_idx (dense id for the per-team embedding in Step 3). Also writes the modal-venue map +
club->idx so Step 4 can map WC fixtures. Usage: python src/build_venue.py
"""
import sys, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import db
ROOT = Path(__file__).resolve().parent.parent
NATc = {9, 10, 11, 12, 13, 14, 15}


def main():
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    mids = [int(m) for m in z["mids"]]
    con = db.connect()
    minfo = {r[0]: (r[1], r[2], r[3], r[4]) for r in
             con.execute("SELECT match_id,home_club_id,away_club_id,venue,competition_id FROM match")}
    # modal home venue per club (>=3 home games with a venue) — this is the club's 'true home'
    hv = defaultdict(Counter)
    for hc, v in con.execute("SELECT home_club_id,venue FROM match WHERE venue IS NOT NULL AND venue!='' AND home_goals IS NOT NULL"):
        hv[hc][v] += 1
    modal = {hc: c.most_common(1)[0][0] for hc, c in hv.items() if sum(c.values()) >= 3}
    con.close()

    club2idx = {}
    def idx(c):
        if c not in club2idx:
            club2idx[c] = len(club2idx)
        return club2idx[c]

    feats, home_idx, natl = [], [], []
    for m in mids:
        hc, ac, v, comp = minfo.get(m, (None, None, None, None))
        vk = 1.0 if v not in (None, "") else 0.0
        th = 1.0 if (vk and modal.get(hc) == v) else 0.0
        neu = 1.0 if (vk and modal.get(hc) != v and modal.get(ac) != v) else 0.0
        feats.append([th, neu, vk]); home_idx.append(idx(hc) if hc is not None else 0)
        natl.append(comp in NATc)
    feats = np.array(feats, np.float32); home_idx = np.array(home_idx, np.int64); natl = np.array(natl, bool)
    np.savez_compressed(ROOT / "data" / "venue.npz", mids=np.array(mids, np.int64),
                        feats=feats, home_idx=home_idx, n_teams=len(club2idx))
    json.dump({"club2idx": {str(k): v for k, v in club2idx.items()},
               "modal": {str(k): v for k, v in modal.items()}},
              open(ROOT / "data" / "venue_map.json", "w", encoding="utf-8"), ensure_ascii=False)

    def share(msk):
        f = feats[msk]; n = len(f)
        return (f[:, 0].mean(), f[:, 1].mean(), 1 - f[:, 2].mean(), n)   # true_home, neutral, unknown
    a = share(np.ones(len(feats), bool)); b = share(natl)
    print(f"saved data/venue.npz: {len(mids)} matches, {len(club2idx)} home teams", flush=True)
    print(f"  ALL : true_home {a[0]*100:.0f}%  neutral {a[1]*100:.0f}%  venue-unknown {a[2]*100:.0f}%  (n={a[3]})", flush=True)
    print(f"  NATL: true_home {b[0]*100:.0f}%  neutral {b[1]*100:.0f}%  venue-unknown {b[2]*100:.0f}%  (n={b[3]})", flush=True)
    assert feats.shape[0] == len(mids) and home_idx.shape[0] == len(mids), "misaligned to players_imp mids"
    print("  aligned to players_imp mids OK", flush=True)


if __name__ == "__main__":
    main()
