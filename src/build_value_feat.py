"""Shared club-value context feature (for baking market value into GoalNet). One source of truth used by both
train_goals (--value, via data/value.npz) and predict_game (WC games, via name_value_map). Per match:
[mean home-XI club value, mean away-XI club value, log-diff, home coverage, away coverage], where a player's
value = their club's squad value (club match: match_player.club_id + season; national/fallback: player's real
club from player_snapshot, latest squad value). Usage: python src/build_value_feat.py  (writes data/value.npz)
"""
import sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import db

ROOT = Path(__file__).resolve().parent.parent


def _lookups(con):
    seasonlab = {r[0]: r[1] for r in con.execute("SELECT season_id,label FROM season")}
    cval, cval_any = {}, {}
    for cid, season, sv in con.execute("SELECT club_id,season,squad_value_eur FROM club_season_tm ORDER BY season"):
        if sv:
            cval[(cid, season)] = sv; cval_any[cid] = sv     # cval_any keeps the LATEST season's value
    pclub = {}
    for pid, cid in con.execute("SELECT player_id,club_id FROM player_snapshot WHERE club_id IS NOT NULL ORDER BY snapshot_date"):
        pclub[pid] = cid
    return seasonlab, cval, cval_any, pclub


def _pval(pid, teamcid, lab, cval, cval_any, pclub):
    v = cval.get((teamcid, lab)) or cval_any.get(teamcid)     # club match: the club itself
    if v:
        return v
    rc = pclub.get(pid)                                       # national/fallback: player's real club (latest)
    return (cval.get((rc, lab)) or cval_any.get(rc)) if rc is not None else None


def _feat(home_vals, away_vals, nh, na):
    lv = lambda x: math.log1p(x) if x else 0.0
    hvv = [x for x in home_vals if x]; avv = [x for x in away_vals if x]
    hm = lv(np.mean(hvv)) if hvv else 0.0; am = lv(np.mean(avv)) if avv else 0.0
    return [hm, am, (hm - am) if (hvv and avv) else 0.0, len(hvv) / max(nh, 1), len(avv) / max(na, 1)]


def build_value(mids, con=None):
    """(len(mids), 5) value features aligned to mids."""
    own = con is None; con = con or db.connect()
    seasonlab, cval, cval_any, pclub = _lookups(con)
    minfo = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT match_id,season_id,home_club_id,away_club_id FROM match")}
    starters = defaultdict(list)
    for mid, pid, cid in con.execute("SELECT match_id,player_id,club_id FROM match_player WHERE started=1"):
        starters[mid].append((pid, cid))
    F = []
    for m in mids:
        si, hc, ac = minfo.get(m, (None, None, None)); lab = seasonlab.get(si)
        hv, av = [], []
        for pid, cid in starters.get(m, []):
            (hv if cid == hc else av).append(_pval(pid, cid, lab, cval, cval_any, pclub))
        F.append(_feat(hv, av, sum(cid == hc for _, cid in starters.get(m, [])),
                       sum(cid == ac for _, cid in starters.get(m, []))))
    if own:
        con.close()
    return np.array(F, np.float32)


def name_value_map(con):
    """norm_name -> latest club squad value (for predict-time WC value: XI player names -> club value)."""
    _, _, cval_any, pclub = _lookups(con)
    out = {}
    for pid, nn in con.execute("SELECT player_id,norm_name FROM player"):
        rc = pclub.get(pid); v = cval_any.get(rc) if rc is not None else None
        if v:
            out[nn] = v
    return out


def xi_value_feat(home_names, away_names, nvmap):
    """5-vec value feature for a predicted game from XI player norm-names (WC/national path)."""
    hv = [nvmap.get(n) for n in home_names]; av = [nvmap.get(n) for n in away_names]
    return np.array(_feat(hv, av, len(home_names), len(away_names)), np.float32)


def main():
    z = np.load(ROOT / "data" / "players_imp.npz", allow_pickle=True)
    mids = [int(m) for m in z["mids"]]
    F = build_value(mids)
    np.savez_compressed(ROOT / "data" / "value.npz", mids=np.array(mids, np.int64), val=F)
    cov = float(((F[:, 3] > 0.5) & (F[:, 4] > 0.5)).mean())
    print(f"saved data/value.npz: {len(mids)} matches, both-sides value coverage {cov*100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
