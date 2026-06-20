"""Enriched dataset for the position-embedding A/B: like build_player_dataset_imp.py (68k, <=1 imputed
/side) but stores a DETAILED position id per starter (9 buckets: GK/CB/FB/DM/CM/WM/AM/W/ST) instead of
only the 4-way role. The model can still derive the 4-role via DET2ROLE for the baseline arm.
Writes data/players_pos.npz (Rh/Ra now hold detailed pos ids 0..8).
Usage: python D:/Programming/claude/FM/src/build_player_dataset_pos.py [--max-imp 1]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd

# detailed position id (also see DET2ROLE for collapsing to GK/DEF/MID/ATT)
DET = {"GK": 0, "CB": 1, "FB": 2, "DM": 3, "CM": 4, "WM": 5, "AM": 6, "W": 7, "ST": 8}
DET2ROLE = np.array([0, 1, 1, 2, 2, 2, 2, 3, 3], np.int64)   # 9 detailed -> 4 role


def pos_detail(pos):
    p = (pos or "").upper().strip()
    if not p or p[0] == "G":
        return DET["GK"]
    if p.startswith("DM"):
        return DET["DM"]
    if p.startswith("AM"):
        return DET["AM"]            # AM, AM-L, AM-R = attacking mid
    if p in ("LB", "RB", "WB", "WBL", "WBR", "DL", "DR"):
        return DET["FB"]
    if p in ("LM", "RM", "ML", "MR"):
        return DET["WM"]            # wide midfielders
    if p in ("LW", "RW", "LF", "RF"):
        return DET["W"]             # wide forwards / wingers
    if p[0] == "F" or p.startswith("CF") or p.startswith("ST") or p in ("S", "RCF", "LCF"):
        return DET["ST"]            # central forwards / strikers
    if p.startswith("CD") or p in ("D", "SW"):
        return DET["CB"]            # CD, CD-L, CD-R, D, sweeper = centre back
    if p[0] == "D":
        return DET["FB"] if ("L" in p or "R" in p) else DET["CB"]
    if p[0] == "M" or p.startswith("CM"):
        return DET["CM"]
    return DET["CM"]


def main():
    MAX_IMP = int(sys.argv[sys.argv.index("--max-imp") + 1]) if "--max-imp" in sys.argv else 1
    con = db.connect()
    snaps = bd.load_snapshots(con); attrs = bd.load_attrs(con)
    idx, has_snap = bd.name_index(con); bd.build_fallback(idx, con)
    xwalk, collisions = bd.load_xwalk(con)
    eclub_to_g, gpid_clubs = bd.load_espn_bridge(con, xwalk)
    roster, ruid_pids = bd.load_roster(con); sfmv = bd.season_fmv(con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}
    attr_set = set()
    for d in attrs.values():
        for (cat, name) in d:
            attr_set.add(name)
    ATTRS = sorted(attr_set); aidx = {n: i for i, n in enumerate(ATTRS)}; A = len(ATTRS)
    print(f"{A} attrs, {len(DET)} detailed positions, max_imp={MAX_IMP}", flush=True)

    def snap_for(mid, pid, cid, season):
        target_fmv = sfmv.get(season); season_end = bd.SEASON_END.get(season, "2026-06-30")
        gp = xwalk.get(pid)
        if gp:
            u = []
            for p in gp[0]:
                u.extend(snaps.get(p, []))
            u.sort(); sn = bd.pick_snapshot(u, target_fmv, season_end)
            if sn:
                return sn[1]
        if pid in collisions:
            bridged = eclub_to_g.get(cid, set())
            cand = [gps for uu, gps in collisions[pid].items()
                    if bridged and any(gpid_clubs.get(g2, set()) & bridged for g2 in gps)]
            if len(cand) == 1:
                u = []
                for p in cand[0]:
                    u.extend(snaps.get(p, []))
                u.sort(); sn = bd.pick_snapshot(u, target_fmv, season_end)
                if sn:
                    return sn[1]
        r = bd.resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
        if r:
            sn = bd.pick_snapshot(snaps.get(r, []), target_fmv, season_end)
            if sn:
                return sn[1]
        rl = roster.get((mid, pid))
        if rl and rl[1] == "high":
            u = []
            for p in ruid_pids.get(rl[0], ()):
                u.extend(snaps.get(p, []))
            u.sort(); sn = bd.pick_snapshot(u, target_fmv, season_end)
            if sn:
                return sn[1]
        return None

    def vec(sid):
        v = np.zeros(A, dtype=np.float32)
        for (cat, name), val in attrs.get(sid, {}).items():
            j = aidx.get(name)
            if j is not None:
                v[j] = val
        return v

    matches = con.execute(
        """SELECT m.match_id, m.match_date, m.home_club_id, m.away_club_id, m.home_goals, m.away_goals, s.label
           FROM match m JOIN season s ON s.season_id=m.season_id
           WHERE m.home_goals IS NOT NULL ORDER BY m.match_date""").fetchall()
    lineups = defaultdict(list)
    for mid, pid, cid, pos in con.execute(
            "SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        lineups[(mid, cid)].append((pid, pos))

    recs = []; rsum = {r: np.zeros(A, np.float64) for r in range(9)}; rcnt = {r: 0 for r in range(9)}
    kept = 0
    for k, (mid, date, hc, ac, hg, ag, season) in enumerate(matches):
        sides = {}; ok = True
        for cid in (hc, ac):
            xi = lineups.get((mid, cid), [])
            if len(xi) < 11:
                ok = False; break
            players = []
            for pid, pos in xi:
                d = pos_detail(pos)
                sid = snap_for(mid, pid, cid, season)
                players.append((d, vec(sid) if sid is not None else None))
            graded = sum(1 for _, v in players if v is not None)
            if graded < 11 - MAX_IMP:
                ok = False; break
            players.sort(key=lambda t: (t[1] is None, t[0]))
            players = players[:11]
            if sum(1 for _, v in players if v is None) > MAX_IMP:
                ok = False; break
            players.sort(key=lambda t: t[0])
            sides[cid] = players
            for d, v in players:
                if v is not None:
                    rsum[d] += v; rcnt[d] += 1
        if not ok:
            continue
        recs.append((sides[hc], sides[ac], 0 if hg > ag else (1 if hg == ag else 2),
                     np.datetime64(date[:10]), mid)); kept += 1
        if (k + 1) % 10000 == 0:
            print(f"  {k+1:,}/{len(matches):,} scanned, kept={kept:,}", flush=True)

    role_mean = {r: (rsum[r] / rcnt[r]).astype(np.float32) if rcnt[r] else np.zeros(A, np.float32) for r in range(9)}

    def fill(side):
        X = np.stack([(v if v is not None else role_mean[d]) for d, v in side])
        R = np.array([d for d, _ in side], np.int8)
        return X, R

    Xh, Xa, Rh, Ra, y, dates, mids = [], [], [], [], [], [], []
    for sh, sa, yy, dt, mid in recs:
        Xh_, Rh_ = fill(sh); Xa_, Ra_ = fill(sa)
        Xh.append(Xh_); Rh.append(Rh_); Xa.append(Xa_); Ra.append(Ra_)
        y.append(yy); dates.append(dt); mids.append(mid)

    out = db.ROOT / "data" / "players_pos.npz"
    np.savez_compressed(out, Xh=np.stack(Xh).astype(np.float32), Xa=np.stack(Xa).astype(np.float32),
                        Rh=np.stack(Rh), Ra=np.stack(Ra), y=np.array(y, np.int8),
                        dates=np.array(dates), mids=np.array(mids, np.int64), attrs=np.array(ATTRS))
    print(f"\nsaved {out}: {kept:,} matches, detailed positions. dist={np.bincount(np.concatenate([r for r in Rh]))}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
