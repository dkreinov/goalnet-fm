"""Per-player position-ordered dataset for the 11v11 model: for every match where BOTH starting XIs
are fully FM-graded, extract each starter's FM attribute vector + role bucket (GK/DEF/MID/ATT), ordered
GK -> DEF -> MID -> ATT per team. Reuses build_dataset's exact starter->snapshot resolution so the set
matches the 11v11-graded definition. Read-only; writes data/players.npz.

Output arrays (M matches kept = both sides exactly 11 graded starters):
  Xh, Xa : (M, 11, A) float32  attribute vectors, players ordered by role
  Rh, Ra : (M, 11)    int8     role id per player (0 GK, 1 DEF, 2 MID, 3 ATT)
  y      : (M,)        int8     result 0=Home 1=Draw 2=Away
  dates  : (M,)        datetime64
  mids   : (M,)        int64    match_id
  attrs  : (A,)        str      attribute names (the feature order)
Usage: python D:/Programming/claude/FM/src/build_player_dataset.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd

ROLE = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}


def main():
    con = db.connect()
    snaps = bd.load_snapshots(con)
    attrs = bd.load_attrs(con)                      # snapshot_id -> {(cat,name): val}
    idx, has_snap = bd.name_index(con)
    bd.build_fallback(idx, con)
    xwalk, collisions = bd.load_xwalk(con)
    eclub_to_g, gpid_clubs = bd.load_espn_bridge(con, xwalk)
    roster, ruid_pids = bd.load_roster(con)
    sfmv = bd.season_fmv(con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}

    # canonical attribute order: union of all attr_names, stable sorted (drop hidden? keep all numeric)
    attr_set = set()
    for d in attrs.values():
        for (cat, name) in d:
            attr_set.add(name)
    ATTRS = sorted(attr_set)
    aidx = {n: i for i, n in enumerate(ATTRS)}
    A = len(ATTRS)
    print(f"{A} attributes, {len(snaps):,} snapshot-players", flush=True)

    def snap_for(mid, pid, cid, season):
        """Replicate build_dataset resolution -> snapshot_id (or None)."""
        target_fmv = sfmv.get(season); season_end = bd.SEASON_END.get(season, "2026-06-30")
        g = xwalk.get(pid)
        if g:
            u = []
            for p in g[0]:
                u.extend(snaps.get(p, []))
            u.sort()
            sn = bd.pick_snapshot(u, target_fmv, season_end)
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
                u.sort()
                sn = bd.pick_snapshot(u, target_fmv, season_end)
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
            u.sort()
            sn = bd.pick_snapshot(u, target_fmv, season_end)
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

    Xh, Xa, Rh, Ra, y, dates, mids = [], [], [], [], [], [], []
    kept = 0
    for k, (mid, date, hc, ac, hg, ag, season) in enumerate(matches):
        sides = {}
        ok = True
        for cid in (hc, ac):
            xi = lineups.get((mid, cid), [])
            players = []
            for pid, pos in xi:
                sid = snap_for(mid, pid, cid, season)
                if sid is None:
                    continue
                players.append((ROLE.get(bd.POS_GROUP.get((pos or " ")[0], "MID"), 2), vec(sid)))
            if len(players) < 11:
                ok = False; break
            players.sort(key=lambda t: t[0])          # order GK->DEF->MID->ATT
            players = players[:11]
            sides[cid] = players
        if not ok:
            continue
        Xh.append(np.stack([v for _, v in sides[hc]]))
        Rh.append(np.array([r for r, _ in sides[hc]], dtype=np.int8))
        Xa.append(np.stack([v for _, v in sides[ac]]))
        Ra.append(np.array([r for r, _ in sides[ac]], dtype=np.int8))
        y.append(0 if hg > ag else (1 if hg == ag else 2))
        dates.append(np.datetime64(date[:10]))
        mids.append(mid)
        kept += 1
        if (k + 1) % 10000 == 0:
            print(f"  {k+1:,}/{len(matches):,} scanned, kept={kept:,}", flush=True)

    out = db.ROOT / "data" / "players.npz"
    np.savez_compressed(
        out, Xh=np.stack(Xh).astype(np.float32), Xa=np.stack(Xa).astype(np.float32),
        Rh=np.stack(Rh), Ra=np.stack(Ra), y=np.array(y, dtype=np.int8),
        dates=np.array(dates), mids=np.array(mids, dtype=np.int64), attrs=np.array(ATTRS))
    print(f"\nsaved {out}: {kept:,} full-11v11 matches, X shape {np.stack(Xh).shape}")
    con.close()


if __name__ == "__main__":
    main()
