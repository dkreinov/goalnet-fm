"""Like build_player_dataset.py but appends per-player scalar quality signals that the 62 attrs don't
fully capture: market value and wage (FM/TM), each log1p-scaled with a missing-indicator. These are
snapshot-id keyed (no club-id bridging), so coverage is whatever the snapshot itself carries.
Feature order: [62 attrs] + [log1p(value_eur), has_value, log1p(wage_eur), has_wage] -> A = 66.
Read-only; writes data/players2.npz (same arrays as v1, wider X).
Usage: python D:/Programming/claude/FM/src/build_player_dataset2.py
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
    attrs = bd.load_attrs(con)
    idx, has_snap = bd.name_index(con)
    bd.build_fallback(idx, con)
    xwalk, collisions = bd.load_xwalk(con)
    eclub_to_g, gpid_clubs = bd.load_espn_bridge(con, xwalk)
    roster, ruid_pids = bd.load_roster(con)
    sfmv = bd.season_fmv(con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}
    # sid -> scalar quality signals
    sval = {r[0]: (r[1], r[2]) for r in
            con.execute("SELECT snapshot_id, value_eur, wage_eur FROM player_snapshot")}

    attr_set = set()
    for d in attrs.values():
        for (cat, name) in d:
            attr_set.add(name)
    ATTRS = sorted(attr_set)
    aidx = {n: i for i, n in enumerate(ATTRS)}
    A = len(ATTRS)
    EXTRA = ["log_value", "has_value", "log_wage", "has_wage"]
    W = A + len(EXTRA)
    print(f"{A} attrs + {len(EXTRA)} scalars = {W}, {len(snaps):,} snapshot-players", flush=True)

    def snap_for(mid, pid, cid, season):
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
        v = np.zeros(W, dtype=np.float32)
        for (cat, name), val in attrs.get(sid, {}).items():
            j = aidx.get(name)
            if j is not None:
                v[j] = val
        val_eur, wage_eur = sval.get(sid, (None, None))
        if val_eur:
            v[A] = np.log1p(val_eur); v[A + 1] = 1.0
        if wage_eur:
            v[A + 2] = np.log1p(wage_eur); v[A + 3] = 1.0
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
            players.sort(key=lambda t: t[0])
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

    out = db.ROOT / "data" / "players2.npz"
    np.savez_compressed(
        out, Xh=np.stack(Xh).astype(np.float32), Xa=np.stack(Xa).astype(np.float32),
        Rh=np.stack(Rh), Ra=np.stack(Ra), y=np.array(y, dtype=np.int8),
        dates=np.array(dates), mids=np.array(mids, dtype=np.int64),
        attrs=np.array(ATTRS + EXTRA))
    print(f"\nsaved {out}: {kept:,} matches, X shape {np.stack(Xh).shape}")
    con.close()


if __name__ == "__main__":
    main()
