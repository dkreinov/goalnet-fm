"""Clean coverage report: per league x season, how many DISTINCT players that started a
match we have an FM rating for (and the total distinct starters, for context).
Uses the exact build_dataset resolution chain (xwalk -> name-fallback -> roster).
Usage: python D:/Programming/claude/FM/src/report_players.py
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as B
import leagues as L

EXCL = {"China Super League", "Ecuador LigaPro", "India Super League", "Paraguay Primera Division",
        "Peru Liga 1", "South Africa Premiership", "Israel Ligat haAl", "Japan J1 League", "Colombia Primera A"}
SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
RANK = {l["name"]: l["rank"] for l in (L.LEAGUES + L.EXTRA_LEAGUES + L.UEFA_CUPS)}


def main():
    con = db.connect()
    snaps = B.load_snapshots(con)
    idx, has_snap = B.name_index(con)
    B.build_fallback(idx, con)
    xwalk = B.load_xwalk(con)
    roster, ruid_pids = B.load_roster(con)
    sfmv = B.season_fmv(con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}

    matches = con.execute(
        """SELECT m.match_id, se.label, m.home_club_id, m.away_club_id, COALESCE(co.name,'?')
           FROM match m JOIN season se ON se.season_id=m.season_id
           LEFT JOIN competition co ON co.competition_id=m.competition_id""").fetchall()
    lineups = defaultdict(list)
    for mid, pid, cid in con.execute(
            "SELECT match_id, player_id, club_id FROM match_player WHERE started=1"):
        lineups[(mid, cid)].append((pid, cid))

    rated = defaultdict(set); total = defaultdict(set)
    for mid, season, hcid, acid, comp in matches:
        tfmv = sfmv.get(season)
        send = B.SEASON_END.get(season, "2027-01-01")
        for cid in (hcid, acid):
            for pid, _ in lineups.get((mid, cid), []):
                total[(comp, season)].add(pid)
                snap = None
                gp = xwalk.get(pid)
                if gp:
                    union = []
                    for p in gp[0]:
                        union.extend(snaps.get(p, []))
                    union.sort()
                    snap = B.pick_snapshot(union, tfmv, send)
                if snap is None:
                    rpid = B.resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
                    snap = B.pick_snapshot(snaps.get(rpid, []), tfmv, send) if rpid else None
                if snap is None:
                    rl = roster.get((mid, pid))
                    if rl and rl[1] == "high":          # conservative: high-confidence roster only
                        union = []
                        for p in ruid_pids.get(rl[0], ()):
                            union.extend(snaps.get(p, []))
                        union.sort()
                        snap = B.pick_snapshot(union, tfmv, send)
                if snap is not None:
                    rated[(comp, season)].add(pid)

    comps = sorted({c for c, _ in total}, key=lambda c: (RANK.get(c, 999), c))
    print("DISTINCT PLAYERS WITH AN FM RATING per league x season   (rated / total starters)")
    print("=" * 104)
    print(f"{'LEAGUE':30}" + "".join(f"{s[2:]:>11}" for s in SEASONS) + f"{'TOTAL':>13}")
    gt_r = gt_t = 0
    for comp in comps:
        if RANK.get(comp, 999) > 45:
            continue
        cells = []; allr = set(); allt = set()
        for s in SEASONS:
            r = len(rated.get((comp, s), ())); t = len(total.get((comp, s), ()))
            allr |= rated.get((comp, s), set()); allt |= total.get((comp, s), set())
            cells.append("·" if t == 0 else f"{r}/{t}")
        tag = "" if comp not in EXCL else " (excl)"
        print(f"{comp[:30]:30}" + "".join(f"{c:>11}" for c in cells) +
              f"{f'{len(allr)}/{len(allt)}':>13}{tag}")
        if comp not in EXCL:
            gt_r += len(allr); gt_t += len(allt)
    print("=" * 104)
    print(f"{'TARGET-LEAGUE TOTAL (distinct/season-union per league)':54}{'':>44}{gt_r}/{gt_t} rated")
    # true global distinct players with any FM rating across target leagues
    allrated = set()
    for (comp, s), st in rated.items():
        if comp not in EXCL:
            allrated |= st
    print(f"\nUnique players (target leagues, any season) we have an FM rating for: {len(allrated):,}")


if __name__ == "__main__":
    main()
