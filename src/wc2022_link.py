"""Link the ~21 unresolved WC2022 national-team starters to their FM grade by picking, among same-name
grade candidates, the ELITE one (highest FM23 CA, with its club shown for sanity-check). WC starters are
top players, so the high-CA same-name candidate is almost always correct; club is printed to verify.
Proposes player_xwalk(espn_player_id -> fm_uid, 'high','wc2022_natl') rows; --apply inserts them.
Then rebuild the dataset and run test_xwalk. Usage: python src/wc2022_link.py [--apply]
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import build_dataset as bd

FM23 = 6
GRADE_SRC = ("fminside", "kaggle", "futek")


def main():
    apply = "--apply" in sys.argv
    con = db.connect()
    snaps = bd.load_snapshots(con)
    idx, has_snap = bd.name_index(con); bd.build_fallback(idx, con)
    xwalk, collisions = bd.load_xwalk(con)
    eclub_to_g, gpid_clubs = bd.load_espn_bridge(con, xwalk)
    roster, ruid_pids = bd.load_roster(con); sfmv = bd.season_fmv(con)
    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}
    cname = {r[0]: r[1] for r in con.execute("SELECT club_id, name FROM club")}
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    grade_sids = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN (?,?,?)", GRADE_SRC)]

    def resolves(mid, pid, cid, season):
        target_fmv = sfmv.get(season); season_end = bd.SEASON_END.get(season, "2026-06-30")
        g = xwalk.get(pid)
        if g:
            u = []
            for p in g[0]:
                u.extend(snaps.get(p, []))
            u.sort()
            if bd.pick_snapshot(u, target_fmv, season_end):
                return True
        if pid in collisions:
            bridged = eclub_to_g.get(cid, set())
            if sum(1 for uu, gps in collisions[pid].items()
                   if bridged and any(gpid_clubs.get(g2, set()) & bridged for g2 in gps)) == 1:
                return True
        r = bd.resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
        if r and bd.pick_snapshot(snaps.get(r, []), target_fmv, season_end):
            return True
        rl = roster.get((mid, pid))
        return bool(rl and rl[1] == "high")

    # unresolved WC2022 starters
    matches = con.execute("""SELECT m.match_id, m.home_club_id, m.away_club_id, s.label FROM match m
                             JOIN season s ON s.season_id=m.season_id WHERE m.competition_id=9""").fetchall()
    lineups = defaultdict(list)
    mids = [m[0] for m in matches]
    for mid, pid, cid, pos in con.execute(
            "SELECT match_id,player_id,club_id,position FROM match_player WHERE started=1 AND match_id IN (%s)"
            % ",".join("?" * len(mids)), mids):
        lineups[(mid, cid)].append((pid, pos))
    unresolved = {}
    for mid, hc, ac, season in matches:
        for cid in (hc, ac):
            for pid, pos in lineups.get((mid, cid), []):
                if not resolves(mid, pid, cid, season):
                    unresolved[pid] = (cid, season)

    # candidate grade players per norm_name: uid -> best CA (prefer FM23) + clubs
    name_cands = defaultdict(dict)     # norm_name -> {uid: (best_ca, fm23_ca, clubs)}
    uid_pid = {}
    for sid in grade_sids:
        for uid, gpid in con.execute(
                "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pid[uid] = gpid
    # snapshot CA/club per grade pid
    pid_snaps = defaultdict(list)
    for gpid, fmv, ca, cid in con.execute("SELECT player_id, fm_version_id, ca, club_id FROM player_snapshot"):
        pid_snaps[gpid].append((fmv, ca or 0, cid))
    nm_uids = defaultdict(list)
    for uid, gpid in uid_pid.items():
        nm = pname.get(gpid)
        if nm:
            nm_uids[nm].append((uid, gpid))

    # ESPN ids per internal pid
    pid_eids = defaultdict(list)
    for eid, pid in con.execute(
            "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (espn_sid,)):
        pid_eids[pid].append(eid)

    # club hint (normalized substring) for ambiguous mononyms / common names — the WC starter's 2022 club
    CLUB_HINT = {"marquinhos": ("paris", "psg"), "pepe": ("porto",), "vitinha": ("paris", "psg"),
                 "aaron ramsey": ("nice",), "luis suarez": ("nacional",), "antony": ("ajax",),
                 "fred": ("man u",)}

    print(f"unresolved WC2022 starters: {len(unresolved)}\n", flush=True)
    links = []
    for pid, (cid, season) in sorted(unresolved.items(), key=lambda x: pname.get(x[0], "")):
        nm = pname.get(pid, "?")
        hint = CLUB_HINT.get(nm)
        best = None
        for uid, gpid in nm_uids.get(nm, []):
            ss = pid_snaps.get(gpid, [])
            fm23_snaps = [(ca, c) for fmv, ca, c in ss if fmv == FM23]
            if not fm23_snaps:                  # SAFETY: must have a snapshot in the WC season (FM23)
                continue
            fm23ca = max((ca for ca, _ in fm23_snaps), default=0)
            fm23_clubs = {cname.get(c, "?") for _, c in fm23_snaps if c}
            bestca = max((ca for _, ca, _ in ss), default=0)
            hit = 1 if (hint and any(h in db.norm(c) for c in fm23_clubs for h in hint)) else 0
            key = (hit, fm23ca, bestca)         # prefer FM23-club-hint match, then ability
            if best is None or key > best[0]:
                best = (key, uid, fm23_clubs)
        if not best:
            print(f"  {nm:26s} [{cname.get(cid,'?')}] -> no FM23 same-name grade (uncovered) — SKIP", flush=True)
            continue
        (hit, fm23, bestca), uid, clubs = best
        # accept: hinted name needs an FM23 snapshot AT the hinted club; un-hinted needs an elite (CA>0) FM23 grade
        if hint and not hit:
            print(f"  {nm:26s} [{cname.get(cid,'?'):12s}] -> hint '{hint}' unmatched (only namesakes) — SKIP", flush=True)
            continue
        if not hint and fm23 == 0:
            print(f"  {nm:26s} [{cname.get(cid,'?'):12s}] -> FM23 grade has no CA, no club hint — SKIP", flush=True)
            continue
        clubs_s = ", ".join(sorted(clubs)[:3]) or "(no club)"
        flag = "  <hint✓" if hit else ""
        print(f"  {nm:26s} [{cname.get(cid,'?'):12s}] -> uid {uid} fm23ca={fm23} max={bestca} @ {clubs_s}{flag}", flush=True)
        for eid in pid_eids.get(pid, []):
            links.append((eid, uid))

    print(f"\nproposed player_xwalk links: {len(links)}", flush=True)
    if apply and links:
        for eid, uid in links:
            con.execute("INSERT OR REPLACE INTO player_xwalk "
                        "(espn_player_id, fm_uid, confidence, method) VALUES (?,?,?,?)",
                        (eid, uid, "high", "wc2022_natl"))
        con.commit()
        print(f"APPLIED {len(links)} links (confidence=high, method=wc2022_natl)", flush=True)
    elif not apply:
        print("(dry-run; re-run with --apply to insert)", flush=True)


if __name__ == "__main__":
    main()
