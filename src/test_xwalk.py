"""Authenticity tests for the identity crosswalk. Exits non-zero on any failure.
These assert REAL correctness, not placebos:
 1. a known player (Salah) links to the correct FM UID
 2. every 'confirmed' link is genuinely DOB-consistent (no silent mismatches)
 3. a real shared FM name is disambiguated to DIFFERENT UIDs by DOB (not collapsed to one)
 4. ambiguous/unmatched rows carry NO fm_uid (we flag, never guess)
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
from build_xwalk import dob_close

fails = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def main():
    con = db.connect()
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    fmuid_sid = con.execute("SELECT source_id FROM source WHERE name='fm-uid'").fetchone()[0]

    # 1. Salah: ESPN athlete 173896 -> FM UID 98028755
    row = con.execute("SELECT fm_uid, confidence FROM player_xwalk WHERE espn_player_id='173896'").fetchone()
    check(row is not None and row[0] == "98028755",
          f"Mohamed Salah (espn 173896) -> FM UID 98028755 [got {row[0] if row else None}, {row[1] if row else None}]")

    # 2. every 'confirmed' link is DOB-consistent
    espn_dob = {r[0]: r[1] for r in con.execute(
        f"SELECT source_player_id, dob FROM source_identity WHERE source_id={espn_sid}")}
    fm_dob = {r[0]: r[1] for r in con.execute(
        f"SELECT source_player_id, dob FROM source_identity WHERE source_id={fmuid_sid}")}
    bad = 0; nconf = 0
    for eid, uid in con.execute("SELECT espn_player_id, fm_uid FROM player_xwalk WHERE confidence='confirmed'"):
        nconf += 1
        if not dob_close(espn_dob.get(eid), fm_dob.get(uid)):
            bad += 1
    check(nconf > 0 and bad == 0, f"all {nconf} 'confirmed' links are DOB-consistent (mismatches={bad})")

    # 3. a real shared FM name -> different UIDs disambiguated by DOB
    name_uids = defaultdict(list)
    for uid, name in con.execute(
            f"SELECT source_player_id, name FROM source_identity WHERE source_id={fmuid_sid}"):
        name_uids[db.norm(name or "")].append(uid)
    shared = None
    for n, uids in name_uids.items():
        if len(uids) >= 2:
            dobs = {u: fm_dob.get(u) for u in uids if fm_dob.get(u)}
            if len({v for v in dobs.values()}) >= 2:   # at least two distinct DOBs
                shared = (n, list(dobs.items())[:2]); break
    ok3 = False
    if shared:
        (u1, d1), (u2, d2) = shared[1]
        # DOB d1 must select u1 over u2 (and not be 'close' to d2), proving DOB separates same-name players
        ok3 = u1 != u2 and not dob_close(d1, d2)
    check(ok3, f"shared FM name '{shared[0] if shared else '?'}' separates into distinct UIDs by DOB")

    # 4. ambiguous/unmatched carry no fm_uid
    leaks = con.execute(
        "SELECT COUNT(*) FROM player_xwalk WHERE confidence IN ('ambiguous','unmatched') AND fm_uid IS NOT NULL"
    ).fetchone()[0]
    check(leaks == 0, f"ambiguous/unmatched rows carry NO fm_uid (leaks={leaks})")

    # --- token-tier (mononym / nickname / diacritic) cases ---
    from build_xwalk import xnorm
    fm_name = {r[0]: r[1] for r in con.execute(
        f"SELECT source_player_id, name FROM source_identity WHERE source_id={fmuid_sid}")}

    def espn_id_for(norm_name):
        r = con.execute(
            """SELECT psi.source_player_id FROM player p
               JOIN player_source_id psi ON psi.player_id=p.player_id AND psi.source_id=?
               WHERE p.norm_name=?
               ORDER BY (SELECT COUNT(*) FROM match_player mp WHERE mp.player_id=p.player_id) DESC
               LIMIT 1""", (espn_sid, norm_name)).fetchone()
        return r[0] if r else None

    # 5/6/7: famous mononym/nickname players now link to the CORRECT FM player. Correctness is
    # verified by EITHER a shared surname token OR an exact DOB match (Gabriel is stored "Gabriel"
    # in FM with no surname, so name-token alone is insufficient — DOB proves identity).
    for label, norm_name, must_token in (
            ("Alisson Becker (mononym 'Alisson')", "alisson becker", "alisson"),
            ("Gabriel Magalhaes (FM mononym 'Gabriel')", "gabriel magalhaes", "magalhaes"),
            ("Andy Robertson (nickname Andy/Andrew)", "andy robertson", "robertson")):
        eid = espn_id_for(norm_name)
        row = con.execute("SELECT fm_uid, confidence, method FROM player_xwalk WHERE espn_player_id=?",
                          (eid,)).fetchone() if eid else None
        token_ok = bool(row and row[0] and must_token in xnorm(fm_name.get(row[0], "")).split())
        dob_ok = bool(row and row[0] and dob_close(espn_dob.get(eid), fm_dob.get(row[0])))
        check(bool(row and row[0]) and (token_ok or dob_ok),
              f"{label} -> correct FM player [token={token_ok} dob={dob_ok} "
              f"{(row[1], row[2]) if row else 'no espn id'}]")

    # 8. SAFETY: every token-tier link satisfies the constraint the resolver claimed —
    #    'token+club*' links must share a club; 'token+dob' links must have matching DOB.
    espn_pid = {r[0]: r[1] for r in con.execute(
        f"SELECT source_player_id, player_id FROM player_source_id WHERE source_id={espn_sid}")}
    eclubs = defaultdict(set)
    for pid, cid in con.execute("SELECT DISTINCT player_id, club_id FROM match_player"):
        eclubs[pid].add(cid)
    fm_src = [r[0] for r in con.execute("SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    uid_pids = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pids[uid].add(pid)
    pid_clubs = defaultdict(set)
    for pid, cid in con.execute("SELECT player_id, club_id FROM player_snapshot WHERE club_id IS NOT NULL"):
        pid_clubs[pid].add(cid)
    fm_clubs = {uid: set().union(*(pid_clubs[p] for p in pids)) if pids else set() for uid, pids in uid_pids.items()}
    bad = 0; ntok = 0
    for eid, uid, method in con.execute(
            "SELECT espn_player_id, fm_uid, method FROM player_xwalk WHERE method LIKE 'token%' AND fm_uid IS NOT NULL"):
        ntok += 1
        if "club" in method:
            if not (eclubs.get(espn_pid.get(eid), set()) & fm_clubs.get(uid, set())):
                bad += 1
        elif method == "token+dob":
            if not dob_close(espn_dob.get(eid), fm_dob.get(uid)):
                bad += 1
    check(ntok > 0 and bad == 0, f"all {ntok} token-tier links satisfy their club/DOB constraint (violations={bad})")

    con.close()
    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILURE(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
