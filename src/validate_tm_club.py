"""Validate the tm_club (our club -> Transfermarkt id) mappings, which were built by first-search-hit and
are unreliable for ambiguous names / national teams (e.g. 'Chad' -> a EUR1bn club). For each mapping, fetch
the TM squad and check how many of its players' names also appear in OUR lineup data for that club: a correct
mapping overlaps heavily; a wrong one overlaps ~0. Writes tm_club.valid (1/0). All downstream TM enrichment
(market value, etc.) must use only valid=1 mappings.

Usage: python D:/Programming/claude/FM/src/validate_tm_club.py [--sample N]
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import fetch
import scrape_tm_squads as tms

SEASONS = ["2023", "2022", "2024", "2021"]   # try a few until a squad is found


def main():
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else None
    con = db.connect()
    con.execute("PRAGMA synchronous=NORMAL")
    if "valid" not in [c[1] for c in con.execute("pragma table_info(tm_club)")]:
        con.execute("ALTER TABLE tm_club ADD COLUMN valid INTEGER")

    # our players per club (norm_name set), from lineups
    club_by_name = {nm: cid for cid, nm in con.execute("SELECT club_id, name FROM club")}
    our = defaultdict(set)
    for cid, nn in con.execute(
            "SELECT DISTINCT mp.club_id, p.norm_name FROM match_player mp JOIN player p USING(player_id)"):
        our[cid].add(nn)

    todo = [(n, t) for n, t in con.execute("SELECT club_name, tm_id FROM tm_club WHERE tm_id IS NOT NULL")]
    if sample:
        todo = todo[:sample]
    print(f"validating {len(todo):,} tm_club mappings...", flush=True)
    valid = invalid = noours = 0
    for i, (name, tid) in enumerate(todo):
        cid = club_by_name.get(name)
        ours = our.get(cid, set())
        if not ours:
            con.execute("UPDATE tm_club SET valid=0 WHERE tm_id=?", (tid,)); noours += 1; continue
        tm_names = set()
        for yr in SEASONS:
            tm_names = {nn for nn, _dob in tms.tm_squad(tid, yr)}
            if tm_names:
                break
        overlap = len(tm_names & ours)
        frac = overlap / max(len(tm_names), 1)
        ok = 1 if (overlap >= 4 and frac >= 0.20) else 0
        con.execute("UPDATE tm_club SET valid=? WHERE tm_id=?", (ok, tid))
        valid += ok; invalid += (1 - ok) if tm_names else 0
        if not ok and i < 9999 and overlap == 0 and tm_names:
            pass
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)}  valid={valid} invalid={invalid}", flush=True)
    tot = con.execute("SELECT COUNT(*) FROM tm_club WHERE tm_id IS NOT NULL").fetchone()[0]
    nv = con.execute("SELECT COUNT(*) FROM tm_club WHERE valid=1").fetchone()[0]
    print(f"\nVALID mappings: {nv:,}/{tot:,} ({100*nv//tot}%);  no-our-players={noours}")
    print("sample INVALID (wrong/unverifiable):")
    for nm, t in con.execute("SELECT club_name, tm_id FROM tm_club WHERE valid=0 AND tm_id IS NOT NULL LIMIT 12"):
        print(f"   {nm}  (tm {t})")
    con.close()


if __name__ == "__main__":
    main()
