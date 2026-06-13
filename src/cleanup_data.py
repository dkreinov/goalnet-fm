"""One-time data cleanup, idempotent and safe to run alongside scrapers:
1. Normalize CA/PA to the 0-99 scale (futek stored raw 0-200; ratio ~2.0 vs fminside).
   Rule: any snapshot with ca/pa > 99 is raw-200 -> rescale ca = round(ca*99/200).
2. Fix futek snapshot.position: the futek search-table parser was off by one column, so
   position got the AGE. The true position sits in the cached row's 'nat' field -> recover it;
   if unrecoverable or not position-like, set NULL (lineups supply position for modelling anyway).
Usage: python D:/Programming/claude/FM/src/cleanup_data.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db

FUTEK_DIR = db.ROOT / "data" / "raw" / "futek"
POS_RE = re.compile(r"[A-Za-z]")  # a real FM position has letters (ST, AMR, D (RC)...)


def normalize_ca_pa(con):
    """Re-derive futek CA/PA from the cached RAW (0-200) values, rescaled to 0-99.
    Idempotent: always recomputes from the source-of-truth cache (×99/200), so it can't
    double-convert and it also fixes low-ability futek rows the naive >99 rule would miss.
    fminside snapshots are already 0-99 and left untouched."""
    fut = con.execute("SELECT source_id FROM source WHERE name='futek'").fetchone()
    if not fut:
        print("no futek source; skip CA/PA rescale")
        return
    fut = fut[0]
    uid_of = {r[0]: r[1] for r in con.execute(
        "SELECT player_id, source_player_id FROM player_source_id WHERE source_id=?", (fut,))}
    rows = con.execute(
        "SELECT snapshot_id, player_id FROM player_snapshot WHERE source_id=?", (fut,)).fetchall()
    fixed = missing = 0
    for sid, pid in rows:
        uid = uid_of.get(pid)
        f = FUTEK_DIR / f"{uid}.json" if uid else None
        if not f or not f.exists():
            missing += 1
            continue
        try:
            meta = json.loads(f.read_text(encoding="utf-8")).get("meta", {})
            raw_ca = meta.get("current_ability")
            raw_pa = meta.get("potential_ability")
            ca = round(int(raw_ca) * 99 / 200) if raw_ca else None
            pa = round(int(raw_pa) * 99 / 200) if raw_pa else None
            con.execute("UPDATE player_snapshot SET ca=?, pa=? WHERE snapshot_id=?", (ca, pa, sid))
            fixed += 1
        except Exception:
            missing += 1
    con.commit()
    print(f"CA/PA re-derived from raw cache to 0-99: {fixed} futek snapshots ({missing} cache-missing)")


def fix_futek_positions(con):
    fut = con.execute("SELECT source_id FROM source WHERE name='futek'").fetchone()
    if not fut:
        print("no futek source; skip position fix")
        return
    fut = fut[0]
    # map player_id -> futek uid
    uid_of = {r[0]: r[1] for r in con.execute(
        "SELECT player_id, source_player_id FROM player_source_id WHERE source_id=?", (fut,))}
    rows = con.execute(
        "SELECT snapshot_id, player_id, position FROM player_snapshot WHERE source_id=?", (fut,)).fetchall()
    recovered = nulled = ok = 0
    for sid, pid, pos in rows:
        if pos and POS_RE.search(pos):
            ok += 1                      # already a real position
            continue
        real = None
        uid = uid_of.get(pid)
        if uid:
            f = FUTEK_DIR / f"{uid}.json"
            if f.exists():
                try:
                    row = json.loads(f.read_text(encoding="utf-8")).get("row", {})
                    cand = row.get("nat")  # off-by-one: real position landed in 'nat'
                    if cand and POS_RE.search(cand):
                        real = cand.strip()
                except Exception:
                    pass
        if real:
            con.execute("UPDATE player_snapshot SET position=? WHERE snapshot_id=?", (real, sid))
            recovered += 1
        else:
            con.execute("UPDATE player_snapshot SET position=NULL WHERE snapshot_id=?", (sid,))
            nulled += 1
    con.commit()
    print(f"futek positions: recovered={recovered}, nulled={nulled}, already_ok={ok}")


def main():
    con = db.connect()
    normalize_ca_pa(con)
    fix_futek_positions(con)
    # report
    mx = con.execute("SELECT MAX(ca), MAX(pa) FROM player_snapshot").fetchone()
    print(f"post-cleanup max ca={mx[0]} pa={mx[1]} (both should be <=99)")
    bad = con.execute(
        "SELECT COUNT(*) FROM player_snapshot WHERE position GLOB '*[0-9]*' AND position NOT GLOB '*[A-Za-z]*'"
    ).fetchone()[0]
    print(f"snapshots with numeric-only position remaining: {bad}")
    con.close()


if __name__ == "__main__":
    main()
