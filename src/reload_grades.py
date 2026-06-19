"""Re-key grade-players by FM game-UID (un-merge the name-merge bug). FULLY OFFLINE.

player_snapshot is 100% grade sources (ESPN has 0 snapshot rows), so we can safely clear all grade
snapshots + attributes + grade player_source_id, then reload with db.player_id(grade_uid=True) which
keys each distinct FM-UID to its own player. ESPN player rows, player_source_id(espn), match_player and
matches are NEVER touched. Single DB writer.

Sources reloaded here (offline):
  - fminside: each cached player page embeds og:url=".../players/{dbid}-slug/{uid}-slug" -> edition+UID;
              parse_player() gives name/club/attrs. Scans data/cache (no network enumeration).
  - futek:    replays data/raw/futek/_epl_index.json + per-uid {uid}.json (no network).
Kaggle is reloaded separately afterwards via `python src/kaggle_load.py --reset` (also UID-keyed now).

Usage: python D:/Programming/claude/FM/src/reload_grades.py [--dry-run]
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db as dbmod
import fetch
import scrape_fminside as sf

DRY = "--dry-run" in sys.argv
CACHE = fetch.CACHE
FUTEK = dbmod.ROOT / "data" / "raw" / "futek"
OG_RE = re.compile(rb'og:url" content="https://fminside\.net/players/(\d+)-[^/]+/(\d+)-')


def clear_grade_data(con):
    gids = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    ph = ",".join("?" * len(gids))
    n_attr = con.execute(
        f"SELECT COUNT(*) FROM player_attribute WHERE snapshot_id IN "
        f"(SELECT snapshot_id FROM player_snapshot WHERE source_id IN ({ph}))", gids).fetchone()[0]
    n_snap = con.execute(f"SELECT COUNT(*) FROM player_snapshot WHERE source_id IN ({ph})", gids).fetchone()[0]
    n_psi = con.execute(f"SELECT COUNT(*) FROM player_source_id WHERE source_id IN ({ph})", gids).fetchone()[0]
    print(f"clearing grade data: {n_attr:,} attrs, {n_snap:,} snapshots, {n_psi:,} source-id maps")
    if DRY:
        return
    con.execute("BEGIN")
    con.execute(f"DELETE FROM player_attribute WHERE snapshot_id IN "
                f"(SELECT snapshot_id FROM player_snapshot WHERE source_id IN ({ph}))", gids)
    con.execute(f"DELETE FROM player_snapshot WHERE source_id IN ({ph})", gids)
    con.execute(f"DELETE FROM player_source_id WHERE source_id IN ({ph})", gids)
    con.execute("COMMIT")


def _parse_file(fp_str):
    """Worker (runs in a child process): read one cache file, reject non-HTML cheaply, and if it is an
    fminside player page parse it. CPU-heavy BeautifulSoup parse is the bottleneck -> fan out over cores.
    Returns (uid, dbid, parsed_player) | ('skip', dbid) | None (not an fminside page / unreadable)."""
    try:
        data = Path(fp_str).read_bytes()
    except Exception:
        return None
    if not data[:64].lstrip().lower().startswith((b"<!doctype", b"<html")):
        return None
    m = OG_RE.search(data)   # og:url sits deep in fminside's <head> (~byte 11k)
    if not m:
        return None
    dbid = int(m.group(1)); uid = m.group(2).decode()
    try:
        p = sf.parse_player(data.decode("utf-8", errors="replace"))
    except Exception:
        return ("skip", dbid)
    if not p or not p["attrs"]:
        return ("skip", dbid)
    return (uid, dbid, p)


def reload_fminside(con):
    from concurrent.futures import ProcessPoolExecutor
    src = dbmod.source_id(con, "fminside", sf.BASE)
    fmv_cache = {}
    for dbid, cfg in sf.DBS.items():
        fmv_cache[dbid] = (dbmod.fm_version_id(con, cfg["game"], cfg["db_version"], cfg["date"]), cfg["date"])
    files = [str(f) for f in CACHE.rglob("*.html")]
    print(f"fminside: scanning {len(files):,} cache files (parallel parse)...", flush=True)
    saved = skipped = matched = bad_db = 0
    pid_memo = {}   # uid -> pid, avoids re-querying player_source_id for the same UID across editions
    if not DRY:
        con.execute("BEGIN")
    in_txn = 0
    ex = ProcessPoolExecutor(max_workers=3)   # 4 cores: 3 parse, 1 main(DB writes)
    try:
        for i, res in enumerate(ex.map(_parse_file, files, chunksize=100)):
            if (i + 1) % 20000 == 0:
                print(f"    {i+1:,}/{len(files):,} files; matched={matched:,} saved={saved:,} skip={skipped:,}", flush=True)
            if res is None:
                continue
            matched += 1
            if res[0] == "skip":
                skipped += 1
                continue
            uid, dbid, p = res
            if dbid not in fmv_cache:
                bad_db += 1
                continue
            fmv, date = fmv_cache[dbid]
            if DRY:
                saved += 1
                continue
            pid = pid_memo.get(uid)
            if pid is None:
                pid = dbmod.player_id(con, p["name"], src=src, src_player_id=uid, grade_uid=True)
                pid_memo[uid] = pid
            cid = dbmod.club_id(con, p["club"]) if p["club"] else None
            sid = dbmod.save_snapshot(
                con, pid=pid, src=src, fmv=fmv, cid=cid, snapshot_date=date, attrs=p["attrs"],
                meta={k: p[k] for k in ("position", "ca", "pa", "value_eur", "wage_eur",
                                        "foot_left", "foot_right", "height_cm", "weight_kg")})
            if sid:
                saved += 1
            else:
                skipped += 1
            in_txn += 1
            if in_txn >= 2000:
                con.execute("COMMIT"); con.execute("BEGIN"); in_txn = 0
        if not DRY and con.in_transaction:
            con.execute("COMMIT")
    finally:
        # never leave workers (and the held write lock) hanging if anything throws
        ex.shutdown(wait=False, cancel_futures=True)
        if not DRY and con.in_transaction:
            con.execute("ROLLBACK")
    print(f"fminside reload: matched={matched:,} saved={saved:,} skipped={skipped:,} bad_db={bad_db:,}")


def reload_futek(con):
    src = dbmod.source_id(con, "futek", "https://www.futek.io")
    fmv = dbmod.fm_version_id(con, "FM24", "24.0-futek-export", "2023-06-01")
    date = "2023-06-01"
    idx_path = FUTEK / "_epl_index.json"
    if idx_path.exists():
        index = json.loads(idx_path.read_text(encoding="utf-8"))
    else:
        index = {}
    # source of truth = the cached per-uid json files; index only supplies fallback name/club
    jfiles = sorted(p for p in FUTEK.glob("*.json") if p.stem != "_epl_index")
    print(f"futek: {len(jfiles):,} cached player jsons")
    saved = skipped = 0
    if not DRY:
        con.execute("BEGIN")
    for jp in jfiles:
        try:
            rec = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            continue
        uid = rec.get("uid") or jp.stem
        row = rec.get("row") or index.get(uid, {})
        attrs = {k: (cv[0], float(cv[1])) for k, cv in rec.get("attrs", {}).items()}
        if not attrs:
            skipped += 1
            continue
        name = (rec.get("meta", {}).get("player_name") or row.get("name") or "").strip()
        if not name:
            skipped += 1
            continue
        if DRY:
            saved += 1
            continue
        pid = dbmod.player_id(con, name, src=src, src_player_id=uid, grade_uid=True)
        club = rec.get("meta", {}).get("club") or row.get("club")
        cid = dbmod.club_id(con, club) if club else None
        ca = int(rec.get("meta", {}).get("current_ability") or row.get("ca") or 0) or None
        pa = int(rec.get("meta", {}).get("potential_ability") or row.get("pa") or 0) or None
        if dbmod.save_snapshot(con, pid=pid, src=src, fmv=fmv, cid=cid, snapshot_date=date,
                               attrs=attrs, meta={"position": row.get("pos"), "ca": ca, "pa": pa}):
            saved += 1
        else:
            skipped += 1
    if not DRY and con.in_transaction:
        con.execute("COMMIT")
    print(f"futek reload: saved={saved:,} skipped={skipped:,}")


def main():
    con = dbmod.connect()
    con.execute("PRAGMA synchronous=NORMAL")
    print("=== reload_grades (DRY-RUN) ===" if DRY else "=== reload_grades (WRITING) ===")
    clear_grade_data(con)
    reload_fminside(con)
    reload_futek(con)
    con.close()
    print("done. Next: python src/kaggle_load.py --reset")


if __name__ == "__main__":
    main()
