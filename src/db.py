"""SQLite layer for FM ratings + match data. DB file: D:/Programming/claude/FM/data/fm.db"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fm.db"
SCHEMA_PATH = ROOT / "schema.sql"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=120)
    con.row_factory = sqlite3.Row
    # autocommit: several collector processes write concurrently; never hold a
    # transaction open across slow scraping loops or the others starve on the lock
    con.isolation_level = None
    con.execute("PRAGMA busy_timeout=120000")
    con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return con


def norm(s: str) -> str:
    """Normalize names for cross-source joins: strip accents, lowercase, alnum+space only."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def get_or_create(con, table: str, key_col: str, key_val, extra: dict | None = None) -> int:
    row = con.execute(f"SELECT rowid FROM {table} WHERE {key_col}=?", (key_val,)).fetchone()
    if row:
        return row[0]
    cols = {key_col: key_val, **(extra or {})}
    placeholders = ",".join("?" * len(cols))
    cur = con.execute(
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", list(cols.values())
    )
    return cur.lastrowid


def source_id(con, name: str, base_url: str = "") -> int:
    return get_or_create(con, "source", "name", name, {"base_url": base_url})


def season_id(con, label: str) -> int:
    return get_or_create(con, "season", "label", label)


def fm_version_id(con, game: str, db_version: str | None, release_date: str | None = None) -> int:
    row = con.execute(
        "SELECT fm_version_id FROM fm_version WHERE game=? AND db_version IS ?", (game, db_version)
    ).fetchone()
    if row:
        return row[0]
    cur = con.execute(
        "INSERT INTO fm_version (game, db_version, release_date) VALUES (?,?,?)",
        (game, db_version, release_date),
    )
    return cur.lastrowid


def club_id(con, name: str) -> int:
    n = norm(name)
    row = con.execute("SELECT club_id FROM club_alias WHERE alias=?", (n,)).fetchone()
    if row:
        return row[0]
    row = con.execute("SELECT club_id FROM club WHERE norm_name=?", (n,)).fetchone()
    if row:
        return row[0]
    cur = con.execute("INSERT INTO club (name, norm_name) VALUES (?,?)", (name, n))
    cid = cur.lastrowid
    con.execute("INSERT OR IGNORE INTO club_alias (alias, club_id) VALUES (?,?)", (n, cid))
    return cid


def add_club_alias(con, alias: str, canonical: str) -> None:
    cid = club_id(con, canonical)
    con.execute("INSERT OR REPLACE INTO club_alias (alias, club_id) VALUES (?,?)", (norm(alias), cid))


def player_id(con, name: str, dob: str | None = None, nationality: str | None = None,
              src: int | None = None, src_player_id: str | None = None) -> int:
    """Resolve player: prefer source id mapping, then (norm_name, dob), then norm_name alone."""
    if src is not None and src_player_id is not None:
        row = con.execute(
            "SELECT player_id FROM player_source_id WHERE source_id=? AND source_player_id=?",
            (src, str(src_player_id)),
        ).fetchone()
        if row:
            return row[0]
    n = norm(name)
    if dob:
        row = con.execute("SELECT player_id FROM player WHERE norm_name=? AND dob=?", (n, dob)).fetchone()
    else:
        row = con.execute("SELECT player_id FROM player WHERE norm_name=?", (n,)).fetchone()
    if row:
        pid = row[0]
    else:
        cur = con.execute(
            "INSERT INTO player (name, norm_name, dob, nationality) VALUES (?,?,?,?)",
            (name, n, dob, nationality),
        )
        pid = cur.lastrowid
    if src is not None and src_player_id is not None:
        con.execute(
            "INSERT OR IGNORE INTO player_source_id (source_id, source_player_id, player_id) VALUES (?,?,?)",
            (src, str(src_player_id), pid),
        )
    return pid


def attrs_hash(attrs: dict) -> str:
    return hashlib.sha1(json.dumps(attrs, sort_keys=True).encode()).hexdigest()


def save_snapshot(con, *, pid: int, src: int, fmv: int | None, cid: int | None,
                  snapshot_date: str, attrs: dict[str, tuple[str, float]],
                  meta: dict | None = None) -> int | None:
    """Save a dated rating snapshot. attrs: {attr_name: (category, value)}.
    meta: position/ca/pa/value_eur/wage_eur/foot_left/foot_right/height_cm/weight_kg.
    Skips insert if the latest prior snapshot for (player,source,fm_version) has identical attrs+meta.
    Returns snapshot_id or None if skipped."""
    meta = meta or {}
    h = attrs_hash({"a": {k: v[1] for k, v in attrs.items()}, "m": meta})
    prev = con.execute(
        """SELECT attrs_hash FROM player_snapshot
           WHERE player_id=? AND source_id=? AND fm_version_id IS ?
           ORDER BY snapshot_date DESC LIMIT 1""",
        (pid, src, fmv),
    ).fetchone()
    if prev and prev[0] == h:
        return None  # unchanged → no new dated row
    cur = con.execute(
        """INSERT OR IGNORE INTO player_snapshot
           (player_id, source_id, fm_version_id, club_id, snapshot_date, position,
            ca, pa, value_eur, wage_eur, foot_left, foot_right, height_cm, weight_kg, attrs_hash)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (pid, src, fmv, cid, snapshot_date, meta.get("position"),
         meta.get("ca"), meta.get("pa"), meta.get("value_eur"), meta.get("wage_eur"),
         meta.get("foot_left"), meta.get("foot_right"), meta.get("height_cm"),
         meta.get("weight_kg"), h),
    )
    if cur.rowcount == 0:
        return None
    sid = cur.lastrowid
    con.executemany(
        "INSERT OR REPLACE INTO player_attribute (snapshot_id, category, attr_name, attr_value) VALUES (?,?,?,?)",
        [(sid, cat, name_, val) for name_, (cat, val) in attrs.items()],
    )
    return sid


def log(con, source: str, url: str, status: str, detail: str = "") -> None:
    con.execute(
        "INSERT INTO scrape_log (source, url, status, detail) VALUES (?,?,?,?)",
        (source, url, status, detail[:500]),
    )
    con.commit()
