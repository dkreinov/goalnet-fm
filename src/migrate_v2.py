"""Schema migration for multi-league + national-team expansion.
Adds: competition dimension, match.competition_id, unmatched_name tracking, match_kind.
Idempotent — safe to run repeatedly. Usage: python D:/Programming/claude/FM/src/migrate_v2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db


def col_exists(con, table, col):
    return any(r[1] == col for r in con.execute(f"PRAGMA table_info({table})"))


def main():
    con = db.connect()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS competition (
        competition_id INTEGER PRIMARY KEY,
        name        TEXT NOT NULL UNIQUE,      -- 'England Premier League'
        country     TEXT,                      -- 'England' or 'International'
        tier        INTEGER,                   -- 1 = top flight, 2 = second, NULL = international
        rank        INTEGER,                   -- our global priority rank (1 = highest)
        kind        TEXT NOT NULL DEFAULT 'league'  -- 'league' | 'national'
    );

    CREATE TABLE IF NOT EXISTS unmatched_name (
        id          INTEGER PRIMARY KEY,
        source      TEXT NOT NULL,             -- where the unmatched name came from ('espn-lineup', ...)
        raw_name    TEXT NOT NULL,
        norm_name   TEXT NOT NULL,
        club        TEXT,
        competition TEXT,
        context     TEXT,                      -- e.g. match_id or season
        n_seen      INTEGER NOT NULL DEFAULT 1,
        resolved_player_id INTEGER REFERENCES player(player_id),  -- NULL until fixed
        note        TEXT,
        UNIQUE (source, norm_name, club)
    );
    CREATE INDEX IF NOT EXISTS ix_unmatched_unresolved
        ON unmatched_name(resolved_player_id) WHERE resolved_player_id IS NULL;
    """)

    if not col_exists(con, "match", "competition_id"):
        con.execute("ALTER TABLE match ADD COLUMN competition_id INTEGER REFERENCES competition(competition_id)")
    if not col_exists(con, "match", "match_kind"):
        con.execute("ALTER TABLE match ADD COLUMN match_kind TEXT NOT NULL DEFAULT 'league'")

    # backfill existing rows as England Premier League
    epl = con.execute(
        "INSERT OR IGNORE INTO competition (name, country, tier, rank, kind) "
        "VALUES ('England Premier League','England',1,1,'league')")
    cid = con.execute("SELECT competition_id FROM competition WHERE name='England Premier League'").fetchone()[0]
    con.execute("UPDATE match SET competition_id=? WHERE competition_id IS NULL", (cid,))
    con.commit()
    print("migration v2 applied.")
    print("matches tagged EPL:", con.execute(
        "SELECT COUNT(*) FROM match WHERE competition_id=?", (cid,)).fetchone()[0])
    con.close()


if __name__ == "__main__":
    main()
