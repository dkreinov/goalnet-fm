"""Schema migration v3: match context (time/venue/attendance/formation), event timeline,
managers + FM manager grades, club FM attributes. Idempotent.
Usage: python D:/Programming/claude/FM/src/migrate_v3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db


def col_exists(con, table, col):
    return any(r[1] == col for r in con.execute(f"PRAGMA table_info({table})"))


def add_col(con, table, col, decl):
    if not col_exists(con, table, col):
        con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def main():
    con = db.connect()
    # match context columns
    for col, decl in [("kickoff_time", "TEXT"), ("venue", "TEXT"), ("venue_city", "TEXT"),
                      ("attendance", "INTEGER"), ("home_formation", "TEXT"),
                      ("away_formation", "TEXT")]:
        add_col(con, "match", col, decl)

    con.executescript("""
    -- minute-stamped events: goals, cards, subs, penalties (for trajectory/autoregressive model)
    CREATE TABLE IF NOT EXISTS match_event (
        match_id   INTEGER NOT NULL REFERENCES match(match_id),
        seq        INTEGER NOT NULL,            -- order within match
        minute     INTEGER,                     -- clock minute (NULL if unparseable)
        type       TEXT NOT NULL,               -- 'Goal'|'Penalty - Scored'|'Yellow Card'|'Red Card'|'Substitution'|'Kickoff'...
        team_side  TEXT,                         -- 'home'|'away'|NULL
        club_id    INTEGER REFERENCES club(club_id),
        player     TEXT,                         -- athlete name if present
        detail     TEXT,
        PRIMARY KEY (match_id, seq)
    );
    CREATE INDEX IF NOT EXISTS ix_event_match ON match_event(match_id);

    -- managers (FM staff) + their dated FM grades, mirroring player_snapshot
    CREATE TABLE IF NOT EXISTS manager (
        manager_id INTEGER PRIMARY KEY,
        name       TEXT NOT NULL,
        norm_name  TEXT NOT NULL,
        nationality TEXT,
        UNIQUE (norm_name)
    );
    CREATE TABLE IF NOT EXISTS manager_source_id (
        source_id  INTEGER NOT NULL REFERENCES source(source_id),
        source_manager_id TEXT NOT NULL,
        manager_id INTEGER NOT NULL REFERENCES manager(manager_id),
        PRIMARY KEY (source_id, source_manager_id)
    );
    CREATE TABLE IF NOT EXISTS manager_snapshot (
        snapshot_id   INTEGER PRIMARY KEY,
        manager_id    INTEGER NOT NULL REFERENCES manager(manager_id),
        source_id     INTEGER NOT NULL REFERENCES source(source_id),
        fm_version_id INTEGER REFERENCES fm_version(fm_version_id),
        club_id       INTEGER REFERENCES club(club_id),   -- club managed at this snapshot
        snapshot_date TEXT NOT NULL,
        ca            INTEGER,
        attrs_hash    TEXT NOT NULL,
        UNIQUE (manager_id, source_id, fm_version_id, snapshot_date)
    );
    CREATE TABLE IF NOT EXISTS manager_attribute (
        snapshot_id INTEGER NOT NULL REFERENCES manager_snapshot(snapshot_id) ON DELETE CASCADE,
        category    TEXT NOT NULL,               -- 'coaching'|'mental'|'knowledge'|'other'
        attr_name   TEXT NOT NULL,
        attr_value  REAL NOT NULL,
        PRIMARY KEY (snapshot_id, attr_name)
    );

    -- club-level FM attributes (reputation, facilities, finances) per db version
    CREATE TABLE IF NOT EXISTS club_attribute (
        club_id       INTEGER NOT NULL REFERENCES club(club_id),
        source_id     INTEGER NOT NULL REFERENCES source(source_id),
        fm_version_id INTEGER REFERENCES fm_version(fm_version_id),
        snapshot_date TEXT NOT NULL,
        attr_name     TEXT NOT NULL,             -- 'reputation'|'training'|'youth'|'stadium_capacity'|'balance'...
        attr_value    REAL,
        attr_text     TEXT,
        PRIMARY KEY (club_id, source_id, fm_version_id, attr_name, snapshot_date)
    );
    """)
    con.commit()
    print("migration v3 applied. match columns:",
          [r[1] for r in con.execute("PRAGMA table_info(match)") if r[1] in
           ("kickoff_time", "venue", "venue_city", "attendance", "home_formation", "away_formation")])
    print("new tables present:", [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('match_event','manager','manager_snapshot','manager_attribute','club_attribute')")])
    con.close()


if __name__ == "__main__":
    main()
