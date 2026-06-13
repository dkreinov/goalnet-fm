-- Football Manager ratings + match data warehouse
-- Versioning rule: player attributes saved as dated snapshots; new row only when values change.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source (
    source_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,          -- 'sortitoutsi', 'fminside', 'kaggle:<dataset>', ...
    base_url    TEXT
);

CREATE TABLE IF NOT EXISTS fm_version (
    fm_version_id INTEGER PRIMARY KEY,
    game          TEXT NOT NULL,               -- 'FM24', 'FM26'
    db_version    TEXT,                        -- '24.3.0', 'winter update', NULL if unknown
    release_date  TEXT,                        -- ISO date if known
    UNIQUE (game, db_version)
);

CREATE TABLE IF NOT EXISTS season (
    season_id INTEGER PRIMARY KEY,
    label     TEXT NOT NULL UNIQUE             -- '2023-24'
);

CREATE TABLE IF NOT EXISTS club (
    club_id   INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE,            -- canonical name, e.g. 'Arsenal'
    norm_name TEXT NOT NULL                    -- lowercased, no punctuation, for joins
);

CREATE TABLE IF NOT EXISTS club_alias (        -- map source-specific spellings to canonical club
    alias     TEXT PRIMARY KEY,                -- normalized alias
    club_id   INTEGER NOT NULL REFERENCES club(club_id)
);

CREATE TABLE IF NOT EXISTS player (
    player_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    norm_name   TEXT NOT NULL,                 -- normalized for cross-source joins
    dob         TEXT,                          -- ISO date
    nationality TEXT,
    UNIQUE (norm_name, dob)
);

CREATE TABLE IF NOT EXISTS player_source_id (  -- source's own id for a player (e.g. sortitoutsi uid = in-game uid)
    source_id        INTEGER NOT NULL REFERENCES source(source_id),
    source_player_id TEXT NOT NULL,
    player_id        INTEGER NOT NULL REFERENCES player(player_id),
    PRIMARY KEY (source_id, source_player_id)
);

-- One snapshot = one observation of a player's full rating set at a point in time.
CREATE TABLE IF NOT EXISTS player_snapshot (
    snapshot_id   INTEGER PRIMARY KEY,
    player_id     INTEGER NOT NULL REFERENCES player(player_id),
    source_id     INTEGER NOT NULL REFERENCES source(source_id),
    fm_version_id INTEGER REFERENCES fm_version(fm_version_id),
    club_id       INTEGER REFERENCES club(club_id),
    snapshot_date TEXT NOT NULL,               -- date this rating set is valid/observed (ISO)
    position      TEXT,                        -- e.g. 'ST (C)', 'D (RC)'
    ca            INTEGER,                     -- current ability
    pa            INTEGER,                     -- potential ability
    value_eur     INTEGER,
    wage_eur      INTEGER,
    foot_left     INTEGER,
    foot_right    INTEGER,
    height_cm     INTEGER,
    weight_kg     INTEGER,
    attrs_hash    TEXT NOT NULL,               -- sha1 of full attribute dict; dedupe identical snapshots
    UNIQUE (player_id, source_id, fm_version_id, snapshot_date)
);
CREATE INDEX IF NOT EXISTS ix_snapshot_player ON player_snapshot(player_id, snapshot_date);
CREATE INDEX IF NOT EXISTS ix_snapshot_hash   ON player_snapshot(player_id, source_id, attrs_hash);

-- All rating categories, EAV: technical/mental/physical/goalkeeping/hidden.
CREATE TABLE IF NOT EXISTS player_attribute (
    snapshot_id INTEGER NOT NULL REFERENCES player_snapshot(snapshot_id) ON DELETE CASCADE,
    category    TEXT NOT NULL,                 -- 'technical'|'mental'|'physical'|'goalkeeping'|'hidden'|'other'
    attr_name   TEXT NOT NULL,                 -- 'finishing', 'composure', ...
    attr_value  REAL NOT NULL,
    PRIMARY KEY (snapshot_id, attr_name)
);

CREATE TABLE IF NOT EXISTS match (
    match_id    INTEGER PRIMARY KEY,
    season_id   INTEGER NOT NULL REFERENCES season(season_id),
    match_date  TEXT NOT NULL,
    home_club_id INTEGER NOT NULL REFERENCES club(club_id),
    away_club_id INTEGER NOT NULL REFERENCES club(club_id),
    home_goals  INTEGER NOT NULL,
    away_goals  INTEGER NOT NULL,
    ht_home_goals INTEGER,
    ht_away_goals INTEGER,
    referee     TEXT,
    -- match stats (from football-data.co.uk)
    hs INTEGER, as_ INTEGER,                   -- shots
    hst INTEGER, ast INTEGER,                  -- shots on target
    hc INTEGER, ac INTEGER,                    -- corners
    hf INTEGER, af INTEGER,                    -- fouls
    hy INTEGER, ay INTEGER, hr INTEGER, ar INTEGER, -- cards
    -- odds (Bet365 + market averages)
    b365h REAL, b365d REAL, b365a REAL,
    avgh REAL, avgd REAL, avga REAL,
    -- xG (understat/fbref)
    xg_home REAL, xg_away REAL,
    UNIQUE (match_date, home_club_id, away_club_id)
);

CREATE TABLE IF NOT EXISTS match_player (      -- lineups
    match_id  INTEGER NOT NULL REFERENCES match(match_id),
    player_id INTEGER NOT NULL REFERENCES player(player_id),
    club_id   INTEGER NOT NULL REFERENCES club(club_id),
    started   INTEGER NOT NULL,                -- 1 starter, 0 sub
    minutes   INTEGER,
    position  TEXT,
    rating    REAL,                            -- post-match rating if source provides
    PRIMARY KEY (match_id, player_id)
);

CREATE TABLE IF NOT EXISTS scrape_log (
    id        INTEGER PRIMARY KEY,
    ts        TEXT NOT NULL DEFAULT (datetime('now')),
    source    TEXT NOT NULL,
    url       TEXT,
    status    TEXT NOT NULL,                   -- 'ok'|'error'|'skip'
    detail    TEXT
);
