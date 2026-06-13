"""Download football-data.co.uk CSVs for all registered leagues x seasons and load into match.
Tags each match with competition_id. Skips (league, season) combos with no file (404).
Usage: python D:/Programming/claude/FM/src/load_matches.py [league_name ...]
"""
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import db
import leagues as L

URL = "https://www.football-data.co.uk/mmz4281/{code}/{fd}.csv"
RAW_DIR = db.ROOT / "data" / "raw" / "football-data"

COLMAP = {
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast", "HC": "hc", "AC": "ac",
    "HF": "hf", "AF": "af", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
    "B365H": "b365h", "B365D": "b365d", "B365A": "b365a",
    "AvgH": "avgh", "AvgD": "avgd", "AvgA": "avga",
}


def load_one(con, lg, comp_id):
    total = 0
    for season, code in L.FD_SEASON.items():
        if not lg.get("fd"):
            continue
        url = URL.format(code=code, fd=lg["fd"])
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200 or len(r.content) < 200:
                db.log(con, "football-data", url, "skip", f"{lg['name']} {season}: HTTP {r.status_code}")
                continue
        except requests.RequestException as e:
            db.log(con, "football-data", url, "error", str(e))
            continue
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_DIR / f"{lg['fd']}_{code}.csv").write_bytes(r.content)
        try:
            df = pd.read_csv(io.StringIO(r.content.decode("utf-8", errors="replace")))
        except Exception as e:
            db.log(con, "football-data", url, "error", f"parse: {e}")
            continue
        if "HomeTeam" not in df.columns:
            continue
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        sid = db.season_id(con, season)
        n = 0
        for _, row in df.iterrows():
            try:
                date = pd.to_datetime(row["Date"], dayfirst=True).strftime("%Y-%m-%d")
            except Exception:
                continue
            hid = db.club_id(con, str(row["HomeTeam"]))
            aid = db.club_id(con, str(row["AwayTeam"]))
            extras = {dst: (None if pd.isna(row.get(src)) else float(row.get(src)))
                      for src, dst in COLMAP.items()}
            set_clause = ", ".join(f"{c}=excluded.{c}" for c in
                                   ["season_id", "competition_id", "match_kind", "home_goals",
                                    "away_goals", "ht_home_goals", "ht_away_goals", "referee",
                                    *COLMAP.values()])
            con.execute(
                f"""INSERT INTO match
                    (season_id, competition_id, match_kind, match_date, home_club_id, away_club_id,
                     home_goals, away_goals, ht_home_goals, ht_away_goals, referee, {','.join(COLMAP.values())})
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,{','.join('?' * len(COLMAP))})
                    ON CONFLICT(match_date, home_club_id, away_club_id) DO UPDATE SET {set_clause}""",
                (sid, comp_id, "league", date, hid, aid, int(row["FTHG"]), int(row["FTAG"]),
                 None if pd.isna(row.get("HTHG")) else int(row["HTHG"]),
                 None if pd.isna(row.get("HTAG")) else int(row["HTAG"]),
                 None if pd.isna(row.get("Referee")) else str(row["Referee"]),
                 *extras.values()))
            n += 1
        con.commit()
        db.log(con, "football-data", url, "ok", f"{lg['name']} {season}: {n}")
        print(f"  {lg['name']} {season}: {n} matches")
        total += n
    return total


def main():
    con = db.connect()
    names = sys.argv[1:]
    targets = [L.BY_NAME[n] for n in names] if names else L.enabled()
    grand = 0
    for lg in targets:
        comp_id = db.competition_id(con, lg["name"], lg["country"], lg["tier"], lg["rank"], "league")
        print(f"== {lg['name']} (rank {lg['rank']}) ==")
        grand += load_one(con, lg, comp_id)
    print(f"TOTAL matches loaded/updated: {grand}")
    con.close()


if __name__ == "__main__":
    main()
