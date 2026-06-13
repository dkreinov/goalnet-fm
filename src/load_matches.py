"""Download football-data.co.uk EPL CSVs and load into match table.
Usage: python D:/Programming/claude/FM/src/load_matches.py
"""
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
import db

SEASONS = {"2324": "2023-24", "2425": "2024-25", "2526": "2025-26"}
URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
RAW_DIR = db.ROOT / "data" / "raw" / "football-data"

COLMAP = {
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast", "HC": "hc", "AC": "ac",
    "HF": "hf", "AF": "af", "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar",
    "B365H": "b365h", "B365D": "b365d", "B365A": "b365a",
    "AvgH": "avgh", "AvgD": "avgd", "AvgA": "avga",
}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    con = db.connect()
    total = 0
    for code, label in SEASONS.items():
        url = URL.format(code=code)
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        (RAW_DIR / f"E0_{code}.csv").write_bytes(r.content)
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8", errors="replace")))
        df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
        sid = db.season_id(con, label)
        n = 0
        for _, row in df.iterrows():
            date = pd.to_datetime(row["Date"], dayfirst=True).strftime("%Y-%m-%d")
            hid = db.club_id(con, str(row["HomeTeam"]))
            aid = db.club_id(con, str(row["AwayTeam"]))
            extras = {}
            for src_col, dst_col in COLMAP.items():
                v = row.get(src_col)
                extras[dst_col] = None if pd.isna(v) else float(v)
            con.execute(
                f"""INSERT OR REPLACE INTO match
                    (season_id, match_date, home_club_id, away_club_id, home_goals, away_goals,
                     ht_home_goals, ht_away_goals, referee, {','.join(COLMAP.values())})
                    VALUES (?,?,?,?,?,?,?,?,?,{','.join('?' * len(COLMAP))})""",
                (sid, date, hid, aid, int(row["FTHG"]), int(row["FTAG"]),
                 None if pd.isna(row.get("HTHG")) else int(row["HTHG"]),
                 None if pd.isna(row.get("HTAG")) else int(row["HTAG"]),
                 None if pd.isna(row.get("Referee")) else str(row["Referee"]),
                 *extras.values()),
            )
            n += 1
        con.commit()
        db.log(con, "football-data.co.uk", url, "ok", f"{label}: {n} matches")
        print(f"{label}: {n} matches")
        total += n
    print(f"TOTAL: {total}")
    con.close()


if __name__ == "__main__":
    main()
