"""Source-independent match linking. football-data, ESPN, and understat all spell clubs
differently, so we align a foreign match to our football-data `match` rows by
(competition, date within ±2d, final score), using club-name token similarity only to
break ties among same-date-same-score candidates.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db


def _sim(a_norm, b_norm):
    ta, tb = set(a_norm.split()), set(b_norm.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    # also reward substring containment (Wolverhampton vs Wolves)
    sub = 1 if (a_norm in b_norm or b_norm in a_norm) else 0
    return inter / max(len(ta), len(tb)) + 0.25 * sub


def find_match(con, comp_id, date, hg, ag, home_name, away_name, day_window=2):
    """Return match_id or None. Matches on competition + date window + exact score, then
    disambiguates by home/away club-name similarity."""
    rows = con.execute(
        """SELECT m.match_id, ch.norm_name, ca.norm_name
           FROM match m JOIN club ch ON ch.club_id=m.home_club_id
                        JOIN club ca ON ca.club_id=m.away_club_id
           WHERE m.competition_id=? AND m.home_goals=? AND m.away_goals=?
             AND date(m.match_date) BETWEEN date(?, ?) AND date(?, ?)""",
        (comp_id, hg, ag, date, f"-{day_window} day", date, f"+{day_window} day")).fetchall()
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0][0]
    hn, an = db.norm(home_name), db.norm(away_name)
    best, best_s = None, -1.0
    for mid, ch, ca in rows:
        s = _sim(hn, ch) + _sim(an, ca)
        if s > best_s:
            best, best_s = mid, s
    return best
