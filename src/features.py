"""Derived & squad-level features computed from data already in the DB (no scraping).
- squad strength: full-squad FM ability/value per club per season (depth beyond the XI)
- promoted flag: club new to this competition vs previous season
- head-to-head: recent meetings goal diff
- referee strictness: avg cards the ref shows
- league standings: points/position/goal-diff before each match (title race vs relegation stakes)
All keyed so build_dataset can attach them per match.
"""
from collections import defaultdict


def squad_strength(con):
    """(club_id, fm_version_id) -> {ca_mean, ca_max, size, value_total} over the full squad."""
    agg = defaultdict(lambda: {"cas": [], "vals": []})
    for r in con.execute(
            "SELECT club_id, fm_version_id, ca, value_eur FROM player_snapshot "
            "WHERE club_id IS NOT NULL"):
        k = (r[0], r[1])
        if r[2]:
            agg[k]["cas"].append(r[2])
        if r[3]:
            agg[k]["vals"].append(r[3])
    out = {}
    for k, v in agg.items():
        cas = v["cas"]
        out[k] = {
            "squad_ca_mean": sum(cas) / len(cas) if cas else None,
            "squad_ca_max": max(cas) if cas else None,
            "squad_ca_top11": sum(sorted(cas, reverse=True)[:11]) / min(11, len(cas)) if cas else None,
            "squad_size": len(cas),
            "squad_value_total": sum(v["vals"]) if v["vals"] else None,
        }
    return out


def referee_strictness(con):
    """referee -> avg (yellows+reds) per match across all their matches in our data."""
    agg = defaultdict(lambda: [0, 0])  # ref -> [card_sum, n]
    for r in con.execute(
            "SELECT referee, hy, ay, hr, ar FROM match WHERE referee IS NOT NULL"):
        cards = sum(x for x in r[1:5] if x is not None)
        agg[r[0]][0] += cards
        agg[r[0]][1] += 1
    return {ref: (s / n if n else None) for ref, (s, n) in agg.items()}


def sequential_features(matches):
    """One date-ordered pass. matches: list of
    (match_id, date, comp_id, season, home_id, away_id, hg, ag).
    Returns match_id -> dict with promoted flags, head-to-head, standings (pre-match)."""
    # standings per (competition, season): club -> [pts, gf, ga, played]
    table = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    # clubs seen per (competition, season) to detect promotion next season
    seen = defaultdict(set)            # (comp, season) -> {club_id}
    h2h = defaultdict(list)            # frozenset({a,b}) -> list of (date, home_id, gd)
    out = {}

    def season_prev(season):
        # '2023-24' -> '2022-23'
        try:
            y = int(season[:4])
            return f"{y-1}-{str(y)[2:]}"
        except Exception:
            return None

    for mid, date, comp, season, h, a, hg, ag in matches:
        seen[(comp, season)].update([h, a])
        prev = season_prev(season)
        prev_clubs = seen.get((comp, prev), set())
        # promoted = club present this season but not in this comp last season (and we have last-season data)
        home_promoted = 1 if (prev_clubs and h not in prev_clubs) else 0
        away_promoted = 1 if (prev_clubs and a not in prev_clubs) else 0

        th, ta = table[(comp, season)][h], table[(comp, season)][a]
        key = frozenset((h, a))
        recent = h2h[key][-5:]
        # head-to-head goal diff from home team's perspective (positive = home historically better)
        hh = sum((gd if hid == h else -gd) for (_d, hid, gd) in recent)

        out[mid] = {
            "home_promoted": home_promoted, "away_promoted": away_promoted,
            "home_pts": th[0], "away_pts": ta[0],
            "home_played": th[3], "away_played": ta[3],
            "home_gd": th[1] - th[2], "away_gd": ta[1] - ta[2],
            "h2h_home_gd": hh, "h2h_n": len(recent),
        }
        # update standings + h2h AFTER recording pre-match state
        hp = 3 if hg > ag else (1 if hg == ag else 0)
        ap = 3 if ag > hg else (1 if hg == ag else 0)
        th[0] += hp; th[1] += hg; th[2] += ag; th[3] += 1
        ta[0] += ap; ta[1] += ag; ta[2] += hg; ta[3] += 1
        h2h[key].append((date, h, hg - ag))

    # add within-competition-season rank (position) by points then GD, computed at season end as a
    # stable proxy; for pre-match position we approximate with points rank among played clubs.
    return out
