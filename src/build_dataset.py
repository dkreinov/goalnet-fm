"""Build NN training dataset: per match, join starters to latest FM snapshot before
match date, aggregate attributes by position group, add Elo + form context.
Output: data/dataset.parquet (one row per match) + join coverage report.
Usage: python D:/Programming/claude/FM/src/build_dataset.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import db

POS_GROUP = {  # ESPN position abbreviations -> group
    "G": "GK", "D": "DEF", "M": "MID", "F": "ATT", "A": "ATT",
}
ATTR_GROUPS = ("technical", "mental", "physical", "set_pieces", "goalkeeping", "hidden")


def load_snapshots(con):
    """player_id -> sorted list of (snapshot_date, snapshot_id, source_name)."""
    snaps = defaultdict(list)
    for r in con.execute(
            "SELECT ps.player_id, ps.snapshot_date, ps.snapshot_id, s.name, ps.ca, ps.pa "
            "FROM player_snapshot ps JOIN source s USING(source_id)"):
        snaps[r[0]].append((r[1], r[2], r[3], r[4], r[5]))
    for v in snaps.values():
        v.sort()
    return snaps


def load_attrs(con):
    attrs = defaultdict(dict)
    for r in con.execute("SELECT snapshot_id, category, attr_name, attr_value FROM player_attribute"):
        attrs[r[0]][(r[1], r[2])] = r[3]
    return attrs


def latest_before(snaps_for_player, date):
    """Latest snapshot strictly before match date; fminside preferred on date ties."""
    best = None
    for sd, sid, srcname, ca, pa in snaps_for_player:
        if sd <= date:
            if best is None or sd > best[0] or (sd == best[0] and srcname == "fminside"):
                best = (sd, sid, srcname, ca, pa)
    return best


def name_index(con):
    """norm_name -> [player_id with snapshots]; for joining lineup players lacking direct ids."""
    has_snap = {r[0] for r in con.execute("SELECT DISTINCT player_id FROM player_snapshot")}
    idx = defaultdict(list)
    for r in con.execute("SELECT player_id, norm_name FROM player"):
        if r[0] in has_snap:
            idx[r[1]].append(r[0])
    return idx, has_snap


def resolve(pid, pname_norm, club_id_, has_snap, idx):
    """Map a lineup player to a player_id with FM snapshots.
    Ambiguous keys are disambiguated by club: candidate must have a snapshot at the
    lineup club. Unique keys resolve directly."""
    if pid in has_snap:
        return pid

    def pick(cands):
        cands = list(cands)
        if len(cands) == 1:
            return cands[0]
        at_club = [c for c in cands if club_id_ in PLAYER_CLUBS.get(c, ())]
        return at_club[0] if len(at_club) == 1 else None

    cands = idx.get(pname_norm)
    if cands:
        r = pick(cands)
        if r:
            return r
    parts = pname_norm.split()
    if len(parts) >= 2:
        for key in ((parts[0][0], parts[-1]), (parts[-1][0], parts[0])):
            if key in FALLBACK_IDX:
                r = pick(FALLBACK_IDX[key])
                if r:
                    return r
    for t in parts:
        if len(t) > 3 and t in TOKEN_IDX:
            r = pick(TOKEN_IDX[t])
            if r:
                return r
    return None


FALLBACK_IDX = {}
TOKEN_IDX = {}
PLAYER_CLUBS = {}


def build_fallback(idx, con):
    for r in con.execute("SELECT DISTINCT player_id, club_id FROM player_snapshot WHERE club_id IS NOT NULL"):
        PLAYER_CLUBS.setdefault(r[0], set()).add(r[1])
    seen = defaultdict(set)
    toks = defaultdict(set)
    for n, pids in idx.items():
        parts = n.split()
        if len(parts) >= 2:
            seen[(parts[0][0], parts[-1])].update(pids)
            seen[(parts[-1][0], parts[0])].update(pids)
        for t in parts:
            if len(t) > 3:
                toks[t].update(pids)
    for k, v in seen.items():
        if len(v) <= 3:
            FALLBACK_IDX[k] = v
    for t, v in toks.items():
        if len(v) <= 3:
            TOKEN_IDX[t] = v


def elo_and_form(matches):
    """Pre-match Elo + last-5 form per match (computed in date order)."""
    elo = defaultdict(lambda: 1500.0)
    hist = defaultdict(list)  # club -> list of (date, points, gf, ga)
    out = {}
    K, HOME_ADV = 20.0, 60.0
    for m in matches:
        mid, date, h, a, hg, ag = m
        eh, ea = elo[h], elo[a]
        fh = [x for x in hist[h]][-5:]
        fa = [x for x in hist[a]][-5:]
        out[mid] = {
            "elo_home": eh, "elo_away": ea,
            "form_pts_home": sum(p for _, p, *_ in fh) / max(len(fh), 1),
            "form_pts_away": sum(p for _, p, *_ in fa) / max(len(fa), 1),
            "form_gd_home": sum(gf - ga for _, _, gf, ga in fh) / max(len(fh), 1),
            "form_gd_away": sum(gf - ga for _, _, gf, ga in fa) / max(len(fa), 1),
            "rest_home": None, "rest_away": None,
        }
        if hist[h]:
            out[mid]["rest_home"] = (pd.Timestamp(date) - pd.Timestamp(hist[h][-1][0])).days
        if hist[a]:
            out[mid]["rest_away"] = (pd.Timestamp(date) - pd.Timestamp(hist[a][-1][0])).days
        exp_h = 1.0 / (1.0 + 10 ** (-((eh + HOME_ADV) - ea) / 400.0))
        score_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        elo[h] += K * (score_h - exp_h)
        elo[a] -= K * (score_h - exp_h)
        hist[h].append((date, 3 if hg > ag else (1 if hg == ag else 0), hg, ag))
        hist[a].append((date, 3 if ag > hg else (1 if hg == ag else 0), ag, hg))
    return out


def main():
    con = db.connect()
    snaps = load_snapshots(con)
    attrs = load_attrs(con)
    idx, has_snap = name_index(con)
    build_fallback(idx, con)
    print(f"snapshot players: {len(snaps)}, name index: {len(idx)}")

    matches = con.execute(
        "SELECT match_id, match_date, home_club_id, away_club_id, home_goals, away_goals "
        "FROM match ORDER BY match_date").fetchall()
    ctx = elo_and_form([tuple(m) for m in matches])

    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}
    lineups = defaultdict(list)  # (match_id, club_id) -> [(player_id, position)]
    for r in con.execute("SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        lineups[(r[0], r[2])].append((r[1], r[3]))

    rows, total_starters, matched_starters = [], 0, 0
    for m in matches:
        mid, date, hcid, acid, hg, ag = m
        row = {"match_id": mid, "date": date, "home_goals": hg, "away_goals": ag,
               "result": "H" if hg > ag else ("A" if ag > hg else "D"), **ctx[mid]}
        mrow = con.execute("SELECT b365h, b365d, b365a, xg_home, xg_away FROM match WHERE match_id=?",
                           (mid,)).fetchone()
        row.update({"b365h": mrow[0], "b365d": mrow[1], "b365a": mrow[2]})
        ok = True
        for side, cid in (("home", hcid), ("away", acid)):
            xi = lineups.get((mid, cid), [])
            grp_vals = defaultdict(list)   # (pos_group, category) -> values
            cas, pas = [], []
            n_matched = 0
            for pid, pos in xi:
                total_starters += 1
                rpid = resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
                if rpid is None:
                    continue
                snap = latest_before(snaps.get(rpid, []), date)
                if snap is None:
                    continue
                n_matched += 1
                matched_starters += 1
                _, sid, _, ca, pa = snap
                if ca:
                    cas.append(ca)
                if pa:
                    pas.append(pa)
                pg = POS_GROUP.get((pos or " ")[0], "MID")
                for (cat, _an), val in attrs.get(sid, {}).items():
                    grp_vals[(pg, cat)].append(val)
                    grp_vals[("ALL", cat)].append(val)
            row[f"{side}_n_matched"] = n_matched
            row[f"{side}_ca_mean"] = sum(cas) / len(cas) if cas else None
            row[f"{side}_pa_mean"] = sum(pas) / len(pas) if pas else None
            for pg in ("GK", "DEF", "MID", "ATT", "ALL"):
                for cat in ATTR_GROUPS:
                    v = grp_vals.get((pg, cat))
                    row[f"{side}_{pg}_{cat}_mean"] = sum(v) / len(v) if v else None
            if n_matched < 8:
                ok = False
        row["complete"] = ok
        rows.append(row)

    df = pd.DataFrame(rows)
    out = db.ROOT / "data" / "dataset.parquet"
    try:
        df.to_parquet(out)
    except Exception:
        out = db.ROOT / "data" / "dataset.csv"
        df.to_csv(out, index=False)
    cov = matched_starters / max(total_starters, 1)
    print(f"matches: {len(df)}, complete(>=8 matched/side): {int(df['complete'].sum())}")
    print(f"starter->FM join coverage: {cov:.1%} ({matched_starters}/{total_starters})")
    print(f"saved {out}")
    con.close()


if __name__ == "__main__":
    main()
