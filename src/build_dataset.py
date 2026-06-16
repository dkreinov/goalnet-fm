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
    """player_id -> sorted list of (date, snapshot_id, source_name, ca, pa, fm_version_id)."""
    snaps = defaultdict(list)
    for r in con.execute(
            "SELECT ps.player_id, ps.snapshot_date, ps.snapshot_id, s.name, ps.ca, ps.pa, ps.fm_version_id "
            "FROM player_snapshot ps JOIN source s USING(source_id)"):
        snaps[r[0]].append((r[1], r[2], r[3], r[4], r[5], r[6]))
    for v in snaps.values():
        v.sort()
    return snaps


def season_fmv(con):
    """season label -> fm_version_id for that season's FM database (per leagues.SEASON_DB)."""
    import leagues as L
    out = {}
    for label, (_dbid, game, db_version, _date) in L.SEASON_DB.items():
        row = con.execute(
            "SELECT fm_version_id FROM fm_version WHERE game=? AND db_version IS ?",
            (game, db_version)).fetchone()
        if row:
            out[label] = row[0]
    return out


def pick_snapshot(snaps_for_player, target_fmv, season_end):
    """Prefer the snapshot from this season's FM database; else the latest snapshot on/before
    the season end; else the earliest available (so a player only graded in later editions
    still resolves). Returns (date, sid, src, ca, pa, fmv) or None."""
    if not snaps_for_player:
        return None
    for snap in snaps_for_player:
        if snap[5] == target_fmv:
            return snap
    before = [s for s in snaps_for_player if s[0] <= season_end]
    if before:
        return before[-1]
    return snaps_for_player[0]


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


def load_xwalk(con):
    """lineup player_id -> (tuple of FM grade player_ids, confidence) via player_xwalk.
    Only single-ESPN-id players resolve through the crosswalk; collision-merged players
    (>=2 ESPN ids) and unmatched ones return nothing here and fall back to name-resolve."""
    if not con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='player_xwalk'").fetchone():
        return {}
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    pid_eids = defaultdict(list)
    for eid, pid in con.execute(
            "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (espn_sid,)):
        pid_eids[pid].append(eid)
    xw = {r[0]: (r[1], r[2]) for r in con.execute(
        "SELECT espn_player_id, fm_uid, confidence FROM player_xwalk WHERE fm_uid IS NOT NULL")}
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    uid_pids = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute(
                "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pids[uid].add(pid)
    out = {}
    for pid, eids in pid_eids.items():
        if len(eids) != 1:
            continue
        fm_uid, conf = xw.get(eids[0], (None, None))
        if fm_uid and uid_pids.get(fm_uid):
            out[pid] = (tuple(sorted(uid_pids[fm_uid])), conf)
    return out


def load_roster(con):
    """match-level roster-constrained links: (match_id, player_id) -> (fm_uid, confidence),
    plus fm_uid -> grade player_ids. Empty if roster_match.py hasn't run."""
    if not con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='match_grade_link'").fetchone():
        return {}, {}
    rmap = {(mid, pid): (uid, conf) for mid, pid, uid, conf in con.execute(
        "SELECT match_id, player_id, fm_uid, confidence FROM match_grade_link WHERE fm_uid IS NOT NULL")}
    fm_src = [r[0] for r in con.execute(
        "SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    uid_pids = defaultdict(set)
    for sid in fm_src:
        for uid, pid in con.execute(
                "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_pids[uid].add(pid)
    return rmap, uid_pids


SEASON_END = {"2020-21": "2021-06-30", "2021-22": "2022-06-30", "2022-23": "2023-06-30",
              "2023-24": "2024-06-30", "2024-25": "2025-06-30", "2025-26": "2026-06-30"}


def main():
    import features
    con = db.connect()
    record_unmatched = "--no-flag" not in sys.argv
    snaps = load_snapshots(con)
    attrs = load_attrs(con)
    idx, has_snap = name_index(con)
    build_fallback(idx, con)
    xwalk = load_xwalk(con)
    roster, ruid_pids = load_roster(con)
    sfmv = season_fmv(con)
    print(f"snapshot players: {len(snaps)}, name index: {len(idx)}, xwalk links: {len(xwalk)}, "
          f"roster links: {len(roster)}, season->fmv: {sfmv}")
    conf_tally = defaultdict(int)

    matches = con.execute(
        """SELECT m.match_id, m.match_date, m.home_club_id, m.away_club_id, m.home_goals,
                  m.away_goals, se.label, COALESCE(co.name,'?'), m.competition_id, m.referee
           FROM match m JOIN season se ON se.season_id=m.season_id
           LEFT JOIN competition co ON co.competition_id=m.competition_id
           ORDER BY m.match_date""").fetchall()
    ctx = elo_and_form([(r[0], r[1], r[2], r[3], r[4], r[5]) for r in matches])
    squad = features.squad_strength(con)
    refstrict = features.referee_strictness(con)
    seqf = features.sequential_features(
        [(r[0], r[1], r[8], r[6], r[2], r[3], r[4], r[5]) for r in matches])

    pname = {r[0]: r[1] for r in con.execute("SELECT player_id, norm_name FROM player")}
    cname = {r[0]: r[1] for r in con.execute("SELECT club_id, name FROM club")}
    lineups = defaultdict(list)
    for r in con.execute("SELECT match_id, player_id, club_id, position FROM match_player WHERE started=1"):
        lineups[(r[0], r[2])].append((r[1], r[3]))

    rows, total_starters, matched_starters = [], 0, 0
    for mid, date, hcid, acid, hg, ag, season, comp, comp_id, referee in matches:
        target_fmv = sfmv.get(season)
        season_end = SEASON_END.get(season, date)
        row = {"match_id": mid, "date": date, "season": season, "competition": comp,
               "home_goals": hg, "away_goals": ag,
               "result": "H" if hg > ag else ("A" if ag > hg else "D"),
               **ctx[mid], **seqf.get(mid, {}),
               "ref_cards_avg": refstrict.get(referee)}
        mrow = con.execute("SELECT b365h, b365d, b365a, xg_home, xg_away FROM match WHERE match_id=?",
                           (mid,)).fetchone()
        row.update({"b365h": mrow[0], "b365d": mrow[1], "b365a": mrow[2]})
        # full-squad strength per side (depth beyond the starting XI), from this season's FM db
        for side, cid in (("home", hcid), ("away", acid)):
            sq = squad.get((cid, target_fmv), {})
            for k in ("squad_ca_mean", "squad_ca_max", "squad_ca_top11", "squad_size",
                      "squad_value_total"):
                row[f"{side}_{k}"] = sq.get(k)
        ok = True
        for side, cid in (("home", hcid), ("away", acid)):
            xi = lineups.get((mid, cid), [])
            grp_vals = defaultdict(list)
            cas, pas = [], []
            n_matched = 0
            conf_counts = defaultdict(int)
            for pid, pos in xi:
                total_starters += 1
                conf = None
                snap = None
                gp = xwalk.get(pid)                       # DOB/club-anchored crosswalk first
                if gp:
                    gps, conf = gp
                    union = []
                    for p in gps:
                        union.extend(snaps.get(p, []))
                    union.sort()
                    snap = pick_snapshot(union, target_fmv, season_end)
                if snap is None:                          # fallback: legacy name-resolve (no coverage loss)
                    rpid = resolve(pid, pname.get(pid, ""), cid, has_snap, idx)
                    snap = pick_snapshot(snaps.get(rpid, []), target_fmv, season_end) if rpid else None
                    conf = "fallback" if snap is not None else None
                if snap is None:                          # roster-constrained club-squad assignment
                    rl = roster.get((mid, pid))
                    if rl:
                        ruid, rconf = rl
                        union = []
                        for p in ruid_pids.get(ruid, ()):
                            union.extend(snaps.get(p, []))
                        union.sort()
                        snap = pick_snapshot(union, target_fmv, season_end)
                        conf = ("roster_" + rconf) if snap is not None else None
                if snap is None:
                    if record_unmatched:
                        db.record_unmatched(con, "build-dataset", pname.get(pid, str(pid)),
                                            club=cname.get(cid), competition=comp, context=season)
                    continue
                conf_counts[conf] += 1
                conf_tally[conf] += 1
                n_matched += 1
                matched_starters += 1
                _, sid, _, ca, pa, _ = snap
                if ca:
                    cas.append(ca)
                if pa:
                    pas.append(pa)
                pg = POS_GROUP.get((pos or " ")[0], "MID")
                for (cat, _an), val in attrs.get(sid, {}).items():
                    grp_vals[(pg, cat)].append(val)
                    grp_vals[("ALL", cat)].append(val)
            row[f"{side}_n_matched"] = n_matched
            row[f"{side}_n_confirmed"] = conf_counts.get("confirmed", 0)
            row[f"{side}_n_high"] = conf_counts.get("high", 0)
            row[f"{side}_n_fallback"] = conf_counts.get("fallback", 0)
            row[f"{side}_n_roster_high"] = conf_counts.get("roster_high", 0)
            row[f"{side}_n_roster_medium"] = conf_counts.get("roster_medium", 0)
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
    con.commit()

    df = pd.DataFrame(rows)
    out = db.ROOT / "data" / "dataset.parquet"
    try:
        df.to_parquet(out)
    except Exception:
        out = db.ROOT / "data" / "dataset.csv"
        df.to_csv(out, index=False)
    cov = matched_starters / max(total_starters, 1)
    print(f"matches: {len(df)}, complete(>=8 matched/side): {int(df['complete'].sum())}")
    print(f"by season complete:")
    print(df[df.complete].groupby('season').size().to_string())
    print(f"starter->FM join coverage: {cov:.1%} ({matched_starters}/{total_starters})")
    tot_conf = sum(conf_tally.values()) or 1
    print("grade-match confidence (of matched starters):")
    for t in ("confirmed", "high", "fallback", "roster_high", "roster_medium"):
        print(f"  {t:14}: {conf_tally.get(t,0):,} ({100*conf_tally.get(t,0)/tot_conf:.0f}%)")
    nun = con.execute("SELECT COUNT(*) FROM unmatched_name WHERE resolved_player_id IS NULL").fetchone()[0]
    print(f"unmatched names flagged: {nun}")
    print(f"saved {out}")
    con.close()


if __name__ == "__main__":
    main()
