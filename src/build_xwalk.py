"""Non-destructive 1:1 identity crosswalk: ESPN lineup player -> exactly one FM player (FM UID).

Tiered matching (never mutates player/match_player/player_snapshot):
  confirmed : name + DOB exact (or ±1 day); or DOB uniquely disambiguates a shared name
  high      : name globally unique in FM DB (no DOB conflict); or shared name uniquely resolved by club
  medium    : single FM name candidate, no DOB/club to confirm (name-only)
  ambiguous : shared name, no discriminator resolves to one  -> fm_uid NULL, flagged
  unmatched : no FM name candidate                            -> fm_uid NULL, flagged

Writes player_xwalk. Re-runnable (drops+rebuilds). Works on partial ESPN DOB (name+club covers most).
Usage: python D:/Programming/claude/FM/src/build_xwalk.py
"""
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db

FM_SOURCES = ("fminside", "kaggle", "futek")


def dob_close(a, b):
    """True if two ISO yyyy-mm-dd dates are equal or within 1 day (timezone artifacts)."""
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        da = date.fromisoformat(a[:10]); db_ = date.fromisoformat(b[:10])
        return abs((da - db_).days) <= 1
    except ValueError:
        return False


# transliterate characters that NFKD doesn't decompose, so "Groß"->"gross", "Søyland"->"soyland"
_TRANSLIT = str.maketrans({"ß": "ss", "ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "æ": "ae",
                           "Æ": "ae", "œ": "oe", "đ": "d", "Đ": "d", "ð": "d", "þ": "th"})


def xnorm(s):
    return db.norm((s or "").translate(_TRANSLIT))


def main():
    con = db.connect()
    espn_sid = con.execute("SELECT source_id FROM source WHERE name='espn'").fetchone()[0]
    fmuid_sid = con.execute("SELECT source_id FROM source WHERE name='fm-uid'").fetchone()[0]
    fm_src_ids = [r[0] for r in con.execute(
        f"SELECT source_id FROM source WHERE name IN ({','.join('?'*len(FM_SOURCES))})", FM_SOURCES)]

    con.execute("DROP TABLE IF EXISTS player_xwalk")
    con.execute("""CREATE TABLE player_xwalk (
        espn_player_id TEXT PRIMARY KEY,
        espn_player_pid INTEGER,
        fm_uid TEXT,
        fm_player_id INTEGER,
        confidence TEXT NOT NULL,
        method TEXT NOT NULL)""")

    # --- FM side: uid -> dob, uid -> grade player_ids -> club_ids ---
    fm_dob = {}
    fm_formal = {}
    for uid, name, dob in con.execute(
            f"SELECT source_player_id, name, dob FROM source_identity WHERE source_id={fmuid_sid}"):
        fm_dob[uid] = dob
        fm_formal[uid] = name
    uid_to_gradepids = defaultdict(set)
    for sid in fm_src_ids:
        for uid, pid in con.execute(
                "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            uid_to_gradepids[uid].add(pid)
    # name index keyed on UID, unioning BOTH naming systems:
    #   - Kaggle formal name (source_identity), covers all 274k UIDs
    #   - common names on the grade-linked player rows (fminside/futek/ESPN-merged), covers graded UIDs
    # so ESPN common names ("Casemiro") and FM formal names both resolve to the right UID.
    pid_name = {pid: name for pid, name in con.execute("SELECT player_id, name FROM player")}
    name_to_uids = defaultdict(set)
    tok_to_uids = defaultdict(set)   # distinctive token (len>=4) -> uids, for mononym/nickname matching

    def index_name(uid, raw):
        n = xnorm(raw)
        if not n:
            return
        name_to_uids[n].add(uid)
        for t in n.split():
            if len(t) >= 4:
                tok_to_uids[t].add(uid)

    for uid, formal in fm_formal.items():
        index_name(uid, formal)
    for uid, pids in uid_to_gradepids.items():
        for p in pids:
            index_name(uid, pid_name.get(p))
    name_to_uids = {k: list(v) for k, v in name_to_uids.items()}
    # keep all tokens (even common first names like "gabriel" — needed for mononym FM records);
    # the strict club-then-DOB filter below resolves to exactly one or stays flagged, so commonness
    # is safe. tok_to_uids stays a dict of sets.
    tok_to_uids = dict(tok_to_uids)
    pid_clubs = defaultdict(set)
    for pid, cid in con.execute(
            "SELECT player_id, club_id FROM player_snapshot WHERE club_id IS NOT NULL"):
        pid_clubs[pid].add(cid)
    fm_clubs = {uid: set().union(*(pid_clubs[p] for p in pids)) if pids else set()
                for uid, pids in uid_to_gradepids.items()}

    # --- ESPN side: espn_id -> name/dob/clubs ---
    espn_name_si = {}; espn_dob = {}
    for eid, name, dob in con.execute(
            f"SELECT source_player_id, name, dob FROM source_identity WHERE source_id={espn_sid}"):
        espn_name_si[eid] = name; espn_dob[eid] = dob
    # espn_id -> player_id (fallback name comes from pid_name built above)
    espn_pid = {}
    for eid, pid in con.execute(
            "SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (espn_sid,)):
        espn_pid[eid] = pid
    # espn player clubs from lineup appearances (by their player_id)
    pid_match_clubs = defaultdict(set)
    for pid, cid in con.execute("SELECT DISTINCT player_id, club_id FROM match_player"):
        pid_match_clubs[pid].add(cid)

    # learned synonyms (roster-discovered ESPN->FM links with different spelling): seed as confirmed,
    # but only when DOB is consistent (or absent on one side) so authenticity stays intact.
    learned = {}
    if con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='learned_alias'").fetchone():
        for eid, uid in con.execute("SELECT espn_player_id, fm_uid FROM learned_alias WHERE fm_uid IS NOT NULL"):
            learned[eid] = uid

    tiers = defaultdict(int)
    rows = []
    for eid, pid in espn_pid.items():
        raw = espn_name_si.get(eid) or pid_name.get(pid) or ""
        nn = xnorm(raw)
        edob = espn_dob.get(eid)
        eclubs = pid_match_clubs.get(pid, set())
        cands = name_to_uids.get(nn, [])

        uid = None; conf = "unmatched"; method = "no_fm_name"
        lu = learned.get(eid)
        if lu and not (edob and fm_dob.get(lu) and not dob_close(edob, fm_dob[lu])):
            # learned synonym, no DOB conflict -> accept directly (confirmed if DOB confirms it)
            uid = lu
            conf = "confirmed" if (edob and fm_dob.get(lu) and dob_close(edob, fm_dob[lu])) else "high"
            method = "learned_alias"
            fpid = sorted(uid_to_gradepids[uid])[0] if uid_to_gradepids.get(uid) else None
            tiers[conf] += 1
            rows.append((eid, pid, uid, fpid, conf, method))
            continue
        if len(cands) == 1:
            u = cands[0]
            if edob and fm_dob.get(u):
                if dob_close(edob, fm_dob[u]):
                    uid, conf, method = u, "confirmed", "name+dob"
                else:
                    uid, conf, method = None, "unmatched", "name_dob_conflict"
            else:
                uid, conf, method = u, "high", "name_unique"
        elif len(cands) >= 2:
            dmatch = [u for u in cands if edob and fm_dob.get(u) and dob_close(edob, fm_dob[u])] if edob else []
            if len(dmatch) == 1:
                uid, conf, method = dmatch[0], "confirmed", "dob_among_shared"
            else:
                cmatch = [u for u in cands if eclubs & fm_clubs.get(u, set())]
                if len(cmatch) == 1:
                    uid, conf, method = cmatch[0], "high", "name+club"
                elif len(dmatch) > 1:
                    uid, conf, method = None, "ambiguous", "multi_dob_match"
                else:
                    uid, conf, method = None, "ambiguous", "shared_name_unresolved"

        # token-subset fallback for mononyms ("Alisson Becker"->"Alisson"), extra names, nicknames
        # ("Andy Robertson"->surname "Robertson") — disambiguated strictly by club, then DOB.
        if uid is None and conf in ("unmatched", "ambiguous"):
            cand2 = set()
            for t in nn.split():
                if len(t) >= 4:
                    cand2 |= tok_to_uids.get(t, set())
            if cand2:
                cmatch = [u for u in cand2 if eclubs & fm_clubs.get(u, set())]
                if len(cmatch) == 1:
                    u = cmatch[0]
                    if edob and fm_dob.get(u) and dob_close(edob, fm_dob[u]):
                        uid, conf, method = u, "confirmed", "token+club+dob"
                    else:
                        uid, conf, method = u, "high", "token+club"
                elif len(cmatch) > 1:
                    dm = [u for u in cmatch if edob and fm_dob.get(u) and dob_close(edob, fm_dob[u])]
                    if len(dm) == 1:
                        uid, conf, method = dm[0], "confirmed", "token+club+dob"
                    else:
                        conf, method = "ambiguous", "token_club_multi"
                else:
                    dm = [u for u in cand2 if edob and fm_dob.get(u) and dob_close(edob, fm_dob[u])]
                    if len(dm) == 1:
                        uid, conf, method = dm[0], "high", "token+dob"
        # representative grade player_id (None if this uid has no grades yet)
        fpid = None
        if uid and uid_to_gradepids.get(uid):
            fpid = sorted(uid_to_gradepids[uid])[0]
        tiers[conf] += 1
        rows.append((eid, pid, uid, fpid, conf, method))

    con.executemany(
        "INSERT INTO player_xwalk (espn_player_id, espn_player_pid, fm_uid, fm_player_id, confidence, method) "
        "VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    print(f"player_xwalk built: {len(rows):,} ESPN players")
    for t in ("confirmed", "high", "medium", "ambiguous", "unmatched"):
        print(f"  {t:10}: {tiers.get(t,0):,}")
    linked = sum(1 for r in rows if r[2])
    with_grades = sum(1 for r in rows if r[3])
    print(f"  -> linked to an FM UID: {linked:,} ({100*linked/len(rows):.0f}%); of those with grades now: {with_grades:,}")
    con.close()


if __name__ == "__main__":
    main()
