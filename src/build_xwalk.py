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
from collections import defaultdict, Counter
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


def posbucket(p):
    """Coarse ESPN/FM position bucket, for breaking same-name same-club ties (a GK vs an outfielder)."""
    if not p:
        return "?"
    p = p.upper()
    if p.startswith("G"):
        return "GK"
    if p[0] == "D" or p in ("CB", "LB", "RB", "RWB", "LWB"):
        return "DEF"
    if p[0] == "M" or p in ("DM", "AM", "CM", "LM", "RM"):
        return "MID"
    if p[0] in ("F", "W", "S") or p in ("ST", "CF", "LW", "RW"):
        return "ATT"
    return "?"


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
                    # Globally-unique FM name but the DOBs differ. ESPN DOBs are frequently wrong by
                    # days/months/years (Giovanni Simeone, Janni Serra, Kevin Paredes all = real same
                    # player, noisy DOB). A unique name is strong evidence on its own — link as 'high'
                    # (NOT 'confirmed', so the DOB-consistency authenticity test is unaffected) rather
                    # than discarding the player. Two distinct real footballers sharing a globally-unique
                    # full name AND both present in our ESPN+FM data is vanishingly unlikely.
                    uid, conf, method = u, "high", "name_unique_dob_differs"
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

    # --- club-season squad disambiguation: the existing name+club tier compares ESPN club_ids to FM
    # grade club_ids (different id-spaces), so it rarely fires. Build a bridge ESPN club -> FM grade-club
    # from the already-confirmed/high links, then resolve 'ambiguous' shared names by which candidate is
    # actually in the player's club squad. Within a club a name is almost always unique.
    # ITERATIVE: in weakly-seeded leagues (Brazil/Portugal: sparse/wrong ESPN DOBs -> few confirmed seeds)
    # one pass barely fires. Each round of newly-confirmed links seeds MORE bridge, enabling the next round,
    # so the disambiguation cascades. Still club-guarded + unique-candidate + DOB-safe + test_xwalk-gated. ---
    # position buckets for breaking cmatch>1 ties safely (same name + same club + a GK vs an outfielder)
    espn_pos_cnt = defaultdict(Counter)
    for ppid, pos in con.execute("SELECT player_id, position FROM match_player WHERE position IS NOT NULL"):
        espn_pos_cnt[ppid][posbucket(pos)] += 1
    espn_posb = {p: c.most_common(1)[0][0] for p, c in espn_pos_cnt.items()}
    snap_pos_cnt = defaultdict(Counter)
    for ppid, pos in con.execute("SELECT player_id, position FROM player_snapshot WHERE position IS NOT NULL"):
        snap_pos_cnt[ppid][posbucket(pos)] += 1
    uid_posb = {}
    for u, pids in uid_to_gradepids.items():
        c = Counter()
        for p in pids:
            c += snap_pos_cnt.get(p, Counter())
        if c:
            uid_posb[u] = {b for b, _ in c.most_common(2)}   # FM lists up to 2 main buckets

    def build_bridge():
        eclub_gc = defaultdict(Counter)
        for eid, pid, uid, fpid, conf, method in rows:
            if uid and conf in ("confirmed", "high"):
                for ec in pid_match_clubs.get(pid, ()):
                    for gc in fm_clubs.get(uid, ()):
                        eclub_gc[ec][gc] += 1
        return {ec: ({c for c, n in cnt.items() if n >= 2} or {cnt.most_common(1)[0][0]})
                for ec, cnt in eclub_gc.items()}

    total_upgraded = 0
    eclub_to_g = {}
    for _round in range(12):
        eclub_to_g = build_bridge()
        upgraded = 0
        for i, (eid, pid, uid, fpid, conf, method) in enumerate(rows):
            if uid is not None or conf != "ambiguous":
                continue
            cands = name_to_uids.get(xnorm(espn_name_si.get(eid) or pid_name.get(pid) or ""), [])
            if len(cands) < 2:
                continue
            bridged = set()
            for ec in pid_match_clubs.get(pid, set()):
                bridged |= eclub_to_g.get(ec, set())
            cmatch = [u for u in cands if fm_clubs.get(u, set()) & bridged]
            if not cmatch:
                continue
            method2 = "name+squad"
            if len(cmatch) > 1:
                # several same-name players bridge to this club -> break the tie by DOB, then position;
                # if neither isolates exactly one candidate, leave it ambiguous (never guess).
                edob = espn_dob.get(eid)
                dm = [u for u in cmatch if edob and fm_dob.get(u) and dob_close(edob, fm_dob[u])]
                if len(dm) == 1:
                    cmatch = dm
                else:
                    epb = espn_posb.get(pid)
                    pm = [u for u in cmatch if epb and epb != "?" and epb in uid_posb.get(u, set())]
                    if len(pm) == 1:
                        cmatch, method2 = pm, "name+squad+pos"
                    else:
                        continue
            u = cmatch[0]
            edob = espn_dob.get(eid)
            if edob and fm_dob.get(u) and not dob_close(edob, fm_dob[u]):   # keep DOB safety
                continue
            fp = sorted(uid_to_gradepids[u])[0] if uid_to_gradepids.get(u) else None
            rows[i] = (eid, pid, u, fp, "high", method2)
            tiers["ambiguous"] -= 1; tiers["high"] += 1; upgraded += 1
        total_upgraded += upgraded
        if upgraded == 0:
            break
    eclub_to_g = build_bridge()   # final bridge for the fuzzy pass below
    print(f"club-squad disambiguation: upgraded {total_upgraded:,} ambiguous -> high (iterative, {_round+1} rounds)")

    # --- club-anchored fuzzy/alias pass: for ESPN starters STILL unmatched/ambiguous, restrict to the
    # bridged club's grade squad and accept a SINGLE close fuzzy/alias name match. Club-first is what makes
    # fuzzy safe — within one squad a near-name is almost always the same person (typos, mononyms, name
    # changes, hyphenation: "Manafá"->"Wilson Manafá", "Tomás Carbonell"). Linked 'high'/'fuzzy+club' (never
    # 'confirmed'); the squad restriction IS the authenticity guard. Requires exactly one squad member to
    # clear the ratio threshold, else skip (never guess between two similar names in a squad). ---
    import difflib
    gradeclub_uids = defaultdict(set)
    for guid, gclubs in fm_clubs.items():
        if uid_to_gradepids.get(guid):
            for gc in gclubs:
                gradeclub_uids[gc].add(guid)

    def _squad_names(u):
        ns = {xnorm(fm_formal.get(u, ""))}
        for p in uid_to_gradepids.get(u, ()):
            ns.add(xnorm(pid_name.get(p, "")))
        return {n for n in ns if n}

    fuzzy_up = 0
    for i, (eid, pid, uid, fpid, conf, method) in enumerate(rows):
        if uid is not None or conf not in ("ambiguous", "unmatched"):
            continue
        nn = xnorm(espn_name_si.get(eid) or pid_name.get(pid) or "")
        if len(nn) < 4:
            continue
        bridged = set()
        for ec in pid_match_clubs.get(pid, set()):
            bridged |= eclub_to_g.get(ec, set())
        if not bridged:
            continue
        # pre-filter: only squad members sharing a distinctive (>=4 char) token with the ESPN name are
        # plausible fuzzy matches -> reuse the global tok_to_uids index, then difflib only those (fast).
        tokcand = set()
        for t in nn.split():
            if len(t) >= 4:
                tokcand |= tok_to_uids.get(t, set())
        if not tokcand:
            continue
        squad = set().union(*(gradeclub_uids.get(gc, set()) for gc in bridged)) if bridged else set()
        cand_pool = tokcand & squad
        scored = []
        for u in cand_pool:
            best = max((difflib.SequenceMatcher(None, nn, un).ratio() for un in _squad_names(u)), default=0.0)
            if best >= 0.82:
                scored.append((best, u))
        cand_uids = {u for _, u in scored}
        if len(cand_uids) != 1:
            continue
        u = next(iter(cand_uids))
        fp = sorted(uid_to_gradepids[u])[0] if uid_to_gradepids.get(u) else None
        rows[i] = (eid, pid, u, fp, "high", "fuzzy+club")
        tiers[conf] -= 1; tiers["high"] += 1; fuzzy_up += 1
    print(f"fuzzy+club alias pass: recovered {fuzzy_up:,} unmatched/ambiguous -> high")

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
