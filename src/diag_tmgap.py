"""Step 1: quantify how many missing starters the closed-set can 'pinpoint' using FINE positions.
ESPN and FM both store detailed positions (CD-L/RB/CM-R/AM vs DC/DL/DR/MC/AMC/ST). With ~10 fine
buckets a club squad has only 1-2 players per slot, so a missing starter is often UNIQUELY the one
unassigned squad member of that slot. Read-only.
"""
import sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db
import roster_match as RM
from diag_roster import EXCL


def fpos(p):
    """Fine position bucket shared by ESPN + FM position strings."""
    if not p:
        return "?"
    p = p.upper().replace(",", " ")
    t = p.split()
    p0 = t[0] if t else ""
    if p0.startswith("G"):
        return "GK"
    if p0 in ("LB", "DL", "WBL", "LWB"):
        return "DL"
    if p0 in ("RB", "DR", "WBR", "RWB"):
        return "DR"
    if p0 in ("CD", "DC", "SW", "CB", "RCB", "LCB", "D"):
        return "DC"
    if p0.startswith("CD-L") or p0.startswith("DL"):
        return "DL"
    if p0.startswith("CD-R") or p0.startswith("DR"):
        return "DR"
    if p0.startswith("D"):
        return "DC"
    if p0 in ("DM", "CDM"):
        return "DM"
    if p0 in ("LM", "ML"):
        return "ML"
    if p0 in ("RM", "MR"):
        return "MR"
    if p0.startswith("AM"):
        return "AM"
    if p0 in ("CM", "MC", "M") or p0.startswith("CM"):
        return "MC"
    if p0.startswith("M"):
        return "MC"
    if p0 in ("CF", "ST", "F", "LF", "RF", "RCF", "LCF", "FW") or p0.startswith("CF") or p0.startswith("F") or p0.startswith("S"):
        return "ST"
    return "?"


def main():
    con = db.connect()
    D = RM.load(con)
    # fine position bucket per FM uid (mode over its snapshots)
    fm_src = [r[0] for r in con.execute("SELECT source_id FROM source WHERE name IN ('fminside','kaggle','futek')")]
    pid_uid = {}
    for sid in fm_src:
        for u, p in con.execute("SELECT source_player_id, player_id FROM player_source_id WHERE source_id=?", (sid,)):
            pid_uid[p] = u
    uid_pos = defaultdict(Counter)
    for pid, pos in con.execute("SELECT player_id, position FROM player_snapshot WHERE position IS NOT NULL"):
        u = pid_uid.get(pid)
        if u:
            uid_pos[u][fpos(pos)] += 1
    uid_fb = {u: c.most_common(1)[0][0] for u, c in uid_pos.items()}

    miss_matches = defaultdict(set)
    cand_sets = defaultdict(list)
    # per (espn_club, season): union squad, globally-assigned uids, unmatched espn players(+fpos)
    cs_squad = defaultdict(set)
    cs_assigned = defaultdict(set)
    cs_unmatched = defaultdict(dict)   # (club,lab) -> {pid: fpos}
    sides = 0
    for (mid, ecid), pls in D["starters"].items():
        if D["comp_name"].get(D["comp_of"].get(mid)) in EXCL:
            continue
        lab = D["season_of"].get(mid)
        if not lab:
            continue
        squad, assigned = RM._squad_of(D, mid, ecid, lab)
        cs_squad[(ecid, lab)] |= squad
        cs_assigned[(ecid, lab)] |= assigned
        unmatched = [(pid, fpos(pos)) for pid, pos in pls if not (D["resolved"].get(pid))]
        if not unmatched or len(unmatched) > 2:
            continue
        sides += 1
        free = squad - assigned
        for pid, fb in unmatched:
            miss_matches[pid].add(mid)
            cs_unmatched[(ecid, lab)][pid] = fb
            cands = {u for u in free if uid_fb.get(u) == fb} or set(free)
            cand_sets[pid].append(cands)

    # GLOBAL greedy elimination per (club,season): iteratively pin any unmatched ESPN player who has
    # exactly one compatible unassigned squad member; each pin removes that member, possibly forcing more.
    global_pinned = set()
    for key, ump in cs_unmatched.items():
        free = cs_squad[key] - cs_assigned[key]
        ump = dict(ump)
        changed = True
        while changed and ump:
            changed = False
            for pid, fb in list(ump.items()):
                comp = [u for u in free if uid_fb.get(u) == fb]
                if len(comp) == 1:
                    free.discard(comp[0]); del ump[pid]; global_pinned.add(pid); changed = True

    forced = inter1 = 0
    for pid, sets in cand_sets.items():
        if any(len(s) == 1 for s in sets):
            forced += 1
        else:
            inter = set.intersection(*sets) if sets else set()
            if len(inter) == 1:
                inter1 += 1
    gated = sum(len(v) for v in miss_matches.values())
    print(f"not-ready target match-sides (missing 1-2): {sides:,}")
    print(f"distinct missing STARTERS: {len(miss_matches):,}  gating {gated:,} match-sides")
    print(f"  FORCED (unique fine-position slot in >=1 match): {forced:,}")
    print(f"  CROSS-MATCH unique (intersection==1):            {inter1:,}")
    print(f"  => per-player pinpoint: {forced + inter1:,} of {len(miss_matches):,} distinct players")
    gp_matches = sum(len(miss_matches[p]) for p in global_pinned)
    print(f"  GLOBAL greedy elimination pins: {len(global_pinned):,} players, gating {gp_matches:,} match-sides")
    lev = Counter({pid: len(v) for pid, v in miss_matches.items()})
    print("\n  top high-leverage missing players:")
    for pid, n in lev.most_common(8):
        print(f"    {D['pname'].get(pid,'?')[:24]:24} gates {n:3d} matches")


if __name__ == "__main__":
    main()
