"""Full ESPN-coverage sweep: for every ESPN soccer domestic-league code we DON'T already have,
report whether ESPN serves LINEUPS (the gate for adding it). Writes data/_espn_sweep.txt.

Usage: python D:/Programming/claude/FM/src/probe_espn_leagues.py
"""
import re
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fetch
import leagues as L

SB = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/scoreboard?dates={win}"
SUM = "https://site.api.espn.com/apis/site/v2/sports/soccer/{code}/summary?event={eid}"
CORE = "https://sports.core.api.espn.com/v2/sports/soccer/leagues?limit=1000"
# tile windows across recent seasons (Euro Aug-May + calendar leagues)
WINDOWS = ["20240801-20241215", "20240401-20240915", "20231001-20240315",
           "20250201-20250715", "20230801-20231215"]


def all_slugs():
    j = json.loads(fetch.get(CORE, min_delay=0.3, timeout=40))
    out = []
    for it in j.get("items", []):
        m = re.search(r"/leagues/([^/?]+)", it.get("$ref", ""))
        if m:
            out.append(m.group(1))
    return out


def have_codes():
    codes = set()
    for lg in L.LEAGUES + L.EXTRA_LEAGUES + L.UEFA_CUPS:
        if lg.get("espn"):
            codes.add(lg["espn"])
    return codes


def probe(code):
    eid = None
    for win in WINDOWS:
        try:
            sb = json.loads(fetch.get(SB.format(code=code, win=win), min_delay=0.2, timeout=25))
        except Exception:
            continue
        for ev in sb.get("events", []):
            if ((ev.get("status") or {}).get("type") or {}).get("completed"):
                eid = ev["id"]; break
        if eid:
            break
    if not eid:
        return "no-espn", 0
    try:
        s = json.loads(fetch.get(SUM.format(code=code, eid=eid), min_delay=0.2, timeout=25))
    except Exception:
        return "summary-err", 0
    rs = s.get("rosters", [])
    if rs and any(r.get("roster") for r in rs):
        ns = sum(sum(1 for e in r.get("roster", []) if e.get("starter")) for r in rs)
        return "LINEUPS", ns
    return "no-lineups", 0


def main():
    slugs = all_slugs()
    have = have_codes()
    # domestic leagues only: code like 'rou.1' / 'ned.2' (country.tier). Exclude cups/intl/women/youth.
    cand = [s for s in slugs if re.fullmatch(r"[a-z]{2,4}\.\d+", s) and s not in have
            and not s.startswith(("fifa", "uefa", "concacaf", "conmebol", "afc", "caf", "ofc"))]
    print(f"{len(slugs)} ESPN soccer slugs; {len(have)} already ours; {len(cand)} domestic-league candidates to probe\n", flush=True)
    addable = []
    results = []
    for i, code in enumerate(cand):
        status, ns = probe(code)
        line = f"{code:10} {status:12} {('('+str(ns)+' starters)') if ns else ''}"
        print(f"  [{i+1}/{len(cand)}] {line}", flush=True)
        results.append(line)
        if status == "LINEUPS":
            addable.append((code, ns))
    print("\n==== ADDABLE (ESPN serves lineups) ====", flush=True)
    for code, ns in addable:
        print(f"  {code}  ({ns} starters)", flush=True)
    Path("data/_espn_sweep.txt").write_text("\n".join(results) + "\n\nADDABLE:\n" +
                                            "\n".join(f"{c} ({n})" for c, n in addable))


if __name__ == "__main__":
    main()
