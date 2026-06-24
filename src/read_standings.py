"""Read the LIVE league standings from the Friends app (Supabase) and write standings.json for the risk
decision. Reuses auto_bet's stored session (refresh_token in wc_bet_auth.json) — NO password, READ-ONLY (no
writes to the app). Computes each player's per-game points/exacts from all picks + finished fixtures, applying
the knockout multipliers (group1 r32x2 r16x4 qf8 sf16 final32). Attaches each player's locked futures (winner
/ top-scorer) bonus picks for the effective-standing calc. Usage: python src/read_standings.py
"""
import sys, json
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import auto_bet as ab

MULT = {"group": 1, "r32": 2, "r16": 4, "qf": 8, "sf": 16, "final": 32}
# locked futures bonus picks by nickname (winner +50, scorer +30), from the app's standings screen
FUTURES = {
    "RIVAL_3": ("Netherlands", "Haaland"), "RIVAL_4": ("Argentina", "Messi"), "RIVAL_5": ("France", "Mbappé"),
    "RIVAL_6": ("Argentina", "Olise"), "RIVAL_7": ("Spain", "Kane"), "RIVAL_1": ("Spain", "Mbappé"),
    "YOU": ("Spain", "Mbappé"), "RIVAL_2": ("Spain", "Mbappé"), "RIVAL_8": ("Brazil", "Endrick"),
}


def fetch():
    bearer, uid = ab.get_access()
    g = lambda p: ab.api(ab.BASE + p, bearer=bearer)
    fx = {f["id"]: f for f in g("/rest/v1/fixtures?select=id,round,status,home_score,away_score")}
    profs = {p["id"]: p["nickname"] for p in g("/rest/v1/profiles?select=id,nickname")}
    picks = g("/rest/v1/picks?select=user_id,fixture_id,home_score,away_score")
    return uid, fx, profs, picks


def compute(uid, fx, profs, picks):
    st = defaultdict(lambda: {"pts": 0, "ex": 0, "cor": 0, "wr": 0})
    for p in picks:
        f = fx.get(p["fixture_id"])
        if not f or f["status"] != "finished" or f["home_score"] is None:
            continue
        m = MULT.get(f["round"], 1); ph, pa, rh, ra = p["home_score"], p["away_score"], f["home_score"], f["away_score"]
        s = st[p["user_id"]]
        if ph == rh and pa == ra: s["pts"] += 3 * m; s["ex"] += 1
        elif (ph > pa) == (rh > ra) and (ph < pa) == (rh < ra): s["pts"] += 1 * m; s["cor"] += 1
        else: s["wr"] += 1
    rows = []
    for u, s in st.items():
        nick = profs.get(u, u[:8]); win, scr = FUTURES.get(nick, (None, None))
        rows.append({"nick": nick, "user_id": u, "is_me": u == uid, **s, "winner": win, "scorer": scr})
    rows.sort(key=lambda r: (-r["pts"], -r["ex"]))
    return rows


def main():
    uid, fx, profs, picks = fetch()
    rows = compute(uid, fx, profs, picks)
    nfin = sum(1 for f in fx.values() if f["status"] == "finished")
    out = ROOT / "standings.json"
    json.dump({"rows": rows, "finished_fixtures": nfin}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{nfin} finished fixtures; standings -> {out}", flush=True)
    print(f"  {'#':>2} {'nick':12s} {'pts':>4} {'ex':>3} {'cor':>3}  futures", flush=True)
    for i, r in enumerate(rows, 1):
        me = "  <-- you" if r["is_me"] else ""
        fut = f"{r['winner']}/{r['scorer']}" if r["winner"] else "?"
        print(f"  {i:>2} {r['nick']:12s} {r['pts']:>4} {r['ex']:>3} {r['cor']:>3}  {fut}{me}", flush=True)


if __name__ == "__main__":
    main()
