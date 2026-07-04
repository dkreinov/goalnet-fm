#!/usr/bin/env python3
"""Headless auto-fill for the Sport5 'Hevre' fantasy board — the Hevre counterpart of auto_bet.py.

Fetches the board (data.php?type=getMyGuesses), and for every game that is editable (status=="before")
AND kicks off in the future AND has no guess yet, computes the EV-optimal scoreline with the Hevre
odds-points optimizer (sport5_optimizer's model: Elo+Dixon-Coles blended with the de-vigged board odds,
per-round pointsMultplyer/bonusExact), then saves the whole board back (data.php?type=guess).

SET-ONCE: only fills games with no existing guess, so it never churns a pick you set by hand. Gates on
kickoff time (not just status) so it can never overwrite a finished group game.

Auth: Authorization: Bearer <loginToken> (valid ~2 days). On auth failure it refreshes via
data.php?type=requestNewActive {refreshToken,email} -> {token,refreshToken}; if that fails it fires the
same ntfy/Windows alert as the friends bot so you know to re-seed.

Run:  python hevre_bot.py            # live
      python hevre_bot.py --dry      # compute + log only, no writes
"""
import os, sys, json, time, datetime, urllib.request, urllib.error, importlib.util, subprocess

FM = os.path.dirname(os.path.abspath(__file__))
AUTH = os.path.join(FM, "hevre_auth.json")
STATE = os.path.join(FM, "hevre_state.json")
LOG = os.path.join(FM, "hevre_bot.log")
NOTIFY = os.path.join(FM, "wc_notify.json")
HEALTH = os.path.join(FM, "hevre_health.json")
BASE = "https://hevre.sport5.co.il/server/data.php?type="
SKILL = r"C:\Users\youruser\.claude\skills\worldcup-predictor\scripts"
UA = "Mozilla/5.0"
W_MODEL = 0.5                 # blend: model vs de-vigged board odds (matches sport5_optimizer)
RENOTIFY_SEC = 2 * 3600

# Sport5 numeric team code -> FIFA 3-letter (the optimizer/model key)
CODE2FIFA = {
    "155609":"CUW","2080":"URU","2094":"CAN","2100":"JOR","2103":"IRQ","2104":"NZL","2172":"CPV",
    "2186":"QAT","2206":"PAN","2218":"HAI","2545":"MAR","2599":"RSA","2600":"ALG","2604":"COD",
    "2678":"SEN","2878":"TUR","2989":"ARG","2990":"ECU","2991":"NED","2992":"CIV","2993":"MEX",
    "2994":"IRN","2996":"USA","2997":"GHA","2998":"JPN","2999":"AUS","3000":"SUI","3002":"TUN",
    "3003":"ESP","3006":"FRA","3007":"PAR","3078":"KSA","3087":"KOR","3535":"AUT","3536":"BRA",
    "3538":"COL","3543":"BEL","3640":"POR","3770":"ENG","3771":"NOR","3775":"CZE","3777":"SCO",
    "3799":"SWE","3849":"GER","3979":"UZB","4386":"BIH","4435":"CRO","4535":"EGY",
}

def log(m): open(LOG, "a", encoding="utf-8").write(f"{datetime.datetime.now():%Y-%m-%d %H:%M} | {m}\n")

# ---- model (shared Elo + Dixon-Coles from the worldcup-predictor skill) ----
_spec = importlib.util.spec_from_file_location("sm", os.path.join(SKILL, "strong_model.py"))
sm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sm)

def hevre_pick(h, a, pts, mult, bonus):
    """EV-optimal (scoreline) for Hevre odds-points scoring: correct outcome pays the listed odds*mult,
    exact adds +bonus. pts=(home,draw,away) listed odds. Returns (home_score, away_score)."""
    la, lb = sm.lambdas(sm.R[h], sm.R[a], h in sm.HOST, a in sm.HOST)
    P = sm.matrix(la, lb); om = sm.outc(P)
    inv = [1/p for p in pts]; s = sum(inv); oi = {"H":inv[0]/s, "D":inv[1]/s, "A":inv[2]/s}
    ob = {k: W_MODEL*om[k] + (1-W_MODEL)*oi[k] for k in "HDA"}
    best = None
    for o in "HDA":
        sc = {k2: p for k2, p in P.items() if ("H" if k2[0] > k2[1] else ("A" if k2[0] < k2[1] else "D")) == o}
        msc = max(sc, key=sc.get); pex = sc[msc]/om[o]*ob[o] if om[o] > 0 else 0
        ev = ob[o]*pts["HDA".index(o)]*mult + bonus*pex
        if best is None or ev > best[0]: best = (ev, msc)
    return int(best[1][0]), int(best[1][1])

# ---- notify (mirrors auto_bet.health_signal) ----
def _toast(t, m):
    try: subprocess.run(["msg", "*", "/TIME:60", f"{t}: {m}"], timeout=15)
    except Exception: pass

def _ping(url, body=None, headers=None):
    try: urllib.request.urlopen(urllib.request.Request(url, data=(body.encode() if body else b""), headers=headers or {}, method="POST"), timeout=15)
    except Exception: pass

def health_signal(ok, reason=""):
    try: cfg = json.load(open(NOTIFY))
    except Exception: cfg = {}
    try: st = json.load(open(HEALTH))
    except Exception: st = {}
    now = int(time.time()); prev = st.get("status")
    fire = (prev == "fail") if ok else (prev != "fail" or now - st.get("last_notify", 0) >= RENOTIFY_SEC)
    if fire:
        title = "Hevre bot RECOVERED" if ok else "Hevre bot AUTH FAILED"
        body = "session works again" if ok else f"{reason[:120]} — re-seed hevre_auth.json"
        if cfg.get("win_toast", True): _toast(title, body)
        if cfg.get("ntfy_url"): _ping(cfg["ntfy_url"], body, {"Title": title, "Priority": ("default" if ok else "high"), "Tags": ("soccer" if ok else "rotating_light")})
        st["last_notify"] = now
    st["status"] = "ok" if ok else "fail"
    try: json.dump(st, open(HEALTH, "w"))
    except Exception: pass

# ---- API ----
_SESS = ""    # PHPSESSID cookie — requestNewActive (and the API generally) needs it alongside the Bearer

def api(t, bearer, body=None):
    h = {"Authorization": "Bearer " + bearer, "User-Agent": UA, "Content-Type": "application/json"}
    if _SESS: h["Cookie"] = "PHPSESSID=" + _SESS
    r = urllib.request.Request(BASE + t, data=json.dumps(body if body is not None else {}).encode(), headers=h, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=40).read().decode("utf-8", "replace") or "null")

def refresh(a):
    """Trade the refreshToken for a fresh loginToken (requestNewActive). NEEDS the PHPSESSID cookie AND the
    real account email — without both the endpoint returns the app HTML shell (silent auth failure)."""
    global _SESS
    _SESS = a.get("phpsessid", "") or _SESS
    email = (a.get("loginData") or {}).get("email") or a.get("email") or ""
    out = api("requestNewActive", a["refreshToken"], {"refreshToken": a["refreshToken"], "email": email})
    tok = (out or {}).get("token") or (out or {}).get("loginToken")
    if not tok: raise RuntimeError(f"refresh failed (session/email/cookie stale?): {str(out)[:120]}")
    a["loginToken"] = tok
    if (out or {}).get("refreshToken"): a["refreshToken"] = out["refreshToken"]
    json.dump(a, open(AUTH, "w")); return a

def get_user(a):
    """appUserGetUser is the AUTHORITATIVE read: returns the user's actual saved guesses (result1/result2
    = the guess) — NOT getMyGuesses (which is fixtures+real-results). On auth error, refresh once and retry."""
    try:
        u = api("appUserGetUser", a["loginToken"], {"token": a["refreshToken"]})
        if not isinstance(u, dict) or u.get("error") or "guesses" not in u: raise RuntimeError(str(u)[:80])
        return u, a
    except Exception:
        a = refresh(a)
        return api("appUserGetUser", a["loginToken"], {"token": a["refreshToken"]}), a

def round_deadline(rd):
    """A round locks when its FIRST game kicks off (whole-round lock, confirmed empirically: R32 closed
    when RSA-CAN started even though later R32 games were still future). Returns epoch seconds or None."""
    kos = []
    for g in rd.get("games", []):
        try:
            k = float(g.get("beggining")); kos.append(k/1000 if k > 1e12 else k)
        except Exception:
            pass
    return min(kos) if kos else None

def main(dry=False):
    global _SESS
    try: a = json.load(open(AUTH))
    except Exception as e:
        log(f"NO AUTH FILE: {e}"); health_signal(False, "no hevre_auth.json"); return
    _SESS = a.get("phpsessid", "")     # send the session cookie on every API call, not just refresh
    try:
        user, a = get_user(a)
    except Exception as e:
        log(f"AUTH/FETCH FAILED: {e}"); health_signal(False, str(e)); print("auth/fetch failed:", e); return
    guesses = user.get("guesses")
    if not isinstance(guesses, list):
        log("BAD USER PAYLOAD"); health_signal(False, "no guesses[]"); return
    health_signal(True)
    try: state = json.load(open(STATE))
    except Exception: state = {}
    now = time.time(); changed = 0; skipped = []
    for rd in guesses:
        dl = round_deadline(rd)
        if dl is None or dl <= now:                                 # round locked (first game started) — skip
            continue
        mult = float(rd.get("pointsMultplyer") or 1); bonus = float(rd.get("bonusExact") or 4)
        for g in rd.get("games", []):
            if g.get("result1") not in (None, ""): continue          # already guessed -> set-once (won't churn manual picks)
            c1 = (g.get("team1") or {}).get("code"); c2 = (g.get("team2") or {}).get("code")
            h, aw = CODE2FIFA.get(str(c1)), CODE2FIFA.get(str(c2))
            if not (h and aw and h in sm.R and aw in sm.R):
                skipped.append(f"{c1}/{c2}"); continue               # e.g. knockout slot whose team isn't decided yet
            pts = (g.get("ratio1"), g.get("ratio3"), g.get("ratio2"))
            if not all(isinstance(x, (int, float)) and x > 0 for x in pts): continue
            hs, as_ = hevre_pick(h, aw, pts, mult, bonus)
            g["result1"], g["result2"] = hs, as_                     # board-format guess field
            g.setdefault("team1", {})["team1Guessed"] = hs           # directive-format guess field
            g.setdefault("team2", {})["team2Guessed"] = as_
            log(f"{'DRY ' if dry else ''}fill {h} {hs}-{as_} {aw}  [{rd.get('name')}, x{mult:.0f} bonus{bonus:.0f}]")
            changed += 1
    if skipped:
        log(f"skipped (team undecided/unmapped): {','.join(sorted(set(skipped)))}")
    if not changed:
        log("run complete: no open-round games to fill"); print("nothing to fill (no open round)"); return
    if dry:
        log(f"DRY: would fill {changed}"); print(f"DRY: would fill {changed}"); return
    # SAVE then VERIFY against the authoritative read — never trust a 200; confirm the guess actually persisted.
    try:
        api("guess", a["loginToken"], {"data": guesses})
        v, a = get_user(a)
        vg = v.get("guesses") or []
        persisted = sum(1 for rd in vg for g in rd.get("games", [])
                        if g.get("result1") not in (None, "") and round_deadline(rd) and round_deadline(rd) > now)
        if persisted > 0:
            state["last_fill"] = int(now); json.dump(state, open(STATE, "w"))
            log(f"WROTE+verified: {persisted} open-round guesses now saved"); print(f"wrote+verified {persisted}")
        else:
            log(f"SAVE DID NOT PERSIST (changed {changed}, verified 0) — endpoint/format/lock issue")
            health_signal(False, "guess save did not persist")
    except Exception as e:
        log(f"SAVE FAILED: {e}"); health_signal(False, f"save: {e}"); print("save failed:", e)

if __name__ == "__main__":
    main(dry=("--dry" in sys.argv))
