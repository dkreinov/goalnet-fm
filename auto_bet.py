#!/usr/bin/env python3
"""Auto-bet the Friends fantasy app from the GoalNet model — with a fallback so no game is missed.

Called by the WorldCupLineups 10-min task. For each upcoming fixture in the 60–80 min pre-kickoff
window (start at 80, hard lock at 60):
  * CONFIRMED lineup is up  -> run GoalNet on the real XI -> set/UPGRADE the pick.
  * not up yet              -> run GoalNet on each team's PREVIOUS tournament-game XI (fallback)
                               -> set the pick now, so something is locked in before the deadline.
When the real lineup later appears (still before lock), the fallback pick is upgraded to it.
State per fixture: 'fallback' or 'confirmed' (a confirmed pick is final and not re-touched).

Run:  python auto_bet.py           # live
      python auto_bet.py --dry     # predict + log only, no writes
"""
import os, sys, json, re, subprocess, urllib.request, datetime, unicodedata, time

FM = os.path.dirname(os.path.abspath(__file__))
WC = r"D:\Programming\claude\worldcup\team_db"
AUTH = os.path.join(FM, "wc_bet_auth.json")
STATE = os.path.join(FM, "wc_bet_state.json")   # {fixture_id: 'fallback'|'confirmed'}
LOG = os.path.join(FM, "wc_bet.log")
TMPLU = os.path.join(FM, "_tmp_lineups.json")
NOTIFY = os.path.join(FM, "wc_notify.json")     # {"hc_url":..,"ntfy_url":..,"win_toast":true}  (gitignored)
HEALTH = os.path.join(FM, "wc_bet_health.json") # {"status":"ok|fail","since":ts,"last_notify":ts}
RENOTIFY_SEC = 2 * 3600                          # re-alert at most every 2h while still down
PY = sys.executable
ANON = ("***REMOVED***"
        "***REMOVED***"
        "***REMOVED***")
BASE = "***REMOVED***"
WINDOW = 80   # start considering a game this many minutes before kickoff
LOCK = 60     # never write within this many minutes of kickoff (app lock)

def log(m): open(LOG, "a", encoding="utf-8").write(f"{datetime.datetime.now():%Y-%m-%d %H:%M} | {m}\n")

def _toast(title, msg):
    """Immediate local popup (Windows Education/Pro has msg.exe). Best-effort, never raises."""
    try: subprocess.run(["msg", "*", "/TIME:60", f"{title}: {msg}"], timeout=15)
    except Exception: pass

def _ping(url, body=None, headers=None):
    try:
        req = urllib.request.Request(url, data=(body.encode("utf-8") if body else b""),
                                     headers=(headers or {}), method="POST")
        urllib.request.urlopen(req, timeout=15)
    except Exception: pass

def health_signal(ok, reason=""):
    """Track ok/fail transitions and PUSH on: first failure, recovery, and every 2h while still down.
    Always pings Healthchecks (so the bot dying entirely — PC asleep, task disabled — also trips its
    dead-man's-switch alert, exactly the silent 17h outage we just had). Channels from wc_notify.json."""
    try: cfg = json.load(open(NOTIFY))
    except Exception: cfg = {}
    try: st = json.load(open(HEALTH))
    except Exception: st = {}
    now = int(time.time())
    hc = cfg.get("hc_url")
    if hc:                                                   # ping every run -> keeps the check alive
        _ping(hc if ok else hc.rstrip("/") + "/fail", None if ok else reason[:400])
    prev = st.get("status")
    if ok:
        fire = (prev == "fail")                             # recovered
    else:
        fire = (prev != "fail") or (now - st.get("last_notify", 0) >= RENOTIFY_SEC)
    if fire:
        if ok:
            title, body = "WC auto-bet RECOVERED", "session works again"
        else:
            title, body = "WC auto-bet AUTH FAILED", f"{reason[:120]} — re-seed token via browser login"
        if cfg.get("win_toast", True): _toast(title, body)
        if cfg.get("ntfy_url"):
            _ping(cfg["ntfy_url"], body,
                  {"Title": title, "Priority": ("default" if ok else "high"), "Tags": ("white_check_mark" if ok else "rotating_light")})
        st["last_notify"] = now
    if prev != ("ok" if ok else "fail"): st["since"] = now
    st["status"] = "ok" if ok else "fail"
    try: json.dump(st, open(HEALTH, "w"))
    except Exception: pass

def api(url, method="GET", data=None, bearer=None):
    h = {"apikey": ANON, "Content-Type": "application/json"}
    if bearer: h["Authorization"] = "Bearer " + bearer
    if method == "POST" and "/rest/" in url: h["Prefer"] = "resolution=merge-duplicates,return=representation"
    r = urllib.request.Request(url, data=(json.dumps(data).encode() if data is not None else None), headers=h, method=method)
    return json.loads(urllib.request.urlopen(r, timeout=25).read().decode() or "null")

def get_access():
    a = json.load(open(AUTH))
    tok = api(BASE + "/auth/v1/token?grant_type=refresh_token", "POST", {"refresh_token": a["refresh_token"]})
    a["refresh_token"] = tok["refresh_token"]; a["access_token"] = tok["access_token"]; a["expires_at"] = tok.get("expires_at")
    json.dump(a, open(AUTH, "w"))
    return tok["access_token"], a["user_id"]

def _surn(name):
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().strip()
    return s.split()[-1] if s else ""

def load_rosters():
    import glob
    R = {}
    for f in glob.glob(os.path.join(WC, "teams", "*.json")):
        code = os.path.basename(f)[:-5]
        try: d = json.load(open(f, encoding="utf-8"))
        except Exception: continue
        R[code] = set(_surn(p["name"]) for p in d.get("players", []))
    return R

def team_prev_xi(code, lu, results, rosters):
    """That team's XI from its most recent FINISHED tournament game (roster-attributed), or None."""
    cands = []
    for k, e in lu.items():
        if e.get("state") != "finished": continue
        if code not in k.split("-"): continue
        hx, ax = e.get("home_xi") or [], e.get("away_xi") or []
        if len(hx) < 11 or len(ax) < 11: continue
        cands.append(((results.get(k) or {}).get("kickoff", 0), hx, ax))
    if not cands: return None
    cands.sort(reverse=True)
    _, hx, ax = cands[0]
    rs = rosters.get(code, set())
    ov = lambda xi: sum(1 for p in xi if _surn(p.get("full", "")) in rs)
    return hx if ov(hx) >= ov(ax) else ax

def predict(key, lineups_path=None):
    cmd = [PY, os.path.join(FM, "src", "predict_game.py"), key]
    if lineups_path: cmd += ["--lineups", lineups_path]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=FM)
    except Exception:
        return None
    m = re.search(r"EV pick:\s+([A-Z]{3})\s+(\d+)-(\d+)\s+([A-Z]{3})", (p.stdout or "") + (p.stderr or ""))
    return (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)) if m else None

def main(dry=False):
    state = {}
    if os.path.exists(STATE):
        try: state = json.load(open(STATE))
        except Exception: pass
    try: bearer, uid = get_access()
    except Exception as e:
        log(f"AUTH FAILED: {e}"); health_signal(False, f"auth: {e}"); print("auth failed:", e); return
    health_signal(True)                                     # auth works -> clear/recover the alert
    try:
        fx = api(BASE + "/rest/v1/fixtures?select=id,round,kickoff_utc,home_team,away_team,status&order=kickoff_utc.asc", bearer=bearer)
        lu = json.load(open(os.path.join(WC, "lineups.json"), encoding="utf-8"))
        results = json.load(open(os.path.join(WC, "results.json"), encoding="utf-8"))
    except Exception as e:
        log(f"READ FAILED: {e}"); health_signal(False, f"read: {e}"); print("read failed:", e); return
    rosters = load_rosters()
    now = datetime.datetime.now(datetime.timezone.utc)
    acted = 0
    for f in fx:
        if f.get("status") != "scheduled": continue
        fid = str(f["id"])
        if state.get(fid) == "confirmed": continue                 # already finalised
        try: kodt = datetime.datetime.fromisoformat((f.get("kickoff_utc") or "").replace("Z", "+00:00"))
        except Exception: continue
        mins = (kodt - now).total_seconds() / 60
        if not (LOCK < mins <= WINDOW): continue                   # outside the 60–80 write window
        H, A = f["home_team"], f["away_team"]
        # 1) confirmed real lineup for this pair?
        realkey = None
        for cand in (f"{H}-{A}", f"{A}-{H}"):
            e = lu.get(cand)
            if e and len(e.get("home_xi") or []) >= 11 and len(e.get("away_xi") or []) >= 11 and e.get("state") == "notstarted":
                realkey = cand; break
        if realkey:
            tag, pred = "confirmed", predict(realkey)
        else:                                                       # 2) fallback: previous-game XIs
            hx, ax = team_prev_xi(H, lu, results, rosters), team_prev_xi(A, lu, results, rosters)
            if not (hx and ax):
                continue                                            # no prior game (e.g. matchday 1) -> wait
            json.dump({f"{H}-{A}": {"home_xi": hx, "away_xi": ax, "state": "notstarted"}}, open(TMPLU, "w"))
            tag, pred = "fallback", predict(f"{H}-{A}", TMPLU)
        if not pred:
            log(f"{H}-{A} fid={fid}: no EV pick ({tag})"); continue
        mh, hs, ascore, ma = pred
        if mh == H and ma == A: home_score, away_score = hs, ascore
        elif mh == A and ma == H: home_score, away_score = ascore, hs
        else: log(f"{H}-{A} fid={fid}: model {mh}-{ma} != fixture pair"); continue
        # decide whether to write: set fallback once; always upgrade to confirmed
        cur = state.get(fid)
        if tag == "confirmed" and cur != "confirmed":
            do = "upgrade->confirmed" if cur == "fallback" else "set(confirmed)"
        elif tag == "fallback" and cur is None:
            do = "set(fallback)"
        else:
            continue
        msg = f"{H} {home_score}-{away_score} {A}  [{do}, KO in {mins:.0f}m]"
        print(msg + ("  [DRY]" if dry else "  WRITING"))
        if dry: log("DRY " + msg); continue
        try:
            api(BASE + "/rest/v1/picks", "POST",
                [{"user_id": uid, "fixture_id": f["id"], "home_score": home_score, "away_score": away_score}], bearer=bearer)
            chk = api(BASE + f"/rest/v1/picks?select=home_score,away_score&fixture_id=eq.{f['id']}&user_id=eq.{uid}", bearer=bearer)
            if chk and chk[0]["home_score"] == home_score and chk[0]["away_score"] == away_score:
                state[fid] = tag; acted += 1; log("WROTE+verified " + msg)
            else:
                log(f"WRITE UNVERIFIED fid={fid}: got {chk}")
        except Exception as e:
            log(f"WRITE FAILED fid={fid}: {e}")
    json.dump(state, open(STATE, "w"))
    log(f"run complete: acted={acted}")
    print(f"run complete, acted={acted}")

if __name__ == "__main__":
    main(dry=("--dry" in sys.argv))
