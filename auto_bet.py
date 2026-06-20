#!/usr/bin/env python3
"""Auto-bet the Friends fantasy app from the GoalNet model.

Every run (called by the WorldCupLineups 10-min task): for each upcoming Friends-app fixture whose
STARTING LINEUP is confirmed AND kickoff is >65 min away (buffer over the 60-min lock), run the model
(predict_game.py), take its EV pick, and upsert that scoreline to the app via Supabase REST. Each
fixture is set once (dedup). Fully headless — refreshes the stored Supabase session each run.

Run:  python auto_bet.py            # live
      python auto_bet.py --dry      # predict + log only, no writes
"""
import os, sys, json, re, subprocess, urllib.request, datetime

FM = os.path.dirname(os.path.abspath(__file__))
WC = r"D:\Programming\claude\worldcup\team_db"
AUTH = os.path.join(FM, "wc_bet_auth.json")
DONE = os.path.join(FM, "wc_bet_done.json")
LOG = os.path.join(FM, "wc_bet.log")
PY = sys.executable
ANON = ("***REMOVED***"
        "***REMOVED***"
        "***REMOVED***")
BASE = "***REMOVED***"
MIN_BEFORE = 65   # only act when kickoff is more than this many minutes away

def log(m): open(LOG, "a", encoding="utf-8").write(f"{datetime.datetime.now():%Y-%m-%d %H:%M} | {m}\n")

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

def predict(key):
    """Run predict_game.py for a lineups key -> (modelHome, hs, as, modelAway) or None."""
    try:
        p = subprocess.run([PY, os.path.join(FM, "src", "predict_game.py"), key],
                           capture_output=True, text=True, timeout=120, cwd=FM)
    except Exception:
        return None
    m = re.search(r"EV pick:\s+([A-Z]{3})\s+(\d+)-(\d+)\s+([A-Z]{3})", (p.stdout or "") + (p.stderr or ""))
    return (m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)) if m else None

def main(dry=False):
    done = set()
    if os.path.exists(DONE):
        try: done = set(json.load(open(DONE)))
        except Exception: pass
    try: bearer, uid = get_access()
    except Exception as e:
        log(f"AUTH FAILED: {e}"); print("auth failed:", e); return
    try:
        fx = api(BASE + "/rest/v1/fixtures?select=id,round,kickoff_utc,home_team,away_team,status&order=kickoff_utc.asc", bearer=bearer)
        lu = json.load(open(os.path.join(WC, "lineups.json"), encoding="utf-8"))
    except Exception as e:
        log(f"READ FAILED: {e}"); print("read failed:", e); return
    now = datetime.datetime.now(datetime.timezone.utc)
    acted = 0
    for f in fx:
        if f.get("status") != "scheduled": continue
        fid = str(f["id"])
        if fid in done: continue
        try: kodt = datetime.datetime.fromisoformat((f.get("kickoff_utc") or "").replace("Z", "+00:00"))
        except Exception: continue
        mins = (kodt - now).total_seconds() / 60
        if mins <= MIN_BEFORE: continue                       # too late to beat the lock
        H, A = f["home_team"], f["away_team"]
        key = None                                            # lineups key with a confirmed XI for this pair
        for cand in (f"{H}-{A}", f"{A}-{H}"):
            e = lu.get(cand)
            if e and len(e.get("home_xi") or []) >= 11 and len(e.get("away_xi") or []) >= 11 and e.get("state") == "notstarted":
                key = cand; break
        if not key: continue                                  # lineup not confirmed yet
        pred = predict(key)
        if not pred: log(f"{H}-{A} fid={fid}: model gave no EV pick (key {key})"); continue
        mh, hs, ascore, ma = pred
        if mh == H and ma == A: home_score, away_score = hs, ascore
        elif mh == A and ma == H: home_score, away_score = ascore, hs
        else: log(f"{H}-{A} fid={fid}: model {mh}-{ma} doesn't match fixture pair"); continue
        msg = f"{H} {home_score}-{away_score} {A}  (KO in {mins:.0f}m, key {key}, model {mh} {hs}-{ascore} {ma})"
        print(msg + ("  [DRY]" if dry else "  WRITING"))
        if dry: log("DRY " + msg); continue
        try:
            api(BASE + "/rest/v1/picks", "POST",
                [{"user_id": uid, "fixture_id": f["id"], "home_score": home_score, "away_score": away_score}], bearer=bearer)
            chk = api(BASE + f"/rest/v1/picks?select=home_score,away_score&fixture_id=eq.{f['id']}&user_id=eq.{uid}", bearer=bearer)
            if chk and chk[0]["home_score"] == home_score and chk[0]["away_score"] == away_score:
                done.add(fid); acted += 1; log("WROTE+verified " + msg)
            else:
                log(f"WRITE UNVERIFIED fid={fid}: got {chk}")
        except Exception as e:
            log(f"WRITE FAILED fid={fid}: {e}")
    json.dump(sorted(done), open(DONE, "w"))
    log(f"run complete: acted={acted}, done_total={len(done)}")
    print(f"run complete, acted={acted}")

if __name__ == "__main__":
    main(dry=("--dry" in sys.argv))
