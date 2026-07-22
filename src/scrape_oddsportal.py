"""Scrape OddsPortal closing 1X2 odds for the WC2026 slate (Phase-6 market layer).

OddsPortal serves its archive tables through an encrypted AJAX endpoint (the static results page is
a Vue shell with no odds in the HTML). This module replicates that endpoint:
  GET /ajax-sport-country-tournament-archive_/1/<hash>/<filter>/1/0[/page/N]//?_=<ms>
      headers: X-Requested-With: XMLHttpRequest, Referer: <results page>
  body: base64( base64(AES-CBC ciphertext) : hex(iv) ), key = PBKDF2-SHA256(pass, salt, 1000, 32),
        plaintext is UTF-8 JSON, gzip-wrapped when it starts with the gzip magic.
The pass/salt below are the site's public client-side constants (extracted from its JS bundle); they
are not secrets and gate no private data — this is the same table any visitor's browser renders.

Verified 2026-07-22: the "World Championship 2026" tournament (hash zeSHfCx3) is the SAME fixture set
as our slate — 33/33 exact scoreline agreement with worldcup/team_db/results.json, 104 matches in the
2026 window. Odds array order per match is [home, draw, away] (bettingTypeId=1, scopeId=2 full-time).

Throttled + disk-cached via src/fetch.py (re-runs free/resumable). Output:
  data/oddsportal_raw.csv  columns: tournament, season, timestamp, home, away, hs, as,
                                    odd_h, odd_d, odd_a   (market-average decimal odds)

Usage:
  python src/scrape_oddsportal.py            # scrape configured tournaments -> CSV
  python src/scrape_oddsportal.py --limit 2  # smoke: first 2 pages of each
"""
import base64
import csv
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fetch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "oddsportal_raw.csv"

# Public client-side constants from the OddsPortal JS bundle (app-*.js). Not secrets.
_PASS = b"J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k"
_SALT = b"5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d"

# Tournaments to scrape: (label, tournament_hash, filter_param, results_page_for_referer).
# The filter param is the long X-delimited state string the site sends for the results view; it is
# stable per tournament. Add national comps here (World Cup 2022, Euro 2024, ...) to widen coverage;
# discover a new hash/filter by opening its /results/ page and reading the archive_ XHR (see module
# docstring). For Phase 6 the WC2026 slate is the essential target.
TOURNAMENTS = [
    ("world-championship-2026", "zeSHfCx3",
     "X2X0X0X0X0X0X16384X0X0X0X0X0X0X0X0X0X0X0X0X0X0X268468224X0X0X0X512X0X167772160X0X0X8",
     "https://www.oddsportal.com/football/world/world-championship-2026/results/"),
]

ARCHIVE = "https://www.oddsportal.com/ajax-sport-country-tournament-archive_/1/{hash}/{flt}/1/0{page}//?_={ms}"


def _decrypt(payload: str) -> str:
    """OddsPortal encryptedResponse -> plaintext JSON string."""
    outer = base64.b64decode(payload).decode()
    ct_b64, iv_hex = outer.split(":")
    key = hashlib.pbkdf2_hmac("sha256", _PASS, _SALT, 1000, dklen=32)
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    dec = Cipher(algorithms.AES(key), modes.CBC(bytes.fromhex(iv_hex))).decryptor()
    pt = dec.update(base64.b64decode(ct_b64)) + dec.finalize()
    pt = pt[: -pt[-1]]                                   # strip PKCS7 padding
    if pt[:2] == b"\x1f\x8b":
        pt = gzip.decompress(pt)
    return pt.decode("utf-8", errors="replace")


def _odds_hda(match):
    """Extract (odd_h, odd_d, odd_a) market-average decimals from a match's 1X2 full-time odds.
    Array order is [home, draw, away]; prefer avgOdds, fall back to maxOdds. None if malformed."""
    ft = [o for o in match.get("odds", [])
          if o.get("bettingTypeId") == 1 and o.get("scopeId") == 2]
    if len(ft) < 3:
        return None
    out = []
    for o in ft[:3]:
        v = o.get("avgOdds") or o.get("maxOdds")
        if not v or v <= 1:
            return None
        out.append(float(v))
    return out


def scrape_tournament(label, thash, flt, referer, limit=None):
    """Yield one row dict per finished match with valid 1X2 odds (all pages)."""
    headers = {"X-Requested-With": "XMLHttpRequest", "Referer": referer}
    seen = set()
    page, pagecount = 1, 1
    while page <= pagecount:
        page_seg = "" if page == 1 else f"/page/{page}"
        url = ARCHIVE.format(hash=thash, flt=flt, page=page_seg, ms=int(time.time() * 1000))
        txt = fetch.get(url, min_delay=1.6, retries=3, timeout=45, cache=True, headers=headers)
        d = json.loads(_decrypt(txt))["d"]
        pagecount = d["pagination"]["pageCount"]
        for m in d["rows"]:
            key = (m["home-name"], m["away-name"], m["date-start-timestamp"])
            if key in seen:
                continue
            seen.add(key)
            hr, ar = m.get("homeResult"), m.get("awayResult")
            if hr in (None, "") or ar in (None, ""):     # not a finished match
                continue
            try:
                hs, as_ = int(hr), int(ar)               # numeric fields cover FT + ET/pen games
            except (ValueError, TypeError):
                continue
            odds = _odds_hda(m)
            if odds is None:
                continue
            season = m.get("tournament-name", label)
            yield {"tournament": label, "season": season,
                   "timestamp": m["date-start-timestamp"],
                   "home": m["home-name"], "away": m["away-name"],
                   "hs": hs, "as": as_,
                   "odd_h": odds[0], "odd_d": odds[1], "odd_a": odds[2]}
        print(f"  [{label}] page {page}/{pagecount}: {len(seen)} matches seen", flush=True)
        if limit and page >= limit:
            break
        page += 1


def main():
    args = sys.argv
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    rows = []
    for label, thash, flt, referer in TOURNAMENTS:
        rows.extend(scrape_tournament(label, thash, flt, referer, limit=limit))
    cols = ["tournament", "season", "timestamp", "home", "away", "hs", "as", "odd_h", "odd_d", "odd_a"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}: {len(rows)} rows", flush=True)
    # 2026-window count (sanity vs the 104-game slate)
    n2026 = sum(1 for r in rows if 1780000000 < r["timestamp"] < 1785000000)
    print(f"  2026-window matches: {n2026}", flush=True)


if __name__ == "__main__":
    main()
