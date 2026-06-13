"""Shared HTTP fetcher: disk cache, retries, polite rate limiting."""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

_session = requests.Session()
_session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
_last_hit: dict[str, float] = {}


def get(url: str, *, min_delay: float = 1.5, retries: int = 3, cache: bool = True,
        timeout: int = 45, headers: dict | None = None) -> str:
    """GET with per-host rate limit and disk cache. Returns text. Raises on final failure."""
    key = hashlib.sha1(url.encode()).hexdigest()
    cpath = CACHE / key[:2] / f"{key}.html"
    if cache and cpath.exists():
        return cpath.read_text(encoding="utf-8", errors="replace")

    host = url.split("/")[2]
    wait = _last_hit.get(host, 0) + min_delay - time.time()
    if wait > 0:
        time.sleep(wait + random.uniform(0, 0.5))

    last_exc = None
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=timeout, headers=headers or {})
            _last_hit[host] = time.time()
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                last_exc = RuntimeError(f"HTTP {r.status_code} {url}")
                continue
            r.raise_for_status()
            if cache:
                cpath.parent.mkdir(parents=True, exist_ok=True)
                cpath.write_text(r.text, encoding="utf-8")
            return r.text
        except requests.RequestException as e:
            last_exc = e
            time.sleep(3 * (attempt + 1))
    raise last_exc
