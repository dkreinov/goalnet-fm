#!/usr/bin/env python3
"""One-shot local capture for the Hevre (Sport5) board session. Receives loginToken/refreshToken/
PHPSESSID from the logged-in board page (fetch POST or top-level GET) and writes hevre_auth.json so the
headless hevre_bot can call data.php unattended. Token never passes through the agent. 127.0.0.1 only."""
import json, os, base64
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hevre_auth.json")
done = {"ok": False}

def _save(d):
    if d.get("loginToken"):
        json.dump(d, open(OUT, "w", encoding="utf-8")); done["ok"] = True

class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query); d64 = (q.get("d") or [""])[0]
        try: _save(json.loads(base64.b64decode(d64).decode("utf-8")))
        except Exception: pass
        self.send_response(200); self.send_header("content-type", "text/html"); self.end_headers()
        self.wfile.write(b"<h2>" + (b"Hevre session saved. Close this tab." if done["ok"] else b"No token.") + b"</h2>")
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        try: _save(json.loads(self.rfile.read(n).decode("utf-8", "replace")))
        except Exception: pass
        self.send_response(200); self._cors(); self.end_headers()
        self.wfile.write(b'{"saved":%s}' % (b"true" if done["ok"] else b"false"))
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 8799), H)
print("hevre capture listening on 127.0.0.1:8799", flush=True)
while not done["ok"]:
    srv.handle_request()
print("hevre session saved ->", OUT, flush=True)
