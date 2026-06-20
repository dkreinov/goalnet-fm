#!/usr/bin/env python3
"""One-shot local capture: receives the Friends-app Supabase refresh token from the browser
(POST from the logged-in page) and writes it to wc_bet_auth.json, so the headless auto-bet
script can refresh the session unattended. Token never passes through the agent's context.
Binds 127.0.0.1 only; handles CORS + Private-Network-Access preflight; exits after first save."""
import json, os, base64
from http.server import BaseHTTPRequestHandler, HTTPServer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wc_bet_auth.json")
done = {"ok": False}

class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()
    def do_GET(self):
        # top-level navigation path: /save?d=<base64(json)> (not blocked by mixed-content/PNA)
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(self.path).query)
        d64 = (q.get("d") or [""])[0]
        try:
            d = json.loads(base64.b64decode(d64).decode("utf-8"))
            if d.get("refresh_token") and d.get("user_id"):
                json.dump(d, open(OUT, "w", encoding="utf-8")); done["ok"] = True
        except Exception:
            pass
        self.send_response(200); self.send_header("content-type", "text/html"); self.end_headers()
        self.wfile.write(b"<h2>" + (b"Token saved. You can close this tab." if done["ok"] else b"No token.") + b"</h2>")
    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n).decode("utf-8", "replace")
        try:
            d = json.loads(body)
            if d.get("refresh_token") and d.get("user_id"):
                json.dump(d, open(OUT, "w", encoding="utf-8"))
                done["ok"] = True
        except Exception:
            pass
        self.send_response(200); self._cors(); self.end_headers()
        self.wfile.write(b'{"saved":%s}' % (b"true" if done["ok"] else b"false"))
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", 8799), H)
print("capture listening on 127.0.0.1:8799", flush=True)
while not done["ok"]:
    srv.handle_request()
print("token saved ->", OUT, flush=True)
