"""Development-only authenticated relay for offline Linux acceptance.

The relay exposes only the configured chat-completions endpoint. It never
logs credentials, requests or response contents and is not part of delivery.
"""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import ipaddress
import os
from pathlib import Path
import socketserver
import sys
import threading
import urllib.error
import urllib.request
import urllib.parse

from dotenv import load_dotenv


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve(path=None, port=None, bind="127.0.0.1"):
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    key = os.environ["OPENAI_API_KEY"]
    endpoint = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme != "https" or parsed.hostname != "changellenge.up.railway.app":
        raise ValueError("The development relay permits only the organizer's HTTPS Qwen endpoint")
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def do_POST(self):
            if self.path != "/v1/chat/completions" or not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + key):
                self.send_error(403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            request = urllib.request.Request(endpoint, data=body, headers={"Authorization": "Bearer " + key,
                "Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    content, status = response.read(), response.status
            except urllib.error.HTTPError as error:
                content, status = b'{"error":"upstream HTTP failure"}', error.code
            except (urllib.error.URLError, TimeoutError):
                content, status = b'{"error":"upstream unavailable"}', 502
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass
    if path and path.exists():
        raise ValueError("Relay socket already exists; do not replace an active channel")
    address = ipaddress.ip_address(bind)
    if address.is_unspecified or not address.is_private:
        raise ValueError("Relay must bind a specific private or loopback address")
    with (Server(str(path), Handler) if path else ThreadingHTTPServer((bind, port), Handler)) as server:
        if path:
            path.chmod(0o600)
        print("Authenticated chat relay ready", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--socket", type=Path)
    group.add_argument("--port", type=int)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    serve(args.socket, args.port, args.bind)
