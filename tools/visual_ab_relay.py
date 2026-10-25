"""Byte-preserving bridge to the host's accounted gateway; no provider key."""
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os

ALLOWED_PATHS = {'/' + arm + '/v1/chat/completions' for arm in os.environ.get('ALLOWED_ARMS',
    'regression_base,regression_visual,classification_base,classification_visual').split(',')}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.forward(None) if self.path == '/health' else self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        if self.path not in ALLOWED_PATHS or not 0 < length <= 262144:
            self.send_error(403)
            return
        self.forward(self.rfile.read(length))

    def forward(self, body):
        connection = HTTPConnection('host.docker.internal', int(os.environ['GATEWAY_PORT']), timeout=10800)
        try:
            connection.request('GET' if body is None else 'POST', self.path, body=body,
                headers={key: self.headers[key] for key in ('Authorization', 'Content-Type') if key in self.headers})
            response = connection.getresponse()
            content, status = response.read(), response.status
            content_type = response.getheader('Content-Type', 'application/json')
        except (OSError, TimeoutError):
            status, content, content_type = 402, b'{"error":{"code":"local_budget_gateway_unavailable"}}', 'application/json'
        finally:
            connection.close()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    ThreadingHTTPServer(('0.0.0.0', 18767), Handler).serve_forever()
