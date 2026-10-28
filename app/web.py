"""Browser dashboard: live MJPEG picture, counts and the CO2 estimate."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

HTML = os.path.join(os.path.dirname(__file__), "dashboard.html")


class _Shared:
    jpeg = None
    cond = threading.Condition()
    stats = lambda: {}


def publish(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if ok:
        with _Shared.cond:
            _Shared.jpeg = buf.tobytes()
            _Shared.cond.notify_all()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.path = self.path.split("?")[0]
        if self.path == "/":
            with open(HTML, "rb") as f:
                self._send(f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/stats":
            self._send(json.dumps(_Shared.stats(), ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
        elif self.path == "/video":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _Shared.cond:
                        _Shared.cond.wait(timeout=2)
                        jpeg = _Shared.jpeg
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: %d\r\n\r\n" % len(jpeg))
                    self.wfile.write(jpeg + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def start(port, stats_fn):
    _Shared.stats = stats_fn
    srv = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[PANO] http://0.0.0.0:{port}/")
    return srv
