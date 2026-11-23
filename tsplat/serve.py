"""Local static server for the replay viewer, with HTTP Range support (browsers need it to seek <video>)."""

from __future__ import annotations

import http.server
import re
from functools import partial
from pathlib import Path

_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def parse_range(header: str, size: int) -> tuple[int, int] | None:
    """One `bytes=` range -> inclusive (start, end), or None if invalid or unsatisfiable for `size`.
    Accepts `bytes=a-b`, `bytes=a-` and the suffix form `bytes=-n` (last n bytes)."""
    m = _RANGE.fullmatch(header.strip())
    if not m or size <= 0:
        return None
    a, b = m.groups()
    if not a and not b:
        return None
    if not a:  # suffix: last n bytes
        n = int(b)
        return (max(0, size - n), size - 1) if n > 0 else None
    start = int(a)
    end = min(int(b), size - 1) if b else size - 1
    if start >= size or (b and int(b) < start):
        return None
    return start, end


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + single byte ranges (206), 416 for malformed or unsatisfiable ranges."""

    _remaining: int | None = None

    def send_head(self):
        self._remaining = None
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not Path(path).is_file():
            return super().send_head()
        size = Path(path).stat().st_size
        span = parse_range(rng, size)
        if span is None:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        start, end = span
        f = open(path, "rb")  # closed by the base handler after copyfile
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        n = self._remaining
        if n is None:
            return super().copyfile(source, outputfile)
        while n > 0:
            chunk = source.read(min(65536, n))
            if not chunk:
                break
            outputfile.write(chunk)
            n -= len(chunk)
        self._remaining = None

    def log_message(self, *a):
        pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True  # a stalled <video> connection must not block shutdown
    allow_reuse_address = True


def make_server(directory: str | Path, port: int, host: str = "127.0.0.1") -> Server:
    return Server((host, port), partial(RangeHandler, directory=str(directory)))
