import http.client
import threading

import pytest

from tsplat import serve

BODY = bytes(range(256)) * 4  # 1024 bytes


@pytest.mark.parametrize(
    ("header", "span"),
    [
        ("bytes=0-9", (0, 9)),
        ("bytes=10-", (10, 1023)),
        ("bytes=-100", (924, 1023)),
        ("bytes=-5000", (0, 1023)),  # suffix longer than the file: whole file
        ("bytes=1000-5000", (1000, 1023)),  # end clamped
        ("bytes=1023-1023", (1023, 1023)),
        ("bytes=1024-", None),  # starts past the end
        ("bytes=-0", None),
        ("bytes=9-3", None),  # end before start
        ("bytes=-", None),
        ("bytes=a-b", None),
        ("bytes=0-1,5-9", None),  # multiple ranges: not supported
        ("items=0-9", None),
    ],
)
def test_parse_range(header, span):
    assert serve.parse_range(header, len(BODY)) == span


@pytest.fixture
def server(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(BODY)
    srv = serve.make_server(tmp_path, 0)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def get(port, headers=None, method="GET", path="/clip.mp4"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request(method, path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r, body


def test_full_file_without_range(server):
    r, body = get(server)
    assert r.status == 200 and body == BODY


@pytest.mark.parametrize(
    ("rng", "start", "end"), [("bytes=0-9", 0, 9), ("bytes=1000-", 1000, 1023), ("bytes=-24", 1000, 1023)]
)
def test_partial_content(server, rng, start, end):
    r, body = get(server, {"Range": rng})
    assert r.status == 206
    assert r.getheader("Content-Range") == f"bytes {start}-{end}/1024"
    assert int(r.getheader("Content-Length")) == end - start + 1
    assert body == BODY[start : end + 1]


@pytest.mark.parametrize("rng", ["bytes=2000-", "bytes=5-2", "bytes=x-y", "bytes=-0"])
def test_invalid_range_is_416(server, rng):
    r, body = get(server, {"Range": rng})
    assert r.status == 416 and body == b""
    assert r.getheader("Content-Range") == "bytes */1024"


def test_head_range_has_no_body(server):
    r, body = get(server, {"Range": "bytes=0-9"}, method="HEAD")
    assert r.status == 206 and body == b"" and r.getheader("Content-Length") == "10"


def test_missing_file_is_404(server):
    r, _ = get(server, {"Range": "bytes=0-9"}, path="/nope.mp4")
    assert r.status == 404


def test_concurrent_requests(server):
    """ThreadingHTTPServer: a held-open connection doesn't block another request."""
    held = http.client.HTTPConnection("127.0.0.1", server, timeout=5)
    held.request("GET", "/clip.mp4", headers={"Range": "bytes=0-"})
    resp = held.getresponse()  # leave the body unread
    r, body = get(server, {"Range": "bytes=0-1"})
    assert r.status == 206 and body == BODY[:2]
    resp.read()
    held.close()
