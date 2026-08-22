"""
_load_prev_clv() fetches the published ledger to seed the next build. It grew a
gzip-compressed path in August 2026 after the plain clv.db crossed GitHub's 100 MiB
per-file push limit (every push was silently rejected for 14+ hours -- see build_static.py's
comment above OUT_CLV). These tests cover the new decompress-on-load path and its fallback
chain, with urllib mocked -- no real network call.
"""
import gzip
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_static as BS


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _sqlite_bytes() -> bytes:
    return b"SQLite format 3\x00" + b"\x00" * 32


def test_loads_and_decompresses_the_gzip_published_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "OUT_CLV", tmp_path / "clv.db")
    raw = _sqlite_bytes()
    gz_bytes = gzip.compress(raw)

    def fake_urlopen(req, timeout=None):
        assert req.full_url.startswith(BS._CLV_GZ_URL)
        return _FakeResponse(gz_bytes)

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    src = BS._load_prev_clv()
    assert src == "data-branch"
    assert BS.OUT_CLV.read_bytes() == raw


def test_falls_back_to_legacy_uncompressed_url_when_gz_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "OUT_CLV", tmp_path / "clv.db")
    raw = _sqlite_bytes()

    import urllib.request

    def fake_urlopen(req, timeout=None):
        if req.full_url.startswith(BS._CLV_GZ_URL):
            raise Exception("404")
        assert req.full_url.startswith(BS._CLV_URL)
        return _FakeResponse(raw)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    src = BS._load_prev_clv()
    assert src == "data-branch (legacy uncompressed)"
    assert BS.OUT_CLV.read_bytes() == raw


def test_falls_back_to_seed_when_both_urls_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "OUT_CLV", tmp_path / "clv.db")
    seed = tmp_path / "seed.db"
    seed.write_bytes(_sqlite_bytes())
    monkeypatch.setattr(BS, "SEED_CLV", seed)

    import urllib.request
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(Exception("network down")))

    src = BS._load_prev_clv()
    assert src == "seed"
    assert BS.OUT_CLV.read_bytes() == seed.read_bytes()


def test_a_404_html_page_never_overwrites_the_ledger_gz_or_plain(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "OUT_CLV", tmp_path / "clv.db")
    monkeypatch.setattr(BS, "SEED_CLV", tmp_path / "no-seed-here.db")

    import urllib.request

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(b"404: Not Found")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    src = BS._load_prev_clv()
    assert src == "empty"
    assert not BS.OUT_CLV.exists()
