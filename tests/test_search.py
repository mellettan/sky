from datetime import datetime, timezone
from pathlib import Path

import pytest

import astro
import satellites

if not (astro.DATA / "stars.npz").exists():
    pytest.skip("нет data/stars.npz, запусти fetch_data.py", allow_module_level=True)

import sky


class FakeTerminal:
    default_bg = None

    def size(self):
        return 200, 57

    def write(self, s):
        pass


@pytest.fixture(scope="module")
def app():
    a = sky.App(FakeTerminal())
    a.sats = satellites.Satellites(Path(__file__).parent / "iss.tle")
    a.set_jd(astro.jd_from_datetime(datetime(2026, 10, 19, 3, 1, tzinfo=timezone.utc)))
    a.snapshot()
    return a


@pytest.mark.parametrize("query, proper", [
    ("вега", "Vega"),
    ("Полярная", "Polaris"),
    ("кас", "Castor"),
    ("barnard", "Barnard's Star"),
    ("Lalande 21185", "Lalande 21185"),
])
def test_finds_star(app, query, proper):
    key = app.search(query)
    assert key and key.startswith("star:")
    assert app.cat.proper[int(key[5:])] == proper


@pytest.mark.parametrize("query, key", [
    ("сатурн", "Сатурн"),
    ("луна", "moon"),
    ("jupiter", "Юпитер"),
    ("мкс", "sat:25544"),
    ("iss", "sat:25544"),
    ("хаббл", "sat:20580"),
])
def test_finds_body(app, query, key):
    assert app.search(query) == key


@pytest.mark.parametrize("query, code", [
    ("орион", "Ori"),
    ("Лира", "Lyr"),
    ("ursa major", "UMa"),
    ("большая", "UMa"),
    ("ori", "Ori"),
])
def test_finds_constellation(app, query, code):
    if not len(app.cons):
        pytest.skip("нет data/constellations.json")
    assert app.search(query) == f"con:{code}"


def test_iss_tooltip(app):
    lines = app.describe("sat:25544")
    assert lines[0] == "МКС · ISS (ZARYA)"
    assert lines[2].startswith("Виден до ") and lines[2].endswith("уходит на ВЮВ")


def test_unknown(app):
    assert app.search("абракадабра") is None
