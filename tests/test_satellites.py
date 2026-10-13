import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import astro
import satellites

if satellites.Satrec is None:
    pytest.skip("нет пакета sgp4", allow_module_level=True)

LAT, LON = math.radians(55.7558), math.radians(37.6173)
TLE = Path(__file__).parent / "iss.tle"

HORIZONS = [
    ((2026, 10, 13, 6, 0), 248.957896, -9.102204, 3589.3),
    ((2026, 10, 13, 16, 30), 89.183570, -57.864798, 11286.0),
    ((2026, 10, 14, 3, 0), 282.253545, -80.210666, 12985.0),
]


@pytest.fixture(scope="module")
def sats():
    return satellites.Satellites(TLE)


def jd(*when):
    return astro.jd_from_datetime(datetime(*when, tzinfo=timezone.utc))


def iss(sats, *when):
    snap = astro.compute_snapshot(jd(*when), LAT, LON, [], False, sats)
    return next(b for b in snap.bodies if b.key == "sat:25544")


def test_names(sats):
    assert sats.names == ["МКС", "Хаббл"]


@pytest.mark.parametrize("when, az, alt, rng", HORIZONS)
def test_iss_matches_horizons(sats, when, az, alt, rng):
    b = iss(sats, *when)
    a, h = math.radians(az), math.radians(alt)
    ref = np.array([math.cos(h) * math.sin(a), math.cos(h) * math.cos(a), math.sin(h)])
    assert math.degrees(math.acos(min(1.0, float(ref @ b.vec)))) < 0.05
    assert b.extra["range"] == pytest.approx(rng, abs=5)


def test_iss_lit_only_out_of_earth_shadow(sats):
    dark = iss(sats, 2026, 10, 19, 2, 55)
    lit = iss(sats, 2026, 10, 19, 3, 1)
    assert dark.extra["light"] < 0.01 and dark.mag == 99.0
    assert lit.extra["light"] == 1.0 and -3.0 < lit.mag < -1.0


def test_iss_next_pass(sats):
    p = sats.next_pass(0, jd(2026, 10, 13, 12, 0), LAT, LON, days=5)
    assert jd(2026, 10, 17, 2, 55) < p["start"] < jd(2026, 10, 17, 3, 0)
    assert 3 / 1440 < p["end"] - p["start"] < 6 / 1440
    assert p["max_alt"] == pytest.approx(17, abs=2)


def test_hubble_never_visible_from_moscow(sats):
    assert sats.next_pass(1, jd(2026, 10, 13, 12, 0), LAT, LON, days=5) is None


def test_stale_elements_are_hidden(sats):
    snap = astro.compute_snapshot(jd(2025, 1, 1, 0, 0), LAT, LON, [], False, sats)
    assert not any(b.kind == "sat" for b in snap.bodies)
