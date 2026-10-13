import math
from datetime import datetime, timezone

import numpy as np
import pytest

import astro

if not (astro.DATA / "stars.npz").exists():
    pytest.skip("нет data/stars.npz, запусти fetch_data.py", allow_module_level=True)

LAT, LON = math.radians(55.7558), math.radians(37.6173)

HORIZONS = [
    ("sun", (2026, 1, 15, 20, 0), 321.923439, -50.658112, 60),
    ("sun", (2026, 10, 13, 12, 0), 223.125002, 17.872341, 60),
    ("sun", (1990, 6, 1, 3, 0), 71.854767, 14.352655, 60),
    ("moon", (2026, 1, 15, 20, 0), 22.079862, -61.588489, 60),
    ("moon", (2026, 10, 13, 12, 0), 189.406934, 9.354560, 60),
    ("moon", (1990, 6, 1, 3, 0), 339.886775, -32.206522, 60),
    ("moon", (2025, 9, 7, 18, 11), 128.349805, 15.251291, 60),
    ("Марс", (2026, 10, 13, 12, 0), 295.500401, 6.168064, 300),
    ("Марс", (1990, 6, 1, 3, 0), 146.313622, 27.959194, 300),
    ("Юпитер", (2026, 1, 15, 20, 0), 150.738927, 53.960531, 300),
    ("Юпитер", (1990, 6, 1, 3, 0), 43.173719, -1.369029, 300),
    ("Сатурн", (2026, 10, 13, 12, 0), 53.353729, -20.206829, 600),
    ("Сатурн", (1990, 6, 1, 3, 0), 212.959981, 7.572623, 600),
]


def snapshot(*when, refraction=False):
    jd = astro.jd_from_datetime(datetime(*when, tzinfo=timezone.utc))
    return astro.compute_snapshot(jd, LAT, LON, [], refraction)


def altaz_vec(az, alt):
    a, h = math.radians(az), math.radians(alt)
    return np.array([math.cos(h) * math.sin(a), math.cos(h) * math.cos(a), math.sin(h)])


@pytest.mark.parametrize("key, when, az, alt, tol", HORIZONS)
def test_matches_horizons(key, when, az, alt, tol):
    body = next(b for b in snapshot(*when).bodies if b.key == key)
    sep = math.degrees(math.acos(min(1.0, float(altaz_vec(az, alt) @ body.vec)))) * 3600
    assert sep < tol


def test_refraction_lifts_low_bodies_only_when_enabled():
    plain = snapshot(2026, 10, 13, 14, 0).sun.vec
    refracted = snapshot(2026, 10, 13, 14, 0, refraction=True).sun.vec
    lift = math.degrees(math.asin(refracted[2]) - math.asin(plain[2]))
    assert 0.15 < lift < 0.25


def test_polaris_altitude_is_latitude():
    cat = astro.StarCatalog()
    i = cat.proper.tolist().index("Polaris")
    snap = snapshot(2026, 10, 13, 20, 0)
    alt = math.degrees(math.asin(cat.vectors(snap.jd)[i] @ snap.hor[2]))
    assert abs(alt - 55.7558) < 0.8


def test_total_solar_eclipse_1887():
    assert snapshot(1887, 8, 19, 4, 16, refraction=True).total
    partial = snapshot(1887, 8, 19, 3, 45, refraction=True)
    assert 0 < partial.eclipse < 1 and not partial.total
    assert snapshot(1887, 8, 19, 5, 30, refraction=True).eclipse == 0


def test_total_lunar_eclipse_2025():
    greatest = snapshot(2025, 9, 7, 18, 11).lunar
    assert greatest.umbral == pytest.approx(1.362, abs=0.01)
    assert greatest.penumbral == pytest.approx(2.346, abs=0.01)
    assert greatest.light < 0.001
    assert snapshot(2025, 9, 7, 17, 28).lunar.umbral < 1 <= snapshot(2025, 9, 7, 17, 33).lunar.umbral
    assert snapshot(2025, 9, 7, 18, 51).lunar.umbral >= 1 > snapshot(2025, 9, 7, 18, 55).lunar.umbral
    assert snapshot(2025, 9, 7, 15, 0).lunar.penumbral < 0


def test_no_lunar_eclipse_at_ordinary_full_moon():
    lunar = snapshot(2026, 10, 26, 4, 12).lunar
    assert lunar.penumbral < 0 and lunar.light == 1.0


def test_constellations():
    cons = astro.Constellations(astro.StarCatalog())
    if not len(cons):
        pytest.skip("нет data/constellations.json")
    assert len(cons.latin) == 88
    assert set(cons.codes) == set(cons.latin)
    assert np.allclose(np.linalg.norm(cons.label_vec, axis=1), 1.0)
