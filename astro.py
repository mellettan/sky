import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

DEG = math.pi / 180.0
J2000 = 2451545.0
OBLIQ_J2000 = 23.4392911 * DEG
K_GAUSS = 0.01720209895
AU_KM = 149597870.7
EARTH_R_KM = 6378.14
LIGHT_DAY_PER_AU = 0.0057755183
EPOCH_UNIX = datetime(1970, 1, 1, tzinfo=timezone.utc)
JD_MIN = 1721426.5 + 1
JD_MAX = 5373484.5 - 1

DATA = Path(__file__).resolve().parent / "data"

PLANETS = {
    "Меркурий": ((0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
                 (0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081)),
    "Венера": ((0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
               (0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418)),
    "Земля": ((1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0),
              (0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0)),
    "Марс": ((1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
             (0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343)),
    "Юпитер": ((5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
               (-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106)),
    "Сатурн": ((9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
               (-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794)),
    "Уран": ((19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503),
             (-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589)),
    "Нептун": ((30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574),
               (0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664)),
}

PLANET_INFO = {
    "Меркурий": (2439.7, (190, 180, 170), "Mercury"),
    "Венера": (6051.8, (255, 250, 225), "Venus"),
    "Марс": (3396.2, (255, 140, 90), "Mars"),
    "Юпитер": (71492.0, (250, 230, 200), "Jupiter"),
    "Сатурн": (60268.0, (245, 220, 160), "Saturn"),
    "Уран": (25559.0, (180, 230, 235), "Uranus"),
    "Нептун": (24764.0, (130, 160, 255), "Neptune"),
}


def jd_from_datetime(dt):
    return (dt - EPOCH_UNIX).total_seconds() / 86400.0 + 2440587.5


def datetime_from_jd(jd):
    return EPOCH_UNIX + timedelta(days=jd - 2440587.5)


def clamp_jd(jd):
    return min(max(jd, JD_MIN), JD_MAX)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


ECL_TO_EQ = rot_x(-OBLIQ_J2000)

POLES = {
    k: np.array([math.cos(dec * DEG) * math.cos(ra * DEG), math.cos(dec * DEG) * math.sin(ra * DEG), math.sin(dec * DEG)])
    for k, (ra, dec) in {
        "moon": (269.9949, 66.5392),
        "Юпитер": (268.057, 64.495),
        "Сатурн": (40.589, 83.537),
        "Марс": (317.681, 52.887),
    }.items()
}

EQ_TO_GAL = np.array([
    [-0.0548755604, -0.8734370902, -0.4838350155],
    [0.4941094279, -0.4448296300, 0.7469822445],
    [-0.8676661490, -0.1980763734, 0.4559837762],
])


def precession_matrix(jd):
    t = (jd - J2000) / 36525.0
    zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t ** 3) / 3600.0 * DEG
    z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t ** 3) / 3600.0 * DEG
    theta = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t ** 3) / 3600.0 * DEG
    return rot_z(-z) @ rot_y(theta) @ rot_z(-zeta)


def gmst(jd):
    t = (jd - J2000) / 36525.0
    g = 280.46061837 + 360.98564736629 * (jd - J2000) + 0.000387933 * t * t - t ** 3 / 38710000.0
    return (g % 360.0) * DEG


def horizon_matrix(lat):
    s, c = math.sin(lat), math.cos(lat)
    return np.array([[0.0, 1.0, 0.0], [-s, 0.0, c], [c, 0.0, s]])


def radec_to_vec(ra, dec):
    cd = np.cos(dec)
    return np.stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)], axis=-1)


def vec_to_altaz(v):
    alt = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    az = np.arctan2(v[..., 0], v[..., 1]) % (2 * math.pi)
    return alt, az


def refract(v):
    alt = np.arcsin(np.clip(v[..., 2], -1.0, 1.0))
    h = np.degrees(alt)
    hc = np.clip(h, -1.5, 90.0)
    r = 1.02 / np.tan(np.radians(hc + 10.3 / (hc + 5.11))) / 60.0
    r = np.where(h > -2.0, np.maximum(r, 0.0), 0.0)
    new = alt + np.radians(r)
    horiz = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    k = np.cos(new) / np.maximum(horiz, 1e-12)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0] * k
    out[..., 1] = v[..., 1] * k
    out[..., 2] = np.sin(new)
    return out


def airmass(alt):
    h = np.degrees(np.maximum(alt, 0.0))
    return 1.0 / (np.sin(np.radians(h)) + 0.50572 * (h + 6.07995) ** -1.6364)


def solve_kepler(m, e):
    m = (m + math.pi) % (2 * math.pi) - math.pi
    ee = m if e < 0.8 else math.pi * (1 if m >= 0 else -1)
    for _ in range(60):
        d = (ee - e * math.sin(ee) - m) / (1 - e * math.cos(ee))
        ee -= d
        if abs(d) < 1e-12:
            break
    return ee


def orbit_to_ecl(xp, yp, w, om, i):
    cw, sw, co, so, ci, si = math.cos(w), math.sin(w), math.cos(om), math.sin(om), math.cos(i), math.sin(i)
    x = (cw * co - sw * so * ci) * xp + (-sw * co - cw * so * ci) * yp
    y = (cw * so + sw * co * ci) * xp + (-sw * so + cw * co * ci) * yp
    z = (sw * si) * xp + (cw * si) * yp
    return np.array([x, y, z])


def planet_helio(name, jd):
    el, rate = PLANETS[name]
    t = (jd - J2000) / 36525.0
    a, e, inc, lon, wbar, node = (el[k] + rate[k] * t for k in range(6))
    m = (lon - wbar) * DEG
    ee = solve_kepler(m, e)
    xp = a * (math.cos(ee) - e)
    yp = a * math.sqrt(1 - e * e) * math.sin(ee)
    ecl = orbit_to_ecl(xp, yp, (wbar - node) * DEG, node * DEG, inc * DEG)
    return ECL_TO_EQ @ ecl


def comet_elements(c, jd):
    return min(c["sets"], key=lambda x: abs(x["tp"] - jd))


def comet_helio(c, jd):
    c = comet_elements(c, jd)
    q, e = c["q"], c["e"]
    dt = jd - c["tp"]
    if abs(e - 1.0) < 1e-6:
        w = 3.0 * K_GAUSS * dt / math.sqrt(2.0 * q ** 3)
        y = (w / 2 + math.sqrt(w * w / 4 + 1)) ** (1 / 3)
        s = y - 1 / y
        nu = 2 * math.atan(s)
        r = q * (1 + s * s)
    elif e < 1.0:
        a = q / (1 - e)
        m = K_GAUSS / a ** 1.5 * dt
        ee = solve_kepler(m, e)
        nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(ee / 2), math.sqrt(1 - e) * math.cos(ee / 2))
        r = a * (1 - e * math.cos(ee))
    else:
        a = q / (e - 1)
        m = K_GAUSS / a ** 1.5 * dt
        hh = math.asinh(m / e)
        for _ in range(80):
            d = (e * math.sinh(hh) - hh - m) / (e * math.cosh(hh) - 1)
            hh -= d
            if abs(d) < 1e-12:
                break
        nu = 2 * math.atan(math.sqrt((e + 1) / (e - 1)) * math.tanh(hh / 2))
        r = a * (e * math.cosh(hh) - 1)
    ecl = orbit_to_ecl(r * math.cos(nu), r * math.sin(nu), c["w"] * DEG, c["om"] * DEG, c["i"] * DEG)
    return ECL_TO_EQ @ ecl


MOON_LR = [
    (0, 0, 1, 0, 6288774, -20905355), (2, 0, -1, 0, 1274027, -3699111), (2, 0, 0, 0, 658314, -2955968),
    (0, 0, 2, 0, 213618, -569925), (0, 1, 0, 0, -185116, 48888), (0, 0, 0, 2, -114332, -3149),
    (2, 0, -2, 0, 58793, 246158), (2, -1, -1, 0, 57066, -152138), (2, 0, 1, 0, 53322, -170733),
    (2, -1, 0, 0, 45758, -204586), (0, 1, -1, 0, -40923, -129620), (1, 0, 0, 0, -34720, 108743),
    (0, 1, 1, 0, -30383, 104755), (2, 0, 0, -2, 15327, 10321), (0, 0, 1, 2, -12528, 0),
    (0, 0, 1, -2, 10980, 79661), (4, 0, -1, 0, 10675, -34782), (0, 0, 3, 0, 10034, -23210),
    (4, 0, -2, 0, 8548, -21636), (2, 1, -1, 0, -7888, 24208), (2, 1, 0, 0, -6766, 30824),
    (1, 0, -1, 0, -5163, -8379), (1, 1, 0, 0, 4987, -16675), (2, -1, 1, 0, 4036, -12831),
    (2, 0, 2, 0, 3994, -10445), (4, 0, 0, 0, 3861, -11650), (2, 0, -3, 0, 3665, 14403),
    (0, 1, -2, 0, -2689, -7003), (2, 0, -1, 2, -2602, 0), (2, -1, -2, 0, 2390, 10056),
    (1, 0, 1, 0, -2348, 6322), (2, -2, 0, 0, 2236, -9884), (0, 1, 2, 0, -2120, 5751),
    (0, 2, 0, 0, -2069, 0), (2, -2, -1, 0, 2048, -4950), (2, 0, 1, -2, -1773, 4130),
    (2, 0, 0, 2, -1595, 0), (4, -1, -1, 0, 1215, -3958), (0, 0, 2, 2, -1110, 0),
    (3, 0, -1, 0, -892, 3258), (2, 1, 1, 0, -810, 2616), (4, -1, -2, 0, 759, -1897),
    (0, 2, -1, 0, -713, -2117), (2, 2, -1, 0, -700, 2354), (2, 1, -2, 0, 691, 0),
    (2, -1, 0, -2, 596, 0), (4, 0, 1, 0, 549, -1423), (0, 0, 4, 0, 537, -1117),
    (4, -1, 0, 0, 520, -1571), (1, 0, -2, 0, -487, -1739), (2, 1, 0, -2, -399, 0),
    (0, 0, 2, -2, -381, -4421), (1, 1, 1, 0, 351, 0), (3, 0, -2, 0, -340, 0),
    (4, 0, -3, 0, 330, 0), (2, -1, 2, 0, 327, 0), (0, 2, 1, 0, -323, 1165),
    (1, 1, -1, 0, 299, 0), (2, 0, 3, 0, 294, 0), (2, 0, -1, -2, 0, 8752),
]

MOON_B = [
    (0, 0, 0, 1, 5128122), (0, 0, 1, 1, 280602), (0, 0, 1, -1, 277693), (2, 0, 0, -1, 173237),
    (2, 0, -1, 1, 55413), (2, 0, -1, -1, 46271), (2, 0, 0, 1, 32573), (0, 0, 2, 1, 17198),
    (2, 0, 1, -1, 9266), (0, 0, 2, -1, 8822), (2, -1, 0, -1, 8216), (2, 0, -2, -1, 4324),
    (2, 0, 1, 1, 4200), (2, 1, 0, -1, -3359), (2, -1, -1, 1, 2463), (2, -1, 0, 1, 2211),
    (2, -1, -1, -1, 2065), (0, 1, -1, -1, -1870), (4, 0, -1, -1, 1828), (0, 1, 0, 1, -1794),
    (0, 0, 0, 3, -1749), (0, 1, -1, 1, -1565), (1, 0, 0, 1, -1491), (0, 1, 1, 1, -1475),
    (0, 1, 1, -1, -1410), (0, 1, 0, -1, -1344), (1, 0, 0, -1, -1335), (0, 0, 3, 1, 1107),
    (4, 0, 0, -1, 1021), (4, 0, -1, 1, 833), (0, 0, 1, -3, 777), (4, 0, -2, 1, 671),
    (2, 0, 0, -3, 607), (2, 0, 2, -1, 596), (2, -1, 1, -1, 491), (2, 0, -2, 1, -451),
    (0, 0, 3, -1, 439), (2, 0, 2, 1, 422), (2, 0, -3, -1, 421), (2, 1, -1, 1, -366),
    (2, 1, 0, 1, -351), (4, 0, 0, 1, 331), (2, -1, 1, 1, 315), (2, -2, 0, -1, 302),
    (0, 0, 1, 3, -283), (2, 1, 1, -1, -229), (1, 1, 0, -1, 223), (1, 1, 0, 1, 223),
    (0, 1, -2, -1, -220), (2, 1, -1, -1, -220), (1, 0, 1, 1, -185), (2, -1, -2, -1, 181),
    (0, 1, 2, 1, -177), (4, 0, -2, -1, 176), (4, -1, -1, -1, 166), (1, 0, -1, 1, -164),
    (4, 0, 1, -1, 132), (1, 0, -1, -1, -119), (4, -1, 0, -1, 115), (2, -2, 0, 1, 107),
]

_LR = np.array(MOON_LR, float)
_B = np.array(MOON_B, float)


def delta_t(jd):
    y = 2000.0 + (jd - J2000) / 365.25
    if y < -500 or y >= 2150:
        u = (y - 1820) / 100
        return -20 + 32 * u * u
    if y < 500:
        u = y / 100
        return (10583.6 - 1014.41 * u + 33.78311 * u ** 2 - 5.952053 * u ** 3 - 0.1798452 * u ** 4
                + 0.022174192 * u ** 5 + 0.0090316521 * u ** 6)
    if y < 1600:
        u = (y - 1000) / 100
        return (1574.2 - 556.01 * u + 71.23472 * u ** 2 + 0.319781 * u ** 3 - 0.8503463 * u ** 4
                - 0.005050998 * u ** 5 + 0.0083572073 * u ** 6)
    if y < 1700:
        t = y - 1600
        return 120 - 0.9808 * t - 0.01532 * t ** 2 + t ** 3 / 7129
    if y < 1800:
        t = y - 1700
        return 8.83 + 0.1603 * t - 0.0059285 * t ** 2 + 0.00013336 * t ** 3 - t ** 4 / 1174000
    if y < 1860:
        t = y - 1800
        return (13.72 - 0.332447 * t + 0.0068612 * t ** 2 + 0.0041116 * t ** 3 - 0.00037436 * t ** 4
                + 0.0000121272 * t ** 5 - 0.0000001699 * t ** 6 + 0.000000000875 * t ** 7)
    if y < 1900:
        t = y - 1860
        return 7.62 + 0.5737 * t - 0.251754 * t ** 2 + 0.01680668 * t ** 3 - 0.0004473624 * t ** 4 + t ** 5 / 233174
    if y < 1920:
        t = y - 1900
        return -2.79 + 1.494119 * t - 0.0598939 * t ** 2 + 0.0061966 * t ** 3 - 0.000197 * t ** 4
    if y < 1941:
        t = y - 1920
        return 21.20 + 0.84493 * t - 0.076100 * t ** 2 + 0.0020936 * t ** 3
    if y < 1961:
        t = y - 1950
        return 29.07 + 0.407 * t - t ** 2 / 233 + t ** 3 / 2547
    if y < 1986:
        t = y - 1975
        return 45.45 + 1.067 * t - t ** 2 / 260 - t ** 3 / 718
    if y < 2005:
        t = y - 2000
        return (63.86 + 0.3345 * t - 0.060374 * t ** 2 + 0.0017275 * t ** 3 + 0.000651814 * t ** 4
                + 0.00002373599 * t ** 5)
    if y < 2050:
        return 64.7 + 0.22 * (y - 2005)
    return -20 + 32 * ((y - 1820) / 100) ** 2 - 0.5628 * (2150 - y)


def moon_geo_of_date(jd_tt):
    t = (jd_tt - J2000) / 36525.0
    lp = 218.3164477 + 481267.88123421 * t - 0.0015786 * t ** 2 + t ** 3 / 538841 - t ** 4 / 65194000
    d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t ** 2 + t ** 3 / 545868 - t ** 4 / 113065000
    m = 357.5291092 + 35999.0502909 * t - 0.0001536 * t ** 2 + t ** 3 / 24490000
    mp = 134.9633964 + 477198.8675055 * t + 0.0087414 * t ** 2 + t ** 3 / 69699 - t ** 4 / 14712000
    f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t ** 2 - t ** 3 / 3526000 + t ** 4 / 863310000
    a1 = (119.75 + 131.849 * t) * DEG
    a2 = (53.09 + 479264.290 * t) * DEG
    a3 = (313.45 + 481266.484 * t) * DEG
    e = 1 - 0.002516 * t - 0.0000074 * t * t
    args = np.radians(np.array([d, m, mp, f]))
    ef = e ** np.abs(_LR[:, 1])
    arg = _LR[:, :4] @ args
    sl = float((_LR[:, 4] * ef * np.sin(arg)).sum())
    sr = float((_LR[:, 5] * ef * np.cos(arg)).sum())
    efb = e ** np.abs(_B[:, 1])
    sb = float((_B[:, 4] * efb * np.sin(_B[:, :4] @ args)).sum())
    lpr, fr, mpr = lp * DEG, f * DEG, mp * DEG
    sl += 3958 * math.sin(a1) + 1962 * math.sin(lpr - fr) + 318 * math.sin(a2)
    sb += (-2235 * math.sin(lpr) + 382 * math.sin(a3) + 175 * math.sin(a1 - fr) + 175 * math.sin(a1 + fr)
           + 127 * math.sin(lpr - mpr) - 115 * math.sin(lpr + mpr))
    lon = (lp + sl / 1e6) * DEG
    lat = (sb / 1e6) * DEG
    dist = (385000.56 + sr / 1000.0) / EARTH_R_KM
    ecl = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)]) * dist
    eps = (23.4392911 - (46.8150 * t + 0.00059 * t * t - 0.001813 * t ** 3) / 3600.0) * DEG
    return rot_x(-eps) @ ecl


def load_comets():
    p = DATA / "comets.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


@dataclass
class Body:
    key: str
    name: str
    kind: str
    vec: np.ndarray
    dist_au: float
    mag: float
    color: tuple
    radius_rad: float
    sun_dir: np.ndarray
    phase: float = 1.0
    extra: dict = field(default_factory=dict)


@dataclass
class Snapshot:
    jd: float
    lat: float
    lon: float
    lst: float
    hor: np.ndarray
    sun: Body
    moon: Body
    bodies: list
    eclipse: float = 0.0
    total: bool = False


def compute_snapshot(jd_ut, lat, lon, comets):
    jd = jd_ut + delta_t(jd_ut) / 86400.0
    lst = gmst(jd_ut) + lon
    prec = precession_matrix(jd)
    hor = horizon_matrix(lat) @ rot_z(lst) @ prec
    earth = planet_helio("Земля", jd)
    obs_date = EARTH_R_KM / AU_KM * np.array([math.cos(lat) * math.cos(lst), math.cos(lat) * math.sin(lst), math.sin(lat)])
    obs = prec.T @ obs_date
    sun_geo = -planet_helio("Земля", jd - LIGHT_DAY_PER_AU * float(np.linalg.norm(earth))) - obs

    def make(key, name, kind, geo, mag, color, radius_km, extra=None):
        dist = float(np.linalg.norm(geo))
        u = geo / dist
        v = refract((hor @ u)[None, :])[0]
        to_sun = sun_geo - geo
        n = float(np.linalg.norm(to_sun))
        if n < 1e-9:
            to_sun_h = np.zeros(3)
            phase = 1.0
        else:
            to_sun_h = hor @ (to_sun / n)
            phase = (1 + float(np.dot(-geo, to_sun) / (dist * n))) / 2
        return Body(key, name, kind, v, dist, mag, color, radius_km / (dist * AU_KM), to_sun_h, phase, extra or {})

    sun = make("sun", "Солнце", "sun", sun_geo, -26.74, (255, 240, 175), 695700.0)
    bodies = [sun]

    moon_date = moon_geo_of_date(jd) * (EARTH_R_KM / AU_KM)
    moon_geo = prec.T @ moon_date - obs
    to_sun = sun_geo - moon_geo
    i_deg = math.degrees(math.acos(np.clip(np.dot(-moon_geo, to_sun) / (np.linalg.norm(moon_geo) * np.linalg.norm(to_sun)), -1, 1)))
    moon_mag = -12.73 + 0.026 * i_deg + 4e-9 * i_deg ** 4
    moon = make("moon", "Луна", "moon", moon_geo, moon_mag, (235, 235, 225), 1737.4, {"waxing": None})
    ecl_n = ECL_TO_EQ @ np.array([0, 0, 1.0])
    moon.extra["waxing"] = float(np.dot(np.cross(sun_geo, moon_geo), ecl_n)) > 0
    bodies.append(moon)

    for name, (radius, color, eng) in PLANET_INFO.items():
        p = planet_helio(name, jd)
        geo = p - earth
        lt = np.linalg.norm(geo) * LIGHT_DAY_PER_AU
        p = planet_helio(name, jd - lt)
        geo = p - earth - obs
        r = float(np.linalg.norm(p))
        delta = float(np.linalg.norm(geo))
        cos_i = np.clip((r * r + delta * delta - np.dot(earth, earth)) / (2 * r * delta), -1, 1)
        i = math.degrees(math.acos(cos_i))
        mag = planet_mag(name, r, delta, i)
        if name == "Сатурн":
            sin_b = abs(float(np.dot(POLES["Сатурн"], geo / delta)))
            mag += -2.6 * sin_b + 1.25 * sin_b * sin_b
        bodies.append(make(name, name, "planet", geo, mag, color, radius, {"eng": eng, "r": r}))

    for c in comets:
        p = comet_helio(c, jd)
        geo = p - earth
        lt = np.linalg.norm(geo) * LIGHT_DAY_PER_AU
        p = comet_helio(c, jd - lt)
        geo = p - earth - obs
        r = float(np.linalg.norm(p))
        delta = float(np.linalg.norm(geo))
        mag = c["M1"] + 5 * math.log10(delta) + c["K1"] * math.log10(r)
        tail_len = min(0.5 / max(r, 0.05) ** 1.5, 1.5) * max(0.0, min(1.0, (9.0 - mag) / 6.0)) if r < 3.0 else 0.0
        tails = []
        if tail_len > 0:
            vel = p - comet_helio(c, jd - lt - 2.0)
            vel /= max(np.linalg.norm(vel), 1e-12)
            ion = geo + p / r * tail_len
            dust = geo + p / r * tail_len * 0.75 - vel * tail_len * 0.35
            for end, kind in ((dust, "dust"), (ion, "ion")):
                t = end / np.linalg.norm(end)
                tails.append((refract((hor @ t)[None, :])[0], kind))
        b = make(c["des"], c["name"], "comet", geo, mag, (200, 255, 225), 0.0,
                 {"des": c["des"], "r": r, "tails": tails})
        bodies.append(b)

    cov, total = eclipse(sun, moon)
    return Snapshot(jd_ut, lat, lon, lst, hor, sun, moon, bodies, cov, total)


def eclipse(sun, moon):
    d = math.acos(float(np.clip(np.dot(sun.vec, moon.vec), -1, 1)))
    rs, rm = sun.radius_rad, moon.radius_rad
    if d >= rs + rm:
        return 0.0, False
    if d <= abs(rm - rs):
        return min(1.0, (rm / rs) ** 2), rm >= rs
    a = (rs * rs * math.acos((d * d + rs * rs - rm * rm) / (2 * d * rs))
         + rm * rm * math.acos((d * d + rm * rm - rs * rs) / (2 * d * rm))
         - 0.5 * math.sqrt((-d + rs + rm) * (d + rs - rm) * (d - rs + rm) * (d + rs + rm)))
    return min(1.0, a / (math.pi * rs * rs)), False


def planet_mag(name, r, d, i):
    k = 5 * math.log10(r * d)
    if name == "Меркурий":
        return -0.42 + k + 0.0380 * i - 0.000273 * i * i + 0.000002 * i ** 3
    if name == "Венера":
        return -4.40 + k + 0.0009 * i + 0.000239 * i * i - 0.00000065 * i ** 3
    if name == "Марс":
        return -1.52 + k + 0.016 * i
    if name == "Юпитер":
        return -9.40 + k + 0.005 * i
    if name == "Сатурн":
        return -8.88 + k + 0.044 * i
    if name == "Уран":
        return -7.19 + k
    return -6.87 + k


class StarCatalog:
    def __init__(self):
        d = np.load(DATA / "stars.npz")
        self.ra = d["ra"]
        self.dec = d["dec"]
        self.pmra = d["pmra"].astype(np.float64)
        self.pmdec = d["pmdec"].astype(np.float64)
        self.mag = d["mag"]
        self.rgb = d["rgb"]
        self.dist = d["dist"]
        self.proper = d["proper"]
        self.bayer = d["bayer"]
        self.flam = d["flam"]
        self.con = d["con"]
        self.hip = d["hip"]
        self.base = radec_to_vec(self.ra, self.dec)
        sa, ca = np.sin(self.ra), np.cos(self.ra)
        sd, cd = np.sin(self.dec), np.cos(self.dec)
        self.e_ra = np.stack([-sa, ca, np.zeros_like(sa)], axis=-1)
        self.e_dec = np.stack([-sd * ca, -sd * sa, cd], axis=-1)
        self._cache_year = None
        self._cache_vec = self.base

    def vectors(self, jd):
        years = (jd - J2000) / 365.25
        key = round(years, 1)
        if key != self._cache_year:
            v = self.base + years * (self.pmra[:, None] * self.e_ra + self.pmdec[:, None] * self.e_dec)
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            self._cache_vec = v
            self._cache_year = key
        return self._cache_vec

    def count_brighter(self, mag):
        return int(np.searchsorted(self.mag, mag, side="right"))
