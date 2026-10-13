import math
import threading
import time
import urllib.request

import numpy as np

import astro

try:
    from sgp4.api import Satrec, SatrecArray
except ImportError:
    Satrec = SatrecArray = None

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle"
TLE_PATH = astro.DATA / "satellites.tle"
MAX_AGE = 86400.0
EPOCH_DAYS = 14.0
WGS_A = 6378.137
WGS_E2 = 0.00669438
EARTH_R = 6378.137
SHADOW_SOFT = 30.0
STD_MAG = {25544: -1.8, 48274: -1.0, 20580: 2.2}
NAMES = {25544: "МКС", 48274: "Тяньгун", 20580: "Хаббл"}
DEFAULT_MAG = 3.0
COLOR = (240, 238, 228)
PASS_ALT = 10.0
PASS_SUN = -6.0
PASS_STEP = 20.0

_lock = threading.Lock()
_busy = False


class Satellites:
    def __init__(self, path=TLE_PATH):
        self.ids, self.names, self.des = [], [], []
        self.std = np.zeros(0)
        self.epoch = np.zeros(0)
        self.sats = []
        self.array = None
        if Satrec is None or not path.exists():
            return
        lines = [line.rstrip() for line in path.read_text().splitlines() if line.strip()]
        for k in range(len(lines) - 2):
            name, l1, l2 = lines[k:k + 3]
            if not (l1.startswith("1 ") and l2.startswith("2 ")) or name.startswith(("1 ", "2 ")):
                continue
            try:
                s = Satrec.twoline2rv(l1, l2)
            except ValueError:
                continue
            self.sats.append(s)
            self.ids.append(s.satnum)
            self.des.append(name.strip())
            self.names.append(NAMES.get(s.satnum, name.strip()))
        if self.sats:
            self.array = SatrecArray(self.sats)
            self.std = np.array([STD_MAG.get(i, DEFAULT_MAG) for i in self.ids])
            self.epoch = np.array([s.jdsatepoch + s.jdsatepochF for s in self.sats])

    def __len__(self):
        return len(self.sats)

    def bodies(self, jd_ut, lat, lst, prec, sun_date, apparent):
        """Спутники для снимка неба: над горизонтом и именованные, только при свежих элементах орбиты."""
        if not self.sats:
            return []
        fresh = np.abs(self.epoch - jd_ut) < EPOCH_DAYS
        if not fresh.any():
            return []
        whole = math.floor(jd_ut)
        err, r, _ = self.array.sgp4(np.array([whole]), np.array([jd_ut - whole]))
        r = r[:, 0, :]
        ok = fresh & (err[:, 0] == 0)
        obs = observer_km(lat, lst)
        up = zenith(lat, lst)
        geo = r - obs
        rng = np.linalg.norm(geo, axis=1)
        alt = np.degrees(np.arcsin(np.clip((geo @ up) / np.maximum(rng, 1e-9), -1, 1)))
        mag, light = brightness(r, geo, rng, sun_date, self.std)
        out = []
        for i in np.nonzero(ok & ((alt > -2.0) | np.isin(self.ids, list(NAMES))))[0]:
            u = prec.T @ (geo[i] / rng[i])
            extra = {"des": self.des[i], "norad": self.ids[i], "index": int(i), "range": float(rng[i]),
                     "height": float(np.linalg.norm(r[i]) - EARTH_R), "light": float(light[i])}
            out.append(astro.Body(f"sat:{self.ids[i]}", self.names[i], "sat", apparent(u), float(rng[i] / astro.AU_KM),
                                  float(mag[i]), COLOR, 0.0, np.zeros(3), 1.0, extra))
        return out

    def next_pass(self, i, jd_ut, lat, lon, days=2.0):
        """Ближайший видимый пролёт: спутник выше 10°, освещён Солнцем, а у наблюдателя Солнце ниже −6°."""
        t = jd_ut + np.arange(0.0, days * 86400.0, PASS_STEP) / 86400.0
        whole = np.floor(t)
        err, r, _ = self.sats[i].sgp4_array(whole, t - whole)
        lst = astro.gmst(t) + lon
        obs = observer_km(lat, lst)
        up = zenith(lat, lst)
        geo = r - obs
        rng = np.linalg.norm(geo, axis=1)
        alt = np.degrees(np.arcsin(np.clip((geo * up).sum(1) / rng, -1, 1)))
        sun = sun_of_date(t)
        sun_alt = np.degrees(np.arcsin(np.clip((sun * up).sum(1), -1, 1)))
        mag, light = brightness(r, geo, rng, sun, np.full(t.size, self.std[i]))
        vis = (err == 0) & (alt > PASS_ALT) & (sun_alt < PASS_SUN) & (light > 0.5)
        idx = np.nonzero(vis)[0]
        if not idx.size:
            return None
        a = idx[0]
        b = a
        while b + 1 < t.size and vis[b + 1]:
            b += 1
        seg = slice(a, b + 1)
        top = a + int(np.argmax(alt[seg]))
        east = np.stack([-np.sin(lst), np.cos(lst), np.zeros_like(lst)], axis=1)
        north = np.stack([-math.sin(lat) * np.cos(lst), -math.sin(lat) * np.sin(lst),
                          np.full(lst.size, math.cos(lat))], axis=1)
        az = np.degrees(np.arctan2((geo * east).sum(1), (geo * north).sum(1))) % 360.0
        return {"start": float(t[a]), "end": float(t[b]), "max_alt": float(alt[top]),
                "az_start": float(az[a]), "az_end": float(az[b]), "mag": float(mag[seg].min())}


def observer_km(lat, lst):
    s, c = math.sin(lat), math.cos(lat)
    n = WGS_A / math.sqrt(1 - WGS_E2 * s * s)
    lst = np.asarray(lst, float)
    return np.stack([n * c * np.cos(lst), n * c * np.sin(lst), np.full(lst.shape, n * (1 - WGS_E2) * s)], axis=-1)


def zenith(lat, lst):
    lst = np.asarray(lst, float)
    s, c = math.sin(lat), math.cos(lat)
    return np.stack([c * np.cos(lst), c * np.sin(lst), np.full(lst.shape, s)], axis=-1)


def sun_of_date(jd):
    t = (np.asarray(jd, float) - astro.J2000) / 36525.0
    l0 = np.radians(280.46646 + 36000.76983 * t)
    m = np.radians(357.52911 + 35999.05029 * t)
    c = np.radians((1.914602 - 0.004817 * t) * np.sin(m) + 0.019993 * np.sin(2 * m) + 0.000289 * np.sin(3 * m))
    lam = l0 + c
    eps = np.radians(23.439291 - 0.0130042 * t)
    return np.stack([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)], axis=-1)


def brightness(r, geo, rng, sun, std):
    """Блеск по модели диффузной сферы и доля солнечного света с мягким краем тени Земли."""
    along = (r * sun).sum(-1)
    perp = np.linalg.norm(r - along[..., None] * sun, axis=-1)
    t = np.clip((perp - EARTH_R) / SHADOW_SOFT, 0, 1)
    light = np.where(along > 0, 1.0, t * t * (3 - 2 * t))
    cos_phase = np.clip((-geo * sun).sum(-1) / np.maximum(rng, 1e-9), -1, 1)
    phase = np.arccos(cos_phase)
    k = np.maximum(np.sin(phase) + (math.pi - phase) * cos_phase, 1e-3)
    mag = std + 5 * np.log10(rng / 1000.0) - 2.5 * np.log10(k) - 2.5 * np.log10(np.maximum(light, 1e-6))
    return np.where(light > 0.01, mag, 99.0), light


def stale(path=TLE_PATH):
    try:
        return time.time() - path.stat().st_mtime > MAX_AGE
    except OSError:
        return True


def download(path=TLE_PATH):
    with urllib.request.urlopen(TLE_URL, timeout=20) as r:
        text = r.read().decode()
    if "ISS" not in text:
        raise ValueError("пустой ответ CelesTrak")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def refresh_async(done):
    """Раз в сутки обновляет элементы орбит в фоне и передаёт новые спутники в done."""
    global _busy
    if Satrec is None or not stale():
        return
    with _lock:
        if _busy:
            return
        _busy = True

    def work():
        global _busy
        try:
            download()
            done(Satellites())
        except Exception:
            pass
        finally:
            _busy = False

    threading.Thread(target=work, daemon=True).start()
