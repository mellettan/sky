import csv
import json
import math
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

import satellites

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
HYG_URL = "https://raw.githubusercontent.com/astronexus/HYG-Database/main/hyg/CURRENT/hygdata_v41.csv"
SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
SKYCULTURE_URL = "https://raw.githubusercontent.com/Stellarium/stellarium/v25.3/skycultures/modern/index.json"
MAG_LIMIT = 11.0

COMETS = [
    ("1P", "Комета Галлея", None, None),
    ("2P", "Комета Энке", None, None),
    ("12P", "Комета Понса — Брукса", 5.0, 11.0),
    ("46P", "Комета Виртанена", None, None),
    ("55P", "Комета Темпеля — Таттла", None, None),
    ("67P", "Комета Чурюмова — Герасименко", None, None),
    ("109P", "Комета Свифта — Таттла", 4.0, 10.0),
    ("C/1965 S1-A", "Комета Икэя — Сэки", 3.5, 10.0),
    ("C/1975 V1-A", "Комета Уэста", 1.5, 10.0),
    ("C/1995 O1", "Комета Хейла — Боппа", -0.8, 7.5),
    ("C/1996 B2", "Комета Хякутакэ", 5.0, 9.0),
    ("C/2006 P1", "Комета Макнота", 5.5, 10.0),
    ("C/2011 L4", "Комета PANSTARRS", 5.5, 9.0),
    ("C/2020 F3", "Комета NEOWISE", 6.5, 9.0),
    ("C/2022 E3", "Комета ZTF", 7.0, 10.0),
    ("C/2023 A3", "Комета Цзыцзиньшань — ATLAS", 4.5, 10.0),
    ("C/2025 A6", "Комета Леммона", 7.5, 10.0),
]


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def bv_to_rgb(bv):
    bv = min(max(bv, -0.4), 2.0)
    t = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62)) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    rgb = np.clip([r, g, b], 0, 255)
    gray = rgb.mean()
    rgb = gray + (rgb - gray) * 0.75
    rgb = rgb * (255.0 / rgb.max())
    return rgb


def build_stars():
    src = RAW / "hygdata_v41.csv"
    if not src.exists():
        download(HYG_URL, src)
    cols = {k: [] for k in ("ra", "dec", "pmra", "pmdec", "mag", "rgb", "dist", "proper", "bayer", "flam", "con", "hip")}
    with open(src, newline="") as f:
        for row in csv.DictReader(f):
            if row["id"] == "0" or not row["mag"]:
                continue
            mag = float(row["mag"])
            if mag > MAG_LIMIT:
                continue
            ci = float(row["ci"]) if row["ci"] else 0.6
            cols["ra"].append(math.radians(float(row["ra"]) * 15.0))
            cols["dec"].append(math.radians(float(row["dec"])))
            mas = math.radians(1.0 / 3.6e6)
            cols["pmra"].append(float(row["pmra"] or 0) * mas)
            cols["pmdec"].append(float(row["pmdec"] or 0) * mas)
            cols["mag"].append(mag)
            cols["rgb"].append(bv_to_rgb(ci))
            d = float(row["dist"] or 0)
            cols["dist"].append(d if 0 < d < 99999 else 0.0)
            cols["proper"].append(row["proper"])
            cols["bayer"].append(row["bayer"])
            cols["flam"].append(row["flam"])
            cols["con"].append(row["con"])
            cols["hip"].append(row["hip"])
    order = np.argsort(np.array(cols["mag"]), kind="stable")
    out = {}
    for k, v in cols.items():
        arr = np.array(v, dtype=np.uint8 if k == "rgb" else None)
        if k in ("ra", "dec", "pmra", "pmdec", "mag", "dist"):
            arr = arr.astype(np.float64 if k in ("ra", "dec") else np.float32)
        out[k] = arr[order]
    np.savez_compressed(DATA / "stars.npz", **out)
    print(f"Звёзд в каталоге: {len(order)}")


def sbdb(designation):
    q = urllib.parse.urlencode({"sstr": designation, "phys-par": 1, "full-prec": 1})
    with urllib.request.urlopen(f"{SBDB_URL}?{q}", timeout=30) as r:
        d = json.load(r)
    if "list" in d and "orbit" not in d:
        pdes = d["list"][0]["pdes"]
        return sbdb(pdes)
    return d


def horizons_elements(spkid, jd):
    q = urllib.parse.urlencode({
        "format": "text",
        "COMMAND": f"'DES={spkid};CAP;'",
        "EPHEM_TYPE": "'ELEMENTS'",
        "CENTER": "'500@10'",
        "TLIST": f"'{jd:.5f}'",
        "REF_PLANE": "'ECLIPTIC'",
        "REF_SYSTEM": "'J2000'",
        "OUT_UNITS": "'AU-D'",
    })
    with urllib.request.urlopen(f"{HORIZONS_URL}?{q}", timeout=60) as r:
        txt = r.read().decode()
    body = txt[txt.index("$$SOE"):txt.index("$$EOE")]
    num = r"\s*([-+0-9.E]+)"
    get = lambda key: float(re.search(re.escape(key) + num, body).group(1))
    return {"q": get("QR="), "e": get("EC="), "i": get("IN="), "om": get("OM="), "w": get("W ="), "tp": get("Tp=")}


def jd_of_year(y):
    return 2451545.0 + (y - 2000.0) * 365.25


def apparition_epochs(tp, e, q):
    if e >= 1.0:
        return [tp]
    period = 365.25 * (q / (1 - e)) ** 1.5
    if period > 300 * 365.25:
        return [tp]
    lo, hi = (1980, 2060) if period < 20 * 365.25 else (1750, 2150)
    out = []
    k = math.floor((jd_of_year(lo) - tp) / period)
    while tp + k * period <= jd_of_year(hi):
        t = tp + k * period
        if t >= jd_of_year(lo):
            out.append(t)
        k += 1
    return out or [tp]


def build_comets():
    out = []
    for des, name, m1, k1 in COMETS:
        try:
            d = sbdb(des)
            el = {e["name"]: e["value"] for e in d["orbit"]["elements"]}
            phys = {p["name"]: p["value"] for p in d.get("phys_par", [])}
            spkid = d["object"]["spkid"]
            base = {k: float(el[k]) for k in ("q", "e", "i", "om", "w", "tp")}
            sets = []
            for t in apparition_epochs(base["tp"], base["e"], base["q"]):
                try:
                    s1 = horizons_elements(spkid, t)
                    s2 = horizons_elements(spkid, s1["tp"])
                    sets.append(s2)
                except Exception:
                    pass
            if not sets:
                sets = [base]
            out.append({
                "des": d["object"]["fullname"].strip(),
                "name": name,
                "sets": sorted(sets, key=lambda x: x["tp"]),
                "M1": m1 if m1 is not None else float(phys.get("M1", 10.0)),
                "K1": k1 if k1 is not None else float(phys.get("K1", 10.0)),
            })
            print(f"  {out[-1]['des']}: {len(sets)} набор(а) элементов")
        except Exception as ex:
            print(f"  {des}: не удалось ({ex})")
    (DATA / "comets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"Комет: {len(out)}")


def build_constellations():
    src = RAW / "skyculture_modern.json"
    if not src.exists():
        download(SKYCULTURE_URL, src)
    out = []
    for c in json.loads(src.read_text())["constellations"]:
        item = {"con": c["id"].split()[-1], "name": c["common_name"]["native"], "lines": c["lines"]}
        if c.get("label_positions"):
            item["labels"] = c["label_positions"]
        out.append(item)
    (DATA / "constellations.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"Созвездий: {len(out)}")


def build_satellites():
    satellites.download()
    print(f"Спутников: {len(satellites.Satellites())}")


def main():
    DATA.mkdir(exist_ok=True)
    what = sys.argv[1:] or ["stars", "comets", "constellations", "satellites"]
    if "stars" in what:
        build_stars()
    if "comets" in what:
        build_comets()
    if "constellations" in what:
        build_constellations()
    if "satellites" in what:
        build_satellites()


if __name__ == "__main__":
    main()
