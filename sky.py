import argparse
import codecs
import json
import math
import os
import re
import select
import signal
import sys
import termios
import time
import tty
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import astro
import names
import render

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.json"
LOCAL_TZ = datetime.now().astimezone().tzinfo
STATUS_ROWS = 2

SPEEDS = [1, 10, 60, 300, 1800, 3600, 10800, 86400, 604800, 2629800, 31557600]
SPEED_NAMES = {
    1: "реальное время", 10: "10 с/с", 60: "1 мин/с", 300: "5 мин/с", 1800: "30 мин/с", 3600: "1 ч/с",
    10800: "3 ч/с", 86400: "1 сут/с", 604800: "1 нед/с", 2629800: "1 мес/с", 31557600: "1 год/с",
}

BAR_BG = (16, 17, 26)
BAR_FG = (170, 174, 190)
ACCENT = (255, 214, 120)
DIM = (110, 114, 130)
TIP_BG = (24, 26, 40)
TIP_FG = (232, 234, 240)
LABEL_STAR = (138, 152, 190)
LABEL_PLANET = (228, 196, 136)
LABEL_LUM = (222, 222, 210)
LABEL_COMET = (140, 196, 236)
LABEL_CON = (92, 128, 196)
LABEL_MAX = 300

HELP = [
    "Звёздное небо — управление",
    "",
    "Мышь: тянуть — повернуть взгляд, колесо — зум,",
    "      навести — подпись, клик — выбрать объект",
    "Стрелки        повернуть взгляд (Shift — быстрее)",
    "+ / -          зум, z — сбросить зум",
    "g              ландшафт вкл/выкл",
    "a              атмосфера вкл/выкл",
    "l              подписи вкл/выкл",
    "k              созвездия вкл/выкл",
    "[ ] или 1..9   засветка (шкала Бортля)",
    "Пробел         пауза / ход времени",
    ", .            медленнее / быстрее (можно назад)",
    "< >            час назад / вперёд",
    "PgUp PgDn      сутки назад / вперёд",
    "n              сейчас",
    "t              ввести дату и время",
    "c              ввести координаты или город",
    "/              найти объект (Юпитер, Вега, Орион…)",
    "f              следить за выбранным объектом",
    "Esc            снять выбор / закрыть",
    "? или h        эта справка,  q — выход",
]


def load_config():
    cfg = {"lat": 55.7558, "lon": 37.6173, "place": "Москва", "bortle": 5, "landscape": True, "atmosphere": True,
           "labels": True, "constellations": False}
    try:
        cfg.update(json.loads(CONFIG.read_text()))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg):
    try:
        CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    except OSError:
        pass


def parse_coords(text):
    t = text.strip().lower()
    if t in names.CITIES:
        lat, lon = names.CITIES[t]
        return lat, lon, text.strip().capitalize() if t != "спб" else "Санкт-Петербург"
    t = t.replace("°", " ").strip()
    parts = [p for p in re.split(r"[\s;]+", t) if p]
    if len(parts) == 1 and "," in parts[0]:
        parts = parts[0].split(",")
    if len(parts) != 2:
        raise ValueError("нужно две координаты: широта и долгота")
    vals = []
    for p in parts:
        m = re.fullmatch(r"([-+]?\d+(?:[.,]\d+)?)([nsewсювз]?)", p.strip().rstrip(","))
        if not m:
            raise ValueError(f"не понял «{p}»")
        v = float(m.group(1).replace(",", "."))
        if m.group(2) in ("s", "w", "ю", "з"):
            v = -v
        vals.append(v)
    lat, lon = vals
    if not (-90 <= lat <= 90 and -180 <= lon <= 360):
        raise ValueError("координаты вне диапазона")
    lon = (lon + 180) % 360 - 180
    return lat, lon, fmt_coords(lat, lon)


def fmt_coords(lat, lon):
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'} {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"


def parse_time(text, now_local):
    t = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M",
                "%d.%m.%Y"):
        try:
            return datetime.strptime(t, fmt).replace(tzinfo=LOCAL_TZ)
        except ValueError:
            pass
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            hm = datetime.strptime(t, fmt)
            return now_local.replace(hour=hm.hour, minute=hm.minute, second=hm.second, microsecond=0)
        except ValueError:
            pass
    raise ValueError("формат: 2026-10-12 21:30, 12.10.2026 21:30 или 21:30")


class Terminal:
    def __init__(self, mouse=True):
        self.mouse = mouse
        self.fd = sys.stdin.fileno()
        self.old = None
        self.decoder = codecs.getincrementaldecoder("utf-8")("ignore")
        self.buf = ""
        self.default_bg = None

    def __enter__(self):
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd)
        self.default_bg = self.query_bg()
        self.write("\x1b[?1049h\x1b[?25l" + ("\x1b[?1003h\x1b[?1006h" if self.mouse else "") + "\x1b[2J")
        return self

    def query_bg(self):
        self.write("\x1b]11;?\x1b\\\x1b[c")
        text = ""
        end = time.time() + 0.5
        while time.time() < end and not re.search(r"\x1b\[\?[\d;]*c", text):
            r, _, _ = select.select([self.fd], [], [], max(0.0, end - time.time()))
            if not r:
                break
            text += self.decoder.decode(os.read(self.fd, 4096))
        m = re.search(r"\x1b\]11;rgb:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)", text)
        text = re.sub(r"\x1b\]11;[^\x07\x1b]*(\x07|\x1b\\)?|\x1b\[\?[\d;]*c", "", text)
        self.buf += text
        if not m:
            return None
        return tuple(round(int(h, 16) * 255 / (16 ** len(h) - 1)) for h in m.groups())

    def __exit__(self, *exc):
        self.write("\x1b[?1003l\x1b[?1006l\x1b[0m\x1b[?25h\x1b[?1049l")
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def write(self, s):
        data = s.encode()
        while data:
            n = os.write(sys.stdout.fileno(), data)
            data = data[n:]

    def size(self):
        s = os.get_terminal_size()
        return s.columns, s.lines

    def read_events(self, timeout):
        r, _, _ = select.select([self.fd], [], [], timeout)
        if not r:
            return []
        chunks = []
        while True:
            chunks.append(os.read(self.fd, 65536))
            r, _, _ = select.select([self.fd], [], [], 0)
            if not r:
                break
        self.buf += self.decoder.decode(b"".join(chunks))
        return self._parse()

    def _parse(self):
        ev = []
        s = self.buf
        i = 0
        while i < len(s):
            c = s[i]
            if c == "\x1b":
                if i + 1 >= len(s):
                    ev.append(("key", "esc"))
                    i += 1
                    continue
                m = re.match(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])", s[i:])
                if m:
                    ev.append(("mouse", int(m.group(1)), int(m.group(2)) - 1, int(m.group(3)) - 1, m.group(4) == "M"))
                    i += m.end()
                    continue
                m = re.match(r"\x1b[\[O]([\d;]*)([A-Za-z~])", s[i:])
                if m:
                    params, final = m.group(1), m.group(2)
                    mods = int(params.split(";")[1]) if ";" in params else 1
                    key = {"A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end"}.get(final)
                    if final == "~":
                        key = {"5": "pgup", "6": "pgdn", "1": "home", "4": "end", "3": "del"}.get(params.split(";")[0])
                    if key:
                        ev.append(("key", ("S-" if mods == 2 else "") + key))
                    i += m.end()
                    continue
                if re.match(r"\x1b[\[O][\d;<]*$", s[i:]):
                    break
                ev.append(("key", "esc"))
                i += 1
                continue
            if c in ("\r", "\n"):
                ev.append(("key", "enter"))
            elif c in ("\x7f", "\x08"):
                ev.append(("key", "backspace"))
            elif c == "\x03":
                ev.append(("key", "ctrl-c"))
            elif c == "\t":
                ev.append(("key", "tab"))
            elif c >= " ":
                ev.append(("char", c))
            i += 1
        self.buf = s[i:]
        return ev


class Screen:
    """Помнит, что уже выведено в терминал, и выводит только изменившиеся ячейки."""

    GAP = 3

    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None

    def render(self, chars, fg, bg, avoid=None):
        if avoid is not None:
            same = (bg == np.array(avoid, np.uint8)).all(axis=-1)
            bg[same, 2] = avoid[2] - 1 if avoid[2] == 255 else avoid[2] + 1
        fgp = (fg[..., 0].astype(np.int32) << 16) | (fg[..., 1].astype(np.int32) << 8) | fg[..., 2]
        bgp = (bg[..., 0].astype(np.int32) << 16) | (bg[..., 1].astype(np.int32) << 8) | bg[..., 2]
        prev = self.prev
        if prev is None or prev[0].shape != chars.shape:
            changed = np.ones(chars.shape, bool)
        else:
            changed = (chars != prev[0]) | (fgp != prev[1]) | (bgp != prev[2])
        self.prev = (chars.copy(), fgp, bgp)
        rows = np.nonzero(changed.any(axis=1))[0]
        if not rows.size:
            return ""
        out = ["\x1b[?2026h"]
        lf = lb = -1
        for y in rows.tolist():
            xs = np.nonzero(changed[y])[0]
            cut = np.nonzero(np.diff(xs) > self.GAP + 1)[0]
            starts, ends = np.r_[xs[0], xs[cut + 1]].tolist(), np.r_[xs[cut], xs[-1]].tolist()
            rf, rb, rc = fgp[y].tolist(), bgp[y].tolist(), chars[y].tolist()
            for x0, x1 in zip(starts, ends):
                out.append(f"\x1b[{y + 1};{x0 + 1}H")
                for x in range(x0, x1 + 1):
                    b = rb[x]
                    if b != lb:
                        out.append(f"\x1b[48;2;{b >> 16};{(b >> 8) & 255};{b & 255}m")
                        lb = b
                    c = rc[x]
                    f = rf[x]
                    if c == 32 or (c == 0x2580 and f == b):
                        out.append(" ")
                        continue
                    if f != lf:
                        out.append(f"\x1b[38;2;{f >> 16};{(f >> 8) & 255};{f & 255}m")
                        lf = f
                    out.append(chr(c))
        out.append("\x1b[0m\x1b[?2026l")
        return "".join(out)


class Canvas:
    def __init__(self, chars, fg, bg):
        self.chars, self.fg, self.bg = chars, fg, bg
        self.rows, self.cols = chars.shape

    def put(self, row, col, text, fg=None, bg=None):
        if not (0 <= row < self.rows):
            return col
        for ch in text:
            if 0 <= col < self.cols:
                self.chars[row, col] = ord(ch)
                if fg is not None:
                    self.fg[row, col] = fg
                if bg is not None:
                    self.bg[row, col] = bg
            col += 1
        return col

    def fill(self, row, col, width, bg):
        for x in range(max(col, 0), min(col + width, self.cols)):
            self.chars[row, x] = 32
            self.bg[row, x] = bg


class App:
    def __init__(self, term, wall=None):
        self.term = term
        self.wall = wall
        self.status_rows = 0 if wall else STATUS_ROWS
        self.cfg = load_config()
        self.cat = astro.StarCatalog()
        self.comets = astro.load_comets()
        self.cons = astro.Constellations(self.cat)
        self.star_index = self.build_star_index()
        self.renderer = render.Renderer(self.cat, self.cons)
        self.view = render.View(bortle=int(self.cfg["bortle"]), landscape=bool(self.cfg["landscape"]),
                                atmosphere=bool(self.cfg["atmosphere"]), labels=bool(self.cfg["labels"]),
                                constellations=bool(self.cfg["constellations"]) and len(self.cons) > 0)
        self.lat = float(self.cfg["lat"])
        self.lon = float(self.cfg["lon"])
        self.place = self.cfg["place"]
        self.speed = 1
        self.paused = False
        self.anchor_jd = astro.jd_from_datetime(datetime.now(timezone.utc))
        self.anchor_real = time.time()
        self.follow = False
        self.mouse = None
        self.drag = None
        self.help = False
        self.prompt = None
        self.message = None
        self.frame = None
        self.snap = None
        self.slider = (0, 0)
        self.dirty = True
        self.last_render = 0.0
        self.render_cost = 0.0
        self.running = True
        self.size = (0, 0)
        self.screen = Screen()
        if wall:
            v = self.view
            v.az, v.alt, v.fov, v.bortle = wall.az % 360, wall.alt, wall.fov, wall.bortle
            v.labels = not wall.no_labels
            v.landscape = not wall.no_landscape
            v.constellations = wall.constellations and len(self.cons) > 0
            v.shift = 0.5 - wall.horizon
            if wall.place:
                self.lat, self.lon, self.place = parse_coords(wall.place)

    def jd(self):
        if self.paused:
            return self.anchor_jd
        return astro.clamp_jd(self.anchor_jd + (time.time() - self.anchor_real) * self.speed / 86400.0)

    def set_jd(self, jd):
        self.anchor_jd = astro.clamp_jd(jd)
        self.anchor_real = time.time()
        self.dirty = "full"

    def set_speed(self, speed):
        self.set_jd(self.jd())
        self.speed = speed
        self.paused = False

    def step_speed(self, direction):
        s = self.speed if not self.paused else 0
        ladder = [-v for v in reversed(SPEEDS)] + SPEEDS
        if s == 0:
            self.set_speed(1 if direction > 0 else -1)
            return
        i = ladder.index(s) if s in ladder else ladder.index(1)
        i = min(max(i + direction, 0), len(ladder) - 1)
        self.set_speed(ladder[i])

    def notify(self, text, secs=3.0):
        self.message = (text, time.time() + secs)
        self.dirty = True

    def save(self):
        if self.wall:
            return
        self.cfg.update({"lat": self.lat, "lon": self.lon, "place": self.place, "bortle": self.view.bortle,
                         "landscape": self.view.landscape, "atmosphere": self.view.atmosphere,
                         "labels": self.view.labels, "constellations": self.view.constellations})
        save_config(self.cfg)

    def frame_interval(self):
        if self.wall:
            return self.wall.interval
        if self.paused:
            return 5.0
        cam_px = self.view.fov / max(self.size[0] * 2, 1)
        rate = abs(self.speed) * 360.0 / 86164.0
        interval = float(np.clip(cam_px / max(rate, 1e-9) * 1.5, 0.04, 1.0))
        if abs(self.speed) > 1:
            interval = min(interval, 0.08)
        return max(interval, self.render_cost * 1.15)

    def run(self):
        while self.running:
            size = self.term.size()
            if size != self.size:
                self.size = size
                self.screen.reset()
                self.dirty = "full"
            now = time.time()
            if self.message and now > self.message[1]:
                self.message = None
                self.dirty = True
            due = now - self.last_render >= self.frame_interval()
            if self.dirty or due:
                self.draw(full=due or self.frame is None or self.dirty == "full")
                self.dirty = False
            wait = max(0.0, self.frame_interval() - (time.time() - self.last_render))
            for ev in self.term.read_events(min(wait, 0.5)):
                self.handle(ev)

    def snapshot(self):
        jd = self.jd()
        self.snap = astro.compute_snapshot(jd, math.radians(self.lat), math.radians(self.lon), self.comets,
                                           self.view.atmosphere)
        return self.snap

    def draw(self, full=True):
        cols, rows = self.size
        sky_rows = max(rows - self.status_rows, 4)
        full = full or self.frame is None
        t0 = time.time()
        if full:
            self.last_render = t0
            snap = self.snapshot()
            if self.follow and self.view.selected:
                pos = self.object_altaz(self.view.selected)
                if pos:
                    self.view.az, self.view.alt = pos
            ut = astro.datetime_from_jd(snap.jd)
            solar = (ut.hour + ut.minute / 60.0 + self.lon / 15.0) % 24.0
            self.frame = self.renderer.render(self.view, snap, cols, sky_rows, solar)
        f = self.frame
        chars = np.full((rows, cols), 32, np.int32)
        fg = np.zeros((rows, cols, 3), np.uint8)
        bg = np.zeros((rows, cols, 3), np.uint8)
        h = min(sky_rows, f.chars.shape[0])
        w = min(cols, f.chars.shape[1])
        chars[:h, :w], fg[:h, :w], bg[:h, :w] = f.chars[:h, :w], f.fg[:h, :w], f.bg[:h, :w]
        cv = Canvas(chars, fg, bg)
        self.draw_labels(cv, sky_rows)
        if self.wall:
            self.output(chars, fg, bg)
            self.render_cost = time.time() - t0
            return
        self.draw_marker(cv)
        self.draw_tooltip(cv, sky_rows)
        self.draw_status(cv, sky_rows)
        if self.help:
            self.draw_help(cv)
        self.output(chars, fg, bg)
        if full:
            self.render_cost = time.time() - t0

    def output(self, chars, fg, bg):
        out = self.screen.render(chars, fg, bg, getattr(self.term, "default_bg", None))
        if out:
            self.term.write(out)

    def object_altaz(self, key):
        if key.startswith(("star:", "con:")):
            if key.startswith("star:"):
                v = self.cat.vectors(self.snap.jd)[int(key[5:])] @ self.snap.hor.T
            else:
                v = self.cons.label_vec[self.cons.codes.index(key[4:])] @ self.snap.hor.T
            if self.view.atmosphere:
                v = astro.refract(v[None, :])[0]
        else:
            b = next((b for b in self.snap.bodies if b.key == key), None)
            if b is None:
                return None
            v = b.vec
        alt, az = astro.vec_to_altaz(v)
        return math.degrees(az), math.degrees(alt)

    def object_screen(self, key):
        f = self.frame
        if key.startswith("star:"):
            pos = self.object_altaz(key)
            if pos is None:
                return None
            az, alt = math.radians(pos[0]), math.radians(pos[1])
            v = np.array([[math.cos(alt) * math.sin(az), math.cos(alt) * math.cos(az), math.sin(alt)]])
            bx, by, zc = f.cam.project(v)
            return (float(bx[0]), float(by[0]), 0.0) if zc[0] > -0.9 else None
        for b, bx, by, r, mag, vis in f.bodies:
            if b.key == key:
                return bx, by, r
        return None

    def draw_marker(self, cv):
        if not self.view.selected or self.frame is None:
            return
        pos = self.object_screen(self.view.selected)
        if pos is None:
            return
        bx, by, r = pos
        cx, cy = int(bx // 2), int(by // 4)
        half = int(r // 2) + 1
        if cy >= cv.rows - self.status_rows:
            return
        cv.put(cy, cx - half - 1, "[", ACCENT)
        cv.put(cy, cx + half + 1, "]", ACCENT)

    def pick(self, col, row):
        f = self.frame
        if f is None:
            return None
        px, py = col * 2 + 1.0, row * 4 + 2.0
        best, best_score = None, 1e9
        for b, bx, by, r, mag, vis in f.bodies:
            if not vis:
                continue
            d = math.hypot(bx - px, (by - py) * 0.8)
            if d <= max(r, 0) + 6:
                score = d - max(r, 0) - 4 - (2 if b.kind != "comet" else 1)
                if score < best_score:
                    best, best_score = b.key, score
        if f.star_x.size:
            d = np.hypot(f.star_x - px, (f.star_y - py) * 0.8)
            score = d - 0.8 * np.clip(f.lim - f.star_mag, 0, 6)
            i = int(np.argmin(score))
            if d[i] <= 6 and score[i] < best_score:
                best = f"star:{int(f.star_idx[i])}"
        return best

    def star_names(self, i):
        c = self.cat
        proper = str(c.proper[i])
        bayer = str(c.bayer[i])
        con = str(c.con[i])
        desig = ""
        if bayer:
            g, _, num = bayer.partition("-")
            sup = {"1": "¹", "2": "²", "3": "³", "4": "⁴"}.get(num, num)
            desig = names.GREEK.get(g, g) + sup + " " + con
        elif str(c.flam[i]):
            desig = f"{c.flam[i]} {con}"
        return proper, desig

    def label_limit(self):
        return 3.0 + 3.5 * math.log10(100.0 / self.view.fov)

    def label_items(self, f, sky_rows, cols):
        items = []
        for b, bx, by, r, mag, vis in f.bodies:
            if not vis:
                continue
            if b.kind in ("sun", "moon"):
                items.append((-100.0, b.key, bx, by, r, b.name, LABEL_LUM))
            elif b.kind == "planet":
                items.append((mag - 50.0, b.key, bx, by, r, b.name, LABEL_PLANET))
            else:
                items.append((mag - 30.0, b.key, bx, by, r, b.name, LABEL_COMET))
        if f.star_x.size:
            sel = np.nonzero(f.star_mag <= self.label_limit())[0]
            sel = sel[np.argsort(f.star_mag[sel], kind="stable")][:LABEL_MAX]
            for k in sel:
                i = int(f.star_idx[k])
                proper, desig = self.star_names(i)
                text = names.STARS_RU.get(proper) or desig or proper
                if text:
                    items.append((float(f.star_mag[k]), f"star:{i}", float(f.star_x[k]), float(f.star_y[k]), 0.0,
                                  text, LABEL_STAR))
        out = []
        for pri, key, bx, by, r, text, col in sorted(items, key=lambda t: t[0]):
            cx, cy = int(bx // 2), int(by // 4)
            if not (0 <= cx < cols and 0 <= cy < sky_rows):
                continue
            ch = int(f.chars[cy, cx])
            if not (0x2800 < ch <= 0x28FF or ch == 0x2588):
                continue
            half = int(r // 2) + (3 if key == self.view.selected else 1)
            out.append((cx, cy, half, int(r // 4) + 1, text, col))
        return out

    def draw_labels(self, cv, sky_rows):
        f = self.frame
        if f is None or not (self.view.labels or self.view.constellations):
            return
        cols = cv.cols
        items = self.label_items(f, sky_rows, cols) if self.view.labels else []
        taken = np.zeros((sky_rows, cols), bool)
        for cx, cy, _, _, _, _ in items:
            taken[cy, cx] = True
        chars = cv.chars
        dotted = ((chars > 0x2800) & (chars <= 0x28FF)) | (chars == 0x2588)
        for cx, cy, half, vr, text, col in items:
            n = len(text)
            cands = [(cy, cx + half + 1), (cy, cx - half - n), (cy - vr, cx - n // 2), (cy + vr, cx - n // 2),
                     (cy - vr, cx + 1), (cy + vr, cx + 1), (cy - vr, cx - n), (cy + vr, cx - n)]
            self.place_label(cv, cands, text, col, taken, dotted, sky_rows)
        if self.view.constellations:
            for code, bx, by in f.con_labels:
                text = names.CONSTELLATIONS.get(code, code)
                cx, cy, n = int(bx // 2), int(by // 4), len(text)
                cands = [(cy, cx - n // 2), (cy - 1, cx - n // 2), (cy + 1, cx - n // 2)]
                self.place_label(cv, cands, text, LABEL_CON, taken, dotted, sky_rows)

    def place_label(self, cv, cands, text, col, taken, dotted, sky_rows):
        n = len(text)
        best, best_cost = None, None
        for k, (y, x) in enumerate(cands):
            if not (0 <= y < sky_rows and 0 <= x and x + n <= cv.cols):
                continue
            if taken[y, x:x + n].any():
                continue
            cost = int(dotted[y, x:x + n].sum()) + 0.3 * k
            if best_cost is None or cost < best_cost:
                best, best_cost = (y, x), cost
        if best is None:
            return
        y, x = best
        lo, hi = max(x - 1, 0), min(x + n + 1, cv.cols)
        taken[y, lo:hi] = True
        half_block = cv.chars[y, x:x + n] == 0x2580
        mixed = (cv.fg[y, x:x + n].astype(np.int32) + cv.bg[y, x:x + n]) // 2
        cv.bg[y, x:x + n][half_block] = mixed[half_block]
        cv.put(y, x, text, col)

    def describe(self, key):
        snap = self.snap
        pos = self.object_altaz(key)
        where = f"Азимут {pos[0]:.1f}°, высота {pos[1]:.1f}°" if pos else ""
        if key.startswith("star:"):
            i = int(key[5:])
            c = self.cat
            con = str(c.con[i])
            proper, desig = self.star_names(i)
            title = names.STARS_RU.get(proper, proper) or desig or (f"HIP {c.hip[i]}" if str(c.hip[i]) else "Звезда")
            if desig and title == desig:
                desig = ""
            line1 = title + (f" · {desig}" if desig else "")
            parts = []
            if con in names.CONSTELLATIONS:
                parts.append(names.CONSTELLATIONS[con])
            parts.append(f"блеск {float(c.mag[i]):.2f}")
            if float(c.dist[i]) > 0:
                ly = float(c.dist[i]) * 3.26156
                parts.append(f"{ly:,.0f} св. лет".replace(",", " ") if ly >= 10 else f"{ly:.1f} св. лет")
            return [line1, " · ".join(parts), where]
        b = next((b for b in snap.bodies if b.key == key), None)
        if b is None:
            return None
        ecl = ""
        if snap.total:
            ecl = "Полное солнечное затмение!"
        elif snap.eclipse > 0.0005:
            ecl = f"Солнечное затмение: диск Солнца закрыт на {snap.eclipse * 100:.0f}%"
        if b.kind == "sun":
            return ["Солнце", ecl, f"{b.dist_au:.3f} а.е.", where]
        if b.kind == "moon":
            ecl = self.lunar_text(snap.lunar) or ecl
            phase = "растущая" if b.extra.get("waxing") else "убывающая"
            km = b.dist_au * astro.AU_KM
            return ["Луна", ecl, f"Освещена на {b.phase * 100:.0f}%, {phase} · {km:,.0f} км".replace(",", " "), where]
        if b.kind == "planet":
            return [b.name, f"блеск {b.mag:.1f} · {b.dist_au:.2f} а.е. от Земли", where]
        return [b.name, f"{b.extra['des']} · блеск {b.mag:.1f}",
                f"{b.extra['r']:.2f} а.е. от Солнца · {b.dist_au:.2f} а.е. от Земли", where]

    @staticmethod
    def lunar_text(sh):
        if sh is None or sh.penumbral <= 0:
            return ""
        if sh.umbral >= 1:
            return f"Полное лунное затмение, фаза {sh.umbral:.2f}"
        if sh.umbral > 0:
            return f"Частное лунное затмение, фаза {sh.umbral:.2f}"
        return f"Полутеневое лунное затмение, фаза {sh.penumbral:.2f}"

    def draw_tooltip(self, cv, sky_rows):
        if self.mouse is None or self.drag or self.prompt or self.help:
            return
        col, row = self.mouse
        if row >= sky_rows:
            return
        key = self.pick(col, row)
        if not key:
            return
        lines = [l for l in (self.describe(key) or []) if l]
        if not lines:
            return
        width = max(len(l) for l in lines) + 2
        x = col + 2
        if x + width > cv.cols:
            x = max(0, col - width - 1)
        y = row + 1
        if y + len(lines) > sky_rows:
            y = max(0, row - len(lines))
        for k, line in enumerate(lines):
            cv.fill(y + k, x, width, TIP_BG)
            cv.put(y + k, x + 1, line, ACCENT if k == 0 else TIP_FG, TIP_BG)

    def draw_status(self, cv, sky_rows):
        r1, r2 = sky_rows, sky_rows + 1
        cv.fill(r1, 0, cv.cols, BAR_BG)
        cv.fill(r2, 0, cv.cols, BAR_BG)
        jd = self.snap.jd
        local = astro.datetime_from_jd(jd).astimezone(LOCAL_TZ)
        off = local.utcoffset()
        hours = off.total_seconds() / 3600 if off else 0
        tz = f"UTC{hours:+.0f}" if hours == int(hours) else f"UTC{hours:+.1f}"
        if self.paused:
            spd = "‖ пауза"
        else:
            arrow = "▶" if self.speed > 0 else "◀"
            spd = f"{arrow} {SPEED_NAMES[abs(self.speed)]}"
        x = cv.put(r1, 1, self.place, ACCENT, BAR_BG)
        if self.place != fmt_coords(self.lat, self.lon):
            x = cv.put(r1, x + 1, fmt_coords(self.lat, self.lon), DIM, BAR_BG)
        x = cv.put(r1, x, "  │  ", DIM, BAR_BG)
        x = cv.put(r1, x, local.strftime("%d.%m.") + f"{local.year:04d} " + local.strftime("%H:%M:%S"), BAR_FG, BAR_BG)
        x = cv.put(r1, x + 1, tz, DIM, BAR_BG)
        x = cv.put(r1, x, "  │  ", DIM, BAR_BG)
        x = cv.put(r1, x, spd, ACCENT if (self.speed != 1 or self.paused) else BAR_FG, BAR_BG)
        x = cv.put(r1, x, "  │  ", DIM, BAR_BG)
        x = cv.put(r1, x, f"Аз {self.view.az % 360:.1f}°  h {self.view.alt:+.1f}°  поле {self.view.fov:.3g}°", BAR_FG, BAR_BG)
        if self.view.selected and self.follow:
            cv.put(r1, x + 2, "· слежение", ACCENT, BAR_BG)
        if self.prompt:
            label, text = self.prompt
            x = cv.put(r2, 1, label + ": ", ACCENT, BAR_BG)
            x = cv.put(r2, x, text, (255, 255, 255), BAR_BG)
            cv.put(r2, x, "█", ACCENT, BAR_BG)
            return
        if self.message:
            cv.put(r2, 1, self.message[0], ACCENT, BAR_BG)
            return
        x = cv.put(r2, 1, "Засветка ", BAR_FG, BAR_BG)
        start = x
        x = cv.put(r2, x, "◂", DIM, BAR_BG)
        for i in range(1, 10):
            t = (i - 1) / 8
            col = (int(60 + 195 * t), int(110 + 60 * t - 40 * t * t), int(220 - 170 * t))
            x = cv.put(r2, x, "●" if i == self.view.bortle else "━", col if i <= self.view.bortle else DIM, BAR_BG)
        x = cv.put(r2, x, "▸", DIM, BAR_BG)
        self.slider = (start, r2)
        x = cv.put(r2, x + 1, f"{self.view.bortle} · {render.BORTLE_NAMES[self.view.bortle - 1]}", BAR_FG, BAR_BG)
        x = cv.put(r2, x, f" · звёзды до {self.frame.lim:.1f} зв. вел.", DIM, BAR_BG)
        x = cv.put(r2, x, "  │  ", DIM, BAR_BG)
        on = lambda v: "вкл" if v else "выкл"
        x = cv.put(r2, x, f"g ландшафт: {on(self.view.landscape)}", BAR_FG, BAR_BG)
        x = cv.put(r2, x, "  │  ", DIM, BAR_BG)
        x = cv.put(r2, x, f"a атмосфера: {on(self.view.atmosphere)}", BAR_FG, BAR_BG)
        x = cv.put(r2, x, "  │  ", DIM, BAR_BG)
        cv.put(r2, x, "? справка", ACCENT, BAR_BG)

    def draw_help(self, cv):
        width = max(len(l) for l in HELP) + 4
        height = len(HELP) + 2
        x0 = max(0, (cv.cols - width) // 2)
        y0 = max(0, (cv.rows - self.status_rows - height) // 2)
        for k in range(height):
            cv.fill(y0 + k, x0, width, TIP_BG)
        for k, line in enumerate(HELP):
            cv.put(y0 + 1 + k, x0 + 2, line, ACCENT if k == 0 else TIP_FG, TIP_BG)

    def handle(self, ev):
        if self.wall:
            return
        if ev[0] == "mouse":
            self.on_mouse(*ev[1:])
            return
        kind, key = ev
        if self.prompt:
            self.on_prompt_key(kind, key)
            return
        if kind == "key":
            self.on_key(key)
        else:
            self.on_char(key)

    def pan(self, dx_deg, dy_deg):
        self.follow = False
        cosalt = max(math.cos(math.radians(self.view.alt)), 0.15)
        self.view.az = (self.view.az + dx_deg / cosalt) % 360.0
        self.view.alt = float(np.clip(self.view.alt + dy_deg, -89.9, 89.9))
        self.dirty = "full"

    def zoom(self, factor, at=None):
        old = self.frame.cam if self.frame else None
        self.view.fov = float(np.clip(self.view.fov * factor, 0.05, 200.0))
        if at and old is not None and not self.follow:
            px, py = at[0] * 2 + 1.0, at[1] * 4 + 2.0
            target = old.unproject(np.array(px), np.array(py))
            for _ in range(3):
                cam = render.Camera(self.view.az, self.view.alt, self.view.fov, old.pw, old.ph, self.view.shift)
                now = cam.unproject(np.array(px), np.array(py))
                ta, tz = astro.vec_to_altaz(target)
                na, nz = astro.vec_to_altaz(now)
                daz = ((math.degrees(tz - nz) + 180) % 360) - 180
                self.view.az = (self.view.az + daz) % 360.0
                self.view.alt = float(np.clip(self.view.alt + math.degrees(ta - na), -89.9, 89.9))
        self.dirty = "full"

    def on_mouse(self, b, col, row, press):
        self.mouse = (col, row)
        sky_rows = self.size[1] - self.status_rows
        if b & 64:
            if row < sky_rows:
                self.zoom(1 / 1.2 if (b & 1) == 0 else 1.2, (col, row))
            return
        button = b & 3
        motion = bool(b & 32)
        if button == 0 and press and not motion:
            if row >= sky_rows:
                self.click_status(col, row)
                return
            self.drag = {"start": (col, row), "last": (col, row), "moved": False}
            return
        if button == 0 and motion and self.drag:
            lx, ly = self.drag["last"]
            if (col, row) != (lx, ly):
                self.drag["moved"] = True
                k = math.degrees(1.0 / self.frame.cam.scale) if self.frame else 0.5
                self.pan(-(col - lx) * 2 * k, (row - ly) * 4 * k)
                self.drag["last"] = (col, row)
            return
        if not press and self.drag:
            if not self.drag["moved"]:
                key = self.pick(col, row)
                self.view.selected = key
                self.follow = False
            self.drag = None
            self.dirty = self.dirty or True
            return
        self.dirty = self.dirty or True

    def click_status(self, col, row):
        x0, r = self.slider
        if row != r:
            return
        i = col - x0
        if i == 0:
            self.set_bortle(self.view.bortle - 1)
        elif 1 <= i <= 9:
            self.set_bortle(i)
        elif i == 10:
            self.set_bortle(self.view.bortle + 1)

    def set_bortle(self, b):
        self.view.bortle = int(np.clip(b, 1, 9))
        self.save()
        self.dirty = "full"

    def on_key(self, key):
        step = self.view.fov * (0.3 if key.startswith("S-") else 0.08)
        k = key[2:] if key.startswith("S-") else key
        if k == "left":
            self.pan(-step, 0)
        elif k == "right":
            self.pan(step, 0)
        elif k == "up":
            self.pan(0, step)
        elif k == "down":
            self.pan(0, -step)
        elif k == "pgup":
            self.set_jd(self.jd() - 1)
        elif k == "pgdn":
            self.set_jd(self.jd() + 1)
        elif k == "esc":
            if self.help:
                self.help = False
            else:
                self.view.selected = None
                self.follow = False
            self.dirty = True
        elif k == "ctrl-c":
            self.running = False

    def on_char(self, c):
        v = self.view
        if c in "qй":
            self.running = False
        elif c in "+=":
            self.zoom(1 / 1.25)
        elif c in "-_":
            self.zoom(1.25)
        elif c in "zя":
            v.fov = 100.0
            self.dirty = "full"
        elif c in "gп":
            v.landscape = not v.landscape
            self.save()
            self.dirty = "full"
        elif c in "aф":
            v.atmosphere = not v.atmosphere
            self.save()
            self.dirty = "full"
        elif c in "lд":
            v.labels = not v.labels
            self.save()
            self.dirty = True
        elif c in "kл":
            if not len(self.cons):
                self.notify("Нет данных о созвездиях: .venv/bin/python fetch_data.py constellations")
                return
            v.constellations = not v.constellations
            self.save()
            self.dirty = "full"
        elif c in "[х":
            self.set_bortle(v.bortle - 1)
        elif c in "]ъ":
            self.set_bortle(v.bortle + 1)
        elif c in "123456789":
            self.set_bortle(int(c))
        elif c == " ":
            if self.paused:
                self.anchor_real = time.time()
                self.paused = False
            else:
                self.set_jd(self.jd())
                self.paused = True
            self.dirty = True
        elif c in ".ю":
            self.step_speed(1)
        elif c in ",б":
            self.step_speed(-1)
        elif c in ">Ю":
            self.set_jd(self.jd() + 1 / 24)
        elif c in "<Б":
            self.set_jd(self.jd() - 1 / 24)
        elif c in "nт":
            self.paused = False
            self.speed = 1
            self.set_jd(astro.jd_from_datetime(datetime.now(timezone.utc)))
        elif c in "tе":
            self.prompt = ["Дата и время (2026-10-12 21:30 или 21:30)", ""]
            self.dirty = True
        elif c in "cс":
            self.prompt = ["Координаты (55.75 37.62) или город", ""]
            self.dirty = True
        elif c == "/":
            self.prompt = ["Найти", ""]
            self.dirty = True
        elif c in "fа":
            if v.selected:
                self.follow = not self.follow
                self.dirty = "full"
        elif c in "?hр":
            self.help = not self.help
            self.dirty = True

    def on_prompt_key(self, kind, key):
        label, text = self.prompt
        if kind == "char":
            self.prompt[1] = text + key
        elif key == "backspace":
            self.prompt[1] = text[:-1]
        elif key in ("esc", "ctrl-c"):
            self.prompt = None
        elif key == "enter":
            self.prompt = None
            try:
                self.submit(label, text)
            except ValueError as ex:
                self.notify(f"Ошибка: {ex}")
        self.dirty = True

    def submit(self, label, text):
        if not text.strip():
            return
        if label.startswith("Дата"):
            now_local = astro.datetime_from_jd(self.jd()).astimezone(LOCAL_TZ)
            dt = parse_time(text, now_local)
            self.set_jd(astro.jd_from_datetime(dt))
            self.notify(f"Время: {dt.strftime('%d.%m.')}{dt.year:04d} {dt.strftime('%H:%M')}")
        elif label.startswith("Коорд"):
            self.lat, self.lon, self.place = parse_coords(text)
            self.save()
            self.notify(f"Место: {self.place} ({fmt_coords(self.lat, self.lon)})")
        else:
            key = self.search(text)
            if key is None:
                raise ValueError(f"не нашёл «{text}»")
            if key.startswith("con:"):
                self.show_constellation(key)
                return
            self.view.selected = key
            self.follow = True
            pos = self.object_altaz(key)
            if pos:
                self.view.az, self.view.alt = pos
                if pos[1] < -2:
                    self.notify(f"{self.describe(key)[0]} сейчас под горизонтом (g — убрать землю)")
            self.dirty = "full"

    def show_constellation(self, key):
        self.view.selected = None
        self.follow = False
        if not self.view.constellations:
            self.view.constellations = True
            self.save()
        self.view.az, self.view.alt = self.object_altaz(key)
        name = names.CONSTELLATIONS.get(key[4:], key[4:])
        if self.view.alt < -2:
            self.notify(f"{name} сейчас под горизонтом (g — убрать землю)")
        else:
            self.notify(f"Созвездие {name}")
        self.dirty = "full"

    def build_star_index(self):
        out = []
        for i, p in enumerate(self.cat.proper.tolist()):
            if p:
                out.append((i, p, p.lower(), names.STARS_RU.get(p, "").lower()))
        return out

    def search(self, text):
        q = text.strip().lower()
        if q.startswith("комета "):
            q = q[7:]
        cands = []
        for b in self.snap.bodies:
            words = [b.name.lower(), names.PLANETS_EN.get(b.name, "").lower(), b.key.lower(), str(b.extra.get("des", "")).lower()]
            for w in words:
                if not w:
                    continue
                w2 = w[7:] if w.startswith("комета ") else w
                if w == q or w2 == q:
                    return b.key
                if w2.startswith(q) or (len(q) >= 3 and q in w):
                    cands.append((0, len(w), b.key))
        if cands:
            return min(cands)[2]
        cons = [(code, names.CONSTELLATIONS.get(code, "").lower(), self.cons.latin[code].lower())
                for code in dict.fromkeys(self.cons.codes)]
        for code, ru, latin in cons:
            if q in (ru, latin, code.lower()):
                return f"con:{code}"
        for i, p, pl, ru in self.star_index:
            if pl == q or ru == q:
                return f"star:{i}"
            if ru.startswith(q) or pl.startswith(q):
                cands.append((1, i, f"star:{i}"))
        if cands:
            return min(cands)[2]
        for code, ru, latin in cons:
            if ru.startswith(q) or latin.startswith(q):
                return f"con:{code}"
        return None


def parse_args():
    ap = argparse.ArgumentParser(description="Звёздное небо в терминале")
    ap.add_argument("--wallpaper", action="store_true", help="режим живых обоев: без панели, без управления")
    ap.add_argument("--az", type=float, default=180.0, help="азимут центра взгляда, градусы")
    ap.add_argument("--alt", type=float, default=0.0, help="высота центра взгляда, градусы")
    ap.add_argument("--fov", type=float, default=100.0, help="поле зрения по горизонтали, градусы")
    ap.add_argument("--bortle", type=int, default=5, choices=range(1, 10), help="засветка по шкале Бортля")
    ap.add_argument("--place", help="город или координаты; по умолчанию место из config.json")
    ap.add_argument("--interval", type=float, default=10.0, help="секунд между кадрами")
    ap.add_argument("--horizon", type=float, default=1 / 3,
                    help="на какой доле высоты экрана от низа стоит центр взгляда (0.33 — горизонт на трети при --alt 0)")
    ap.add_argument("--no-labels", action="store_true", help="без подписей")
    ap.add_argument("--no-landscape", action="store_true", help="без ландшафта")
    ap.add_argument("--constellations", action="store_true", help="линии и названия созвездий")
    args = ap.parse_args()
    if args.wallpaper:
        args.fov = min(max(args.fov, 0.05), 200.0)
        args.interval = max(args.interval, 0.5)
        args.horizon = min(max(args.horizon, 0.05), 0.95)
        if args.place:
            try:
                parse_coords(args.place)
            except ValueError as e:
                ap.error(f"--place: {e}")
    return args


def main():
    args = parse_args()
    if not (astro.DATA / "stars.npz").exists():
        print("Нет каталога звёзд. Запусти сначала:  .venv/bin/python fetch_data.py")
        sys.exit(1)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("Нужен настоящий терминал: запусти ./sky в отдельном окне kitty.")
        sys.exit(1)
    wall = args if args.wallpaper else None
    with Terminal(mouse=not wall) as term:
        app = App(term, wall)
        signal.signal(signal.SIGWINCH, lambda *a: None)
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
        try:
            app.run()
        finally:
            app.save()


if __name__ == "__main__":
    main()
