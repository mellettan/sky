import math
from dataclasses import dataclass, field

import numpy as np

import astro
from landscape import WINDOW, Scene, smoothstep

BORTLE_NELM = [7.6, 7.1, 6.6, 6.2, 5.8, 5.3, 4.8, 4.3, 3.8]
BORTLE_NAMES = [
    "идеально тёмное небо", "тёмное небо", "сельское небо", "сельско-пригородное", "пригород",
    "яркий пригород", "окраина города", "город", "центр города",
]
BORTLE_GLOW = [0.0, 0.03, 0.08, 0.16, 0.28, 0.42, 0.58, 0.78, 1.0]

NIGHT_ZENITH = np.array([5, 7, 17], float)
NIGHT_HORIZON = np.array([16, 19, 32], float)
GLOW_ZENITH = np.array([34, 32, 40], float)
GLOW_HORIZON = np.array([118, 86, 58], float)
TW_ZENITH = np.array([26, 46, 100], float)
TW_SUN = np.array([238, 118, 46], float)
TW_ANTI = np.array([84, 74, 118], float)
DAY_ZENITH = np.array([42, 96, 200], float)
DAY_HORIZON = np.array([168, 202, 236], float)
SUN_GLARE = np.array([255, 236, 200], float)
MW_COL = np.array([30, 32, 40], float)
ION_COL = np.array([130, 180, 255], float)
DUST_COL = np.array([235, 228, 205], float)
CARD_COL = np.array([214, 92, 70], float)
ECL_ZENITH = np.array([14, 20, 46], float)
ECL_HORIZON = np.array([196, 124, 70], float)
CORONA = np.array([236, 240, 255], float)

BRAILLE_BITS = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]], np.int32)
PLUS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
SQUARE = [(0, 0), (1, 0), (0, 1), (1, 1)]
DOT = [(0, 0)]
MARIA = [
    (32.8, -15.6, 10.5), (40.0, -25.0, 6.0), (28.0, 17.5, 8.5), (8.5, 31.4, 9.0), (2.0, 24.0, 5.0),
    (17.0, 59.1, 5.5), (-7.8, 51.3, 7.0), (-2.0, 46.0, 4.0), (-15.2, 35.5, 4.5), (-21.3, -16.6, 7.5),
    (-10.0, -22.0, 5.0), (7.5, -30.9, 7.0), (10.0, -58.0, 11.0), (25.0, -55.0, 11.0), (-3.0, -48.0, 9.0),
    (38.0, -45.0, 7.0), (-24.4, -38.6, 4.5), (13.3, 3.6, 3.5), (2.0, 1.0, 3.0), (56.0, -30.0, 4.5),
    (57.0, -10.0, 4.0), (57.0, 10.0, 4.0), (55.0, 30.0, 3.5),
]
CARDINALS = [("С", 0), ("СВ", 45), ("В", 90), ("ЮВ", 135), ("Ю", 180), ("ЮЗ", 225), ("З", 270), ("СЗ", 315)]


class Camera:
    def __init__(self, az0, alt0, fov, pw, ph, shift=0.0):
        a, h = math.radians(az0), math.radians(alt0)
        f = np.array([math.cos(h) * math.sin(a), math.cos(h) * math.cos(a), math.sin(h)])
        u = np.array([0.0, 0.0, 1.0]) - f * f[2]
        u /= np.linalg.norm(u)
        self.f, self.u, self.r = f, u, np.cross(f, u)
        self.pw, self.ph = pw, ph
        self.cx, self.cy = pw / 2.0, ph * (0.5 + shift)
        self.scale = (pw / 2.0) / (2.0 * math.tan(math.radians(fov) / 4.0))
        self.fov = fov

    def project(self, v):
        xc, yc, zc = v @ self.r, v @ self.u, v @ self.f
        k = 2.0 / np.maximum(1.0 + zc, 1e-6)
        return self.cx + k * xc * self.scale, self.cy - k * yc * self.scale, zc

    def unproject(self, bx, by):
        x = (np.asarray(bx, float) - self.cx) / self.scale
        y = -(np.asarray(by, float) - self.cy) / self.scale
        den = 4.0 + x * x + y * y
        xc, yc, zc = 4 * x / den, 4 * y / den, (4 - x * x - y * y) / den
        return xc[..., None] * self.r + yc[..., None] * self.u + zc[..., None] * self.f

    def px_deg(self):
        return math.degrees(1.0 / self.scale)


@dataclass
class View:
    az: float = 180.0
    alt: float = 0.0
    fov: float = 100.0
    landscape: bool = True
    atmosphere: bool = True
    labels: bool = True
    shift: float = 0.5 - 1 / 3
    bortle: int = 4
    selected: str | None = None


@dataclass
class Frame:
    chars: np.ndarray
    fg: np.ndarray
    bg: np.ndarray
    cam: Camera
    lim: float
    star_x: np.ndarray
    star_y: np.ndarray
    star_idx: np.ndarray
    star_mag: np.ndarray
    bodies: list = field(default_factory=list)


@dataclass
class SkyCtx:
    sun: np.ndarray
    sun_alt: float
    moon: np.ndarray
    moon_k: float
    atmosphere: bool
    bortle: int
    mw_vis: float
    hor: np.ndarray
    ecl: float = 1.0
    total: bool = False
    sun_r: float = 0.27
    moon_r: float = 0.27


def mw_intensity(v_eq):
    g = v_eq @ astro.EQ_TO_GAL.T
    l = np.degrees(np.arctan2(g[..., 1], g[..., 0]))
    b = np.degrees(np.arcsin(np.clip(g[..., 2], -1, 1)))
    lw = (l + 180.0) % 360.0 - 180.0
    width = 6.0 + 5.0 * np.exp(-(lw / 60.0) ** 2)
    core = np.exp(-(b / width) ** 2)
    along = 0.3 + 0.7 * np.exp(-(lw / 80.0) ** 2) + 0.25 * np.exp(-((lw - 78) / 14.0) ** 2)
    bulge = 0.9 * np.exp(-((lw / 14.0) ** 2 + (b / 9.0) ** 2))
    clump = 1 + 0.22 * np.sin(lw * 0.21 + 1.3) * np.sin(b * 0.5 + 0.7) + 0.18 * np.sin(lw * 0.13 + 2.1)
    rift = 1 - 0.6 * np.exp(-((b - 2.0) / 2.4) ** 2) * smoothstep(-30, -18, lw) * (1 - smoothstep(48, 62, lw))
    lmc = 0.55 * np.exp(-(((l - 280.5 + 180) % 360 - 180) ** 2 + (b + 32.9) ** 2) / 14.0)
    smc = 0.35 * np.exp(-(((l - 302.8 + 180) % 360 - 180) ** 2 + (b + 44.3) ** 2) / 5.0)
    return (core * along + bulge) * clump * rift + lmc + smc


def sky_colors(v, ctx):
    alt = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    v_eq = v @ ctx.hor
    mw = mw_intensity(v_eq)[..., None] * ctx.mw_vis
    if not ctx.atmosphere:
        return np.array([3, 3, 7], float) + MW_COL * mw * 1.3
    s = ctx.sun_alt
    hz = np.exp(-np.maximum(alt, 0) / 14.0)[..., None]
    below = np.clip(1 + np.minimum(alt, 0) / 25.0, 0.3, 1.0)[..., None]
    g = BORTLE_GLOW[ctx.bortle - 1]
    col = NIGHT_ZENITH + (NIGHT_HORIZON - NIGHT_ZENITH) * hz
    col = col + g * (GLOW_ZENITH + (GLOW_HORIZON - GLOW_ZENITH) * hz ** 0.8)
    col = col + MW_COL * mw * (1 - 0.65 * hz)
    tw = smoothstep(-18.0, -2.0, s) ** 1.6
    if tw > 0:
        sun_h = ctx.sun[:2] / max(np.linalg.norm(ctx.sun[:2]), 1e-9)
        vh = v[..., :2] / np.maximum(np.linalg.norm(v[..., :2], axis=-1, keepdims=True), 1e-9)
        sf = ((1 + vh @ sun_h) / 2)[..., None] ** 2
        horizon = TW_ANTI + (TW_SUN - TW_ANTI) * sf
        col = col + tw * (TW_ZENITH + (horizon - TW_ZENITH) * hz ** 0.8)
    day = smoothstep(-4.0, 10.0, s)
    if day > 0:
        col = col + (DAY_ZENITH + (DAY_HORIZON - DAY_ZENITH) * hz - col) * day
    ang_sun = np.degrees(np.arccos(np.clip(v @ ctx.sun, -1, 1)))[..., None]
    glare = smoothstep(-8.0, 2.0, s)
    if glare > 0:
        col = col + SUN_GLARE * (0.5 * np.exp(-ang_sun / 3.0) + 0.25 * day * np.exp(-ang_sun / 25.0)) * glare
    if ctx.moon_k > 0:
        ang_moon = np.degrees(np.arccos(np.clip(v @ ctx.moon, -1, 1)))[..., None]
        col = col + ctx.moon_k * (1 - day) * (np.array([55, 60, 72]) * np.exp(-ang_moon / 7.0)
                                               + np.array([14, 18, 30]) * (0.5 + 0.5 * hz))
    if ctx.ecl < 1.0:
        dark = ECL_ZENITH + (ECL_HORIZON - ECL_ZENITH) * hz ** 1.5
        col = dark + (col - dark) * ctx.ecl
    if ctx.total:
        e1 = np.cross(ctx.sun, [0.0, 0.0, 1.0])
        e1 /= max(np.linalg.norm(e1), 1e-9)
        e2 = np.cross(ctx.sun, e1)
        pa = np.arctan2(v @ e2, v @ e1)[..., None]
        ang_moon = np.degrees(np.arccos(np.clip(v @ ctx.moon, -1, 1)))[..., None]
        x = np.maximum(ang_sun - ctx.moon_r, 0) / max(ctx.sun_r, 1e-4)
        rays = 0.6 + 0.4 * np.cos(2 * pa) ** 2 + 0.15 * np.cos(7 * pa + 1.3)
        corona = np.exp(-x / (0.35 * rays)) * (ang_moon > ctx.moon_r)
        col = col + CORONA * corona
    return np.clip(col * below, 0, 255)


def sizes_for(mag):
    return np.where(mag < -0.5, 2, np.where(mag < 1.2, 1, 0))


class Renderer:
    def __init__(self, catalog):
        self.cat = catalog
        self.scene = Scene()
        self._geo_key = None
        self._geo = None

    def eclipse_light(self, view, snap, sun_alt):
        if not view.atmosphere or snap.eclipse <= 0:
            return 1.0
        k = (1.0 - snap.eclipse) ** 0.35
        return 1.0 - (1.0 - k) * float(smoothstep(-4.0, 2.0, sun_alt))

    def limits(self, view, snap):
        nelm = BORTLE_NELM[view.bortle - 1]
        sun_alt = math.degrees(math.asin(snap.sun.vec[2]))
        moon_alt = math.degrees(math.asin(snap.moon.vec[2]))
        moon_k = snap.moon.phase ** 2 * float(smoothstep(-1.0, 12.0, moon_alt))
        if view.atmosphere:
            lim_sun = float(np.interp(sun_alt, [-18, -15, -12, -9, -6, -3, 0, 10],
                                      [9, 6.5, 4.5, 3.0, 1.8, 0.0, -2.5, -4.5]))
            lim = min(nelm, lim_sun) - 1.3 * moon_k
            k = self.eclipse_light(view, snap, sun_alt)
            lim = max(lim, lim * k + 1.8 * (1 - k))
        else:
            lim = nelm
            moon_k = 0.0
        lim += 2.2 * math.log10(max(1.0, 60.0 / view.fov))
        return min(lim, 11.0), sun_alt, moon_k

    def render(self, view, snap, cols, rows, solar_hour):
        cam = Camera(view.az, view.alt, view.fov, cols * 2, rows * 4, view.shift)
        lim, sun_alt, moon_k = self.limits(view, snap)
        mw_vis = float(np.clip((lim - 4.9 - 2.2 * math.log10(max(1.0, 60.0 / view.fov))) / 2.4, 0, 1))
        ecl = self.eclipse_light(view, snap, sun_alt)
        ctx = SkyCtx(snap.sun.vec, sun_alt, snap.moon.vec, moon_k, view.atmosphere, view.bortle, mw_vis, snap.hor,
                     ecl, snap.total and view.atmosphere, math.degrees(snap.sun.radius_rad),
                     math.degrees(snap.moon.radius_rad))

        hy, hx = np.mgrid[0:rows * 2, 0:cols]
        v = cam.unproject((hx + 0.5) * 2.0, (hy + 0.5) * 2.0)
        flat = v.reshape(-1, 3)
        alt = np.degrees(np.arcsin(np.clip(flat[:, 2], -1, 1)))
        az = np.degrees(np.arctan2(flat[:, 0], flat[:, 1])) % 360.0
        half = sky_colors(flat, ctx)
        land = np.zeros(flat.shape[0], bool)
        if view.landscape:
            q = cam.px_deg() * 0.5
            key = (round(view.az / q), round(view.alt / q), round(view.fov, 5), cols, rows, view.shift)
            if key != self._geo_key:
                self._geo = self.scene.geometry(az, alt, math.radians(cam.px_deg()) * 2, cols)
                self._geo_key = key
            geo = self._geo
            depth, kind, wid = geo.depth, geo.kind, geo.wid
            base = self.scene.lit(geo, snap.sun.vec, sun_alt)
            land = np.isfinite(depth)
            if land.any():
                hv = np.stack([np.sin(np.radians(az[land])), np.cos(np.radians(az[land])),
                               np.full(land.sum(), 0.005)], axis=-1)
                haze = sky_colors(hv, ctx)
                amb = self.ambient(sun_alt, moon_k, view.bortle) * (0.06 + 0.94 * ecl)
                lit = self.scene.lights_on(solar_hour, sun_alt < -4)
                half[land] = self.scene.shade(base[land], depth[land], kind[land], wid[land], amb, haze, lit)
                self._window_points(cam, half, depth, kind, wid, lit, cols, rows)
        else:
            line = np.abs(alt) < cam.px_deg() * 1.1
            half[line] = half[line] * 0.4 + np.array([120, 70, 50]) * 0.6
        half = half.reshape(rows * 2, cols, 3)
        land = land.reshape(rows * 2, cols)

        dots = np.zeros((rows * 4, cols * 2), bool)
        cell_col = np.zeros((rows, cols, 3))
        cell_pri = np.full((rows, cols), -1.0)
        self._solid = np.zeros((rows, cols), bool)

        sx, sy, sidx, smag = self._stars(view, snap, cam, lim, land, half, dots, cell_col, cell_pri, rows, cols)
        bodies = self._bodies(view, snap, cam, lim, land, half, dots, cell_col, cell_pri, rows, cols)

        top, bot = half[0::2], half[1::2]
        any_dot = dots.reshape(rows, 4, cols, 2).any(axis=(1, 3))
        sky_cell = ~(land[0::2] | land[1::2])
        braille_cell = any_dot & sky_cell
        bits = (dots.reshape(rows, 4, cols, 2) * BRAILLE_BITS[None, :, None, :]).sum(axis=(1, 3))
        chars = np.where(braille_cell, np.where((bits == 0xFF) & self._solid, 0x2588, 0x2800 + bits), 0x2580).astype(np.int32)
        fg = np.where(braille_cell[..., None], cell_col, top)
        bg = np.where(braille_cell[..., None], (top + bot) / 2, bot)
        fg = np.clip(fg, 0, 255).astype(np.uint8)
        bg = np.clip(bg, 0, 255).astype(np.uint8)
        frame = Frame(chars, fg, bg, cam, lim, sx, sy, sidx, smag, bodies)
        self._cardinals(frame, view, rows, cols)
        return frame

    def ambient(self, sun_alt, moon_k, bortle):
        level = 0.02 + 0.98 * float(smoothstep(-10.0, 8.0, sun_alt)) ** 1.2 + 0.05 * moon_k + 0.004 * bortle
        t = float(smoothstep(-6.0, 10.0, sun_alt))
        tint = np.array([0.7, 0.82, 1.1]) * (1 - t) + np.array([1.0, 1.0, 1.0]) * t
        tint = tint + np.array([0.18, 0.02, -0.18]) * math.exp(-((sun_alt - 2.0) / 5.0) ** 2)
        return level * tint

    def _window_points(self, cam, half, depth, kind, wid, lit, cols, rows):
        if not lit.any():
            return
        pts = self.scene.window_lights()
        az = np.radians([p[0] for p in pts])
        alt = np.radians([p[1] for p in pts])
        v = np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], axis=-1)
        bx, by, zc = cam.project(v)
        hx = np.floor(bx / 2).astype(int)
        hy = np.floor(by / 2).astype(int)
        for k, (_, _, d, house) in enumerate(pts):
            if not lit[house] or zc[k] < 0 or not (0 <= hx[k] < cols and 0 <= hy[k] < rows * 2):
                continue
            i = hy[k] * cols + hx[k]
            if depth[i] >= d - 0.5:
                half[i] = np.array([255, 196, 110]) * (0.75 if cam.px_deg() > 0.05 else 1.0)

    def _plot(self, px, py, col, pri, land, dots, cell_col, cell_pri, rows, cols):
        ok = (px >= 0) & (px < cols * 2) & (py >= 0) & (py < rows * 4)
        px, py, col, pri = px[ok], py[ok], col[ok], pri[ok]
        ok = ~land[py // 2, px // 2]
        px, py, col, pri = px[ok], py[ok], col[ok], pri[ok]
        if not px.size:
            return
        dots[py, px] = True
        cy, cx = py // 4, px // 2
        order = np.argsort(pri, kind="stable")
        cy, cx, col, pri = cy[order], cx[order], col[order], pri[order]
        better = pri >= cell_pri[cy, cx]
        cell_col[cy[better], cx[better]] = col[better]
        cell_pri[cy[better], cx[better]] = pri[better]

    def _stars(self, view, snap, cam, lim, land, half, dots, cell_col, cell_pri, rows, cols):
        n = self.cat.count_brighter(lim)
        empty = (np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0))
        if n == 0:
            return empty
        vec = self.cat.vectors(snap.jd)[:n] @ snap.hor.T
        if view.atmosphere:
            near = vec[:, 2] < 0.2
            vec[near] = astro.refract(vec[near])
        mag = self.cat.mag[:n].astype(float)
        alt = np.arcsin(np.clip(vec[:, 2], -1, 1))
        if view.atmosphere:
            alt_e = alt if view.landscape else np.abs(alt)
            mag = mag + np.where(alt_e > 0, 0.22 * (astro.airmass(alt_e) - 1), 8.6)
        if view.landscape:
            vis = alt > -0.02
        else:
            vis = np.ones(n, bool)
        vis &= mag <= lim
        bx, by, zc = cam.project(vec)
        vis &= (zc > -0.95) & (bx >= -2) & (bx < cols * 2 + 2) & (by >= -2) & (by < rows * 4 + 2)
        idx = np.nonzero(vis)[0]
        if not idx.size:
            return empty
        bx, by, m = bx[idx], by[idx], mag[idx]
        ix = np.floor(bx).astype(int)
        iy = np.floor(by).astype(int)
        inten = np.clip(0.3 + 0.17 * (lim - m), 0.3, 1.0)
        hcx = np.clip(ix // 2, 0, cols - 1)
        hcy = np.clip(iy // 2, 0, rows * 2 - 1)
        bgc = half[hcy, hcx]
        col = bgc + (self.cat.rgb[idx].astype(float) - bgc) * inten[:, None]
        size = sizes_for(m)
        for s, shape in ((0, DOT), (1, SQUARE), (2, PLUS)):
            sel = size == s
            if not sel.any():
                continue
            for ox, oy in shape:
                self._plot(ix[sel] + ox, iy[sel] + oy, col[sel], inten[sel] - 0.05 * (ox != 0 or oy != 0),
                           land, dots, cell_col, cell_pri, rows, cols)
        return bx, by, idx, m

    def _bodies(self, view, snap, cam, lim, land, half, dots, cell_col, cell_pri, rows, cols):
        out = []
        for b in sorted(snap.bodies, key=lambda b: (b.kind == "moon", -b.mag)):
            v = b.vec
            alt = math.asin(max(-1.0, min(1.0, v[2])))
            if view.landscape and alt < math.radians(-2.0) and b.kind != "comet":
                continue
            mag = b.mag
            if view.atmosphere and alt > 0:
                mag += 0.22 * (float(astro.airmass(np.array(alt))) - 1)
            bx, by, zc = cam.project(v[None, :])
            bx, by, zc = float(bx[0]), float(by[0]), float(zc[0])
            if zc < -0.9:
                continue
            r_px = b.radius_rad * cam.scale
            visible = b.kind in ("sun", "moon") or mag <= lim + (0.5 if b.kind == "planet" else 0.0)
            if b.kind == "comet" and mag <= lim + 1.5:
                for end, kind in b.extra.get("tails", []):
                    self._tail(end, kind, bx, by, mag, lim, cam, land, half, dots, cell_col, cell_pri, rows, cols)
            margin = r_px * 2.4 + 4
            if not (-margin <= bx < cols * 2 + margin and -margin <= by < rows * 4 + margin):
                continue
            if visible:
                if b.kind in ("sun", "moon"):
                    r_px = max(r_px, 1.7)
                if r_px >= 1.2:
                    self._disk(b, snap, bx, by, r_px, cam, land, dots, cell_col, cell_pri, rows, cols)
                else:
                    self._point(b, bx, by, mag, lim, land, half, dots, cell_col, cell_pri, rows, cols)
            out.append((b, bx, by, max(r_px, 0.0), mag, visible))
        return out

    def _point(self, b, bx, by, mag, lim, land, half, dots, cell_col, cell_pri, rows, cols):
        inten = float(np.clip(0.35 + 0.17 * (lim - mag), 0.35, 1.0))
        shape = (DOT, SQUARE, PLUS)[int(sizes_for(np.array(mag)))]
        ix, iy = int(math.floor(bx)), int(math.floor(by))
        hcx = min(max(ix // 2, 0), cols - 1)
        hcy = min(max(iy // 2, 0), rows * 2 - 1)
        bgc = half[hcy, hcx]
        col = bgc + (np.array(b.color, float) - bgc) * inten
        xs = np.array([ix + ox for ox, _ in shape])
        ys = np.array([iy + oy for _, oy in shape])
        self._plot(xs, ys, np.repeat(col[None, :], len(shape), 0), np.full(len(shape), 5.0 + inten),
                   land, dots, cell_col, cell_pri, rows, cols)
        if b.kind == "comet":
            rad = float(np.clip((lim - mag) / 1.6, 0, 5.0))
            if rad >= 1:
                n = int(rad) + 1
                gy, gx = np.mgrid[-n:n + 1, -n:n + 1]
                dist = np.hypot(gx, gy) / rad
                cx, cy = ix + gx.ravel(), iy + gy.ravel()
                w = np.clip(1 - dist.ravel(), 0, 1)
                noise = (np.sin(cx * 12.9898 + cy * 78.233) * 43758.5453) % 1.0
                keep = (w > 0) & (noise < w * 1.2)
                cc = bgc + (np.array(b.color, float) - bgc) * np.clip(w[keep] + 0.3, 0, 1)[:, None]
                self._plot(cx[keep], cy[keep], cc, 4.0 + w[keep], land, dots, cell_col, cell_pri, rows, cols)

    def _disk(self, b, snap, bx, by, r, cam, land, dots, cell_col, cell_pri, rows, cols):
        ring = b.key == "Сатурн" and r >= 2.5
        ext = r * (2.35 if ring else 1.0)
        x0, x1 = max(int(math.floor(bx - ext - 1)), 0), min(int(math.ceil(bx + ext + 1)), cols * 2)
        y0, y1 = max(int(math.floor(by - ext - 1)), 0), min(int(math.ceil(by + ext + 1)), rows * 4)
        if x0 >= x1 or y0 >= y1:
            return
        gy, gx = np.mgrid[y0:y1, x0:x1]
        dx = (gx + 0.5 - bx) / r
        dy = (gy + 0.5 - by) / r
        rr = dx * dx + dy * dy
        f = b.vec / np.linalg.norm(b.vec)
        u = cam.u - f * np.dot(cam.u, f)
        u /= max(np.linalg.norm(u), 1e-9)
        rgt = np.cross(f, u)
        nz = np.sqrt(np.clip(1 - rr, 0, 1))
        normal = dx[..., None] * rgt - dy[..., None] * u - nz[..., None] * f
        disk = rr <= 1.0
        lit = disk.copy()
        if b.kind != "sun" and np.linalg.norm(b.sun_dir) > 0:
            lit &= normal @ b.sun_dir > 0
        col = np.broadcast_to(np.array(b.color, float), dx.shape + (3,)).copy()
        if b.kind == "sun":
            col *= (1.0 - np.array([0.1, 0.3, 0.6]) * rr[..., None])
        else:
            col *= (0.82 + 0.18 * nz)[..., None]
        pole = astro.POLES.get(b.key)
        if pole is not None and r >= 3:
            ph = snap.hor @ pole
            lat = np.degrees(np.arcsin(np.clip(normal @ ph, -1, 1)))
            if b.key == "moon":
                col = self._maria(col, normal, ph, f)
            elif b.key == "Юпитер":
                band = ((lat > 7) & (lat < 18)) | ((lat < -8) & (lat > -20))
                col[band] *= np.array([0.78, 0.66, 0.55])
                col[np.abs(lat) > 45] *= 0.85
            elif b.key == "Сатурн":
                col[(np.abs(lat) > 18) & (np.abs(lat) < 34)] *= np.array([0.88, 0.82, 0.72])
            elif b.key == "Марс":
                col[np.abs(lat) > 72] = (245, 240, 235)
        if b.kind == "moon":
            dark = disk & ~lit
            dots[gy[dark], gx[dark]] = False
        draw = lit
        if ring:
            ph = snap.hor @ astro.POLES["Сатурн"]
            a = ph - f * np.dot(ph, f)
            a = np.cross(ph, f)
            a /= max(np.linalg.norm(a), 1e-9)
            bvec = np.cross(ph, a)
            m = np.array([[a @ rgt, bvec @ rgt], [-(a @ u), -(bvec @ u)]])
            if abs(np.linalg.det(m)) > 1e-3:
                inv = np.linalg.inv(m)
                p = inv[0, 0] * dx + inv[0, 1] * dy
                q = inv[1, 0] * dx + inv[1, 1] * dy
                rad = np.hypot(p, q)
                in_ring = (rad > 1.24) & (rad < 2.27) & ~((rad > 1.95) & (rad < 2.02))
                toward = -(p * (a @ f) + q * (bvec @ f))
                front = in_ring & ((toward > 0) | ~disk)
                rc = np.array([226, 210, 168], float) * np.where(rad < 1.53, 0.7, 1.0)[..., None]
                col = np.where(front[..., None], rc, col)
                draw = lit | front
        px, py = gx[draw], gy[draw]
        if not px.size:
            return
        self._plot_avg(px, py, col[draw], 10.0, land, dots, cell_col, cell_pri, rows, cols)

    def _maria(self, col, normal, pole, f):
        e = -f
        x = e - pole * np.dot(e, pole)
        x /= max(np.linalg.norm(x), 1e-9)
        y = np.cross(pole, x)
        lat = np.arcsin(np.clip(normal @ pole, -1, 1))
        lon = np.arctan2(normal @ y, normal @ x)
        wobble = 1 + 0.18 * np.sin(7 * lon + 1.1) * np.sin(6 * lat + 0.4) + 0.1 * np.sin(13 * lon - 9 * lat)
        shade = np.ones(lat.shape)
        for mlat, mlon, mr in MARIA:
            d = np.arccos(np.clip(np.sin(lat) * math.sin(math.radians(mlat))
                                  + np.cos(lat) * math.cos(math.radians(mlat)) * np.cos(lon - math.radians(mlon)), -1, 1))
            d = np.degrees(d) * wobble
            shade = np.minimum(shade, 0.6 + 0.4 * smoothstep(mr * 0.7, mr * 1.15, d))
        return col * shade[..., None]

    def _plot_avg(self, px, py, col, pri, land, dots, cell_col, cell_pri, rows, cols):
        ok = ~land[py // 2, px // 2]
        px, py, col = px[ok], py[ok], col[ok]
        if not px.size:
            return
        dots[py, px] = True
        cid = (py // 4) * cols + px // 2
        uniq, inv = np.unique(cid, return_inverse=True)
        sums = np.zeros((uniq.size, 3))
        np.add.at(sums, inv, col)
        cnt = np.bincount(inv)
        cy, cx = uniq // cols, uniq % cols
        better = pri >= cell_pri[cy, cx]
        cell_col[cy[better], cx[better]] = sums[better] / cnt[better, None]
        cell_pri[cy[better], cx[better]] = pri
        self._solid[cy[better], cx[better]] = True

    def _tail(self, end, kind, bx, by, mag, lim, cam, land, half, dots, cell_col, cell_pri, rows, cols):
        tx, ty, tz = cam.project(end[None, :])
        tx, ty, tz = float(tx[0]), float(ty[0]), float(tz[0])
        if tz < -0.5:
            return
        dxl, dyl = tx - bx, ty - by
        length = math.hypot(dxl, dyl)
        if length < 2 or length > 6000:
            return
        ux, uy = dxl / length, dyl / length
        px_, py_ = -uy, ux
        bright = float(np.clip((lim - mag) / 4.0, 0.25, 1.0))
        spread = 0.05 if kind == "ion" else 0.16
        tint = ION_COL if kind == "ion" else DUST_COL
        gain = 0.9 if kind == "ion" else 1.25
        t = np.linspace(0.03, 1.0, int(length * 1.5) + 2)
        ks = np.arange(-8, 9) / 8.0
        w = 0.8 + t * length * spread
        xs = (bx + t * dxl)[None, :] + ks[:, None] * w[None, :] * px_
        ys = (by + t * dyl)[None, :] + ks[:, None] * w[None, :] * py_
        ws = ((1 - t) ** 1.3)[None, :] * bright * gain * (1 - np.abs(ks[:, None]) ** 2 * 0.8)
        ws = ws / np.maximum(0.4, w[None, :] ** 0.35)
        ix, iy, ws = np.floor(xs).ravel().astype(int), np.floor(ys).ravel().astype(int), ws.ravel()
        noise = (np.sin(ix * 12.9898 + iy * 78.233) * 43758.5453) % 1.0
        keep = (ws > 0.05) & (noise < ws * 1.6)
        ix, iy, ws = ix[keep], iy[keep], ws[keep]
        ok = (ix >= 0) & (ix < cols * 2) & (iy >= 0) & (iy < rows * 4)
        ix, iy, ws = ix[ok], iy[ok], ws[ok]
        if not ix.size:
            return
        bgc = half[iy // 2, ix // 2]
        col = bgc + (tint - bgc) * np.clip(ws + 0.25, 0, 1)[:, None]
        self._plot(ix, iy, col, ws * 0.25, land, dots, cell_col, cell_pri, rows, cols)

    def _cardinals(self, frame, view, rows, cols):
        cam = frame.cam
        lift = 0.0 if view.landscape else 1.0
        for label, az in CARDINALS:
            a, h = math.radians(az), math.radians(lift)
            v = np.array([[math.cos(h) * math.sin(a), math.cos(h) * math.cos(a), math.sin(h)]])
            bx, by, zc = cam.project(v)
            if zc[0] < 0:
                continue
            cx = int(bx[0] // 2) - len(label) // 2
            cy = int(math.ceil(by[0] / 4)) if view.landscape else int(by[0] // 4)
            if 0 <= cy < rows:
                for k, ch in enumerate(label):
                    if 0 <= cx + k < cols:
                        if frame.chars[cy, cx + k] == 0x2580:
                            mix = (frame.fg[cy, cx + k].astype(float) + frame.bg[cy, cx + k]) / 2
                            frame.bg[cy, cx + k] = mix.astype(np.uint8)
                        frame.chars[cy, cx + k] = ord(ch)
                        frame.fg[cy, cx + k] = CARD_COL
