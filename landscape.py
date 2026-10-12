import math
from dataclasses import dataclass

import numpy as np

EYE = 1.7
SKY, SOLID, WINDOW, FAR = 0, 1, 2, 3
LEAF_LIT, NEEDLE_LIT, BARK_LIT = 1, 2, 3

MEADOW = np.array([60, 90, 40], float)
WHEAT = np.array([190, 156, 82], float)
ROAD = np.array([140, 118, 90], float)
FAR_COL = np.array([92, 104, 128], float)
FOREST = np.array([30, 54, 30], float)
TRUNK = np.array([62, 46, 34], float)
HAY = np.array([188, 158, 82], float)
GLASS = np.array([44, 50, 62], float)
LAMP = np.array([255, 196, 110], float)
WHITE = np.array([228, 224, 214], float)
DARK = np.array([30, 30, 34], float)
LEAF_STOPS = [0.0, 0.45, 0.8, 1.05]
LEAF_COLS = np.array([[16, 34, 26], [40, 72, 34], [92, 124, 48], [146, 162, 72]], float)
NEEDLE_STOPS = [0.0, 0.5, 1.0]
NEEDLE_COLS = np.array([[12, 30, 26], [28, 60, 42], [70, 104, 62]], float)
MEADOW_DRY = np.array([82, 100, 44], float)
FIELD_EDGE = np.array([50, 76, 34], float)
VILLAGE = (248.0, 0.6, 0.8)
LIGHT = np.array([-0.45, 0.7, 0.55]) / np.linalg.norm([-0.45, 0.7, 0.55])


@dataclass
class Geo:
    """Пейзаж для текущего вида: цвета и дальность пикселей, нормали для освещения Солнцем."""
    base: np.ndarray
    depth: np.ndarray
    kind: np.ndarray
    wid: np.ndarray
    lid: np.ndarray
    ln: np.ndarray
    lg: np.ndarray
    la: np.ndarray
    lm: np.ndarray


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def _mix(h):
    h &= 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return (h ^ (h >> 16)) * (1.0 / 4294967296.0)


def vnoise(x, y, seed=0.0):
    xf, yf = np.floor(x), np.floor(y)
    u, v = x - xf, y - yf
    u, v = u * u * (3 - 2 * u), v * v * (3 - 2 * v)
    hx = xf.astype(np.int64) * 374761393
    hy = yf.astype(np.int64) * 668265263 + int(seed * 7919) * 2246822519
    a, b = _mix(hx + hy), _mix(hx + 374761393 + hy)
    c, d = _mix(hx + hy + 668265263), _mix(hx + 374761393 + hy + 668265263)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fbm(x, y, seed=0.0):
    return 0.6 * vnoise(x, y, seed) + 0.4 * vnoise(x * 2.7, y * 2.7, seed + 1.7)


def ramp(v, stops, cols):
    return np.stack([np.interp(v, stops, cols[:, k]) for k in range(3)], axis=-1)


def az_window(az, start, width, soft):
    rel = (az - start) % 360.0
    edge = np.minimum(rel, width - rel)
    return np.where(rel <= width, smoothstep(0.0, soft, edge), 0.0)


class Ridge:
    def __init__(self, rng, freqs, power):
        self.k = np.array(freqs, float)
        a = self.k ** -power
        self.a = a / a.sum()
        self.p = rng.uniform(0, 2 * math.pi, len(freqs))

    def smooth(self, azr):
        return 0.5 + 0.5 * (np.sin(np.multiply.outer(azr, self.k) + self.p) * self.a).sum(-1)

    def ridged(self, azr):
        return ((1 - np.abs(np.sin(np.multiply.outer(azr, self.k) + self.p))) * self.a).sum(-1)


class Scene:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.far = Ridge(rng, [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233], 0.6)
        rng.uniform(0, 2 * math.pi, 19)
        self.belt_p = rng.uniform(0, 2 * math.pi, 4)
        self.road_az = 245.0
        walls = [(205, 194, 172), (168, 118, 78), (150, 162, 178), (186, 160, 112), (196, 186, 150)]
        roofs = [(122, 52, 42), (72, 76, 82), (88, 110, 88), (112, 86, 60)]
        self.houses = []
        for k in range(9):
            w = rng.uniform(7, 11)
            nwin = 2 if w < 9 else 3
            xs = np.linspace(-w / 2 + 1.6, w / 2 - 1.6, nwin)
            self.houses.append({
                "az": 229.0 + k * 3.9 + rng.uniform(-1.2, 1.2),
                "D": rng.uniform(260, 520),
                "w": w,
                "wall": rng.uniform(2.8, 3.4),
                "roof": rng.uniform(2.0, 3.2),
                "wall_col": np.array(walls[rng.integers(len(walls))], float),
                "roof_col": np.array(roofs[rng.integers(len(roofs))], float),
                "win": [(x - 0.5, x + 0.5) for x in xs],
                "chimney": rng.uniform(-w / 3, w / 3),
                "bed": rng.uniform(21.5, 25.5),
                "wake": rng.uniform(5.5, 7.5),
            })
        self.houses.append({
            "az": 267.0, "D": 420.0, "w": 18.0, "wall": 5.0, "roof": 4.0,
            "wall_col": np.array([104, 74, 52], float), "roof_col": np.array([70, 70, 74], float),
            "win": [], "chimney": None, "bed": 0, "wake": 0,
        })
        self.church = {"az": 249.4, "D": 650.0}
        self.trees = [
            {"kind": "leaf", "az": 200.0, "D": 38.0, "tw": 0.75, "hc": 9.2, "r": 4.8},
            {"kind": "spruce", "az": 214.0, "D": 56.0, "tw": 0.32, "h": 8.5, "r": 2.3, "b": 0.5, "tiers": 7},
            {"kind": "leaf", "az": 236.0, "D": 330.0, "tw": 0.5, "hc": 8.0, "r": 3.8},
            {"kind": "spruce", "az": 243.5, "D": 460.0, "tw": 0.5, "h": 14.0, "r": 4.2, "b": 1.0, "tiers": 9},
            {"kind": "leaf", "az": 258.5, "D": 380.0, "tw": 0.5, "hc": 7.5, "r": 3.4},
            {"kind": "spruce", "az": 262.0, "D": 540.0, "tw": 0.5, "h": 16.0, "r": 4.6, "b": 1.0, "tiers": 10},
            {"kind": "leaf", "az": 273.0, "D": 360.0, "tw": 0.5, "hc": 8.5, "r": 4.0},
        ]
        for t in self.trees:
            t["p"] = rng.uniform(0, 2 * math.pi, 5)
        for t in self.trees:
            if t["kind"] == "leaf":
                self._clumps(t, rng)
        az0, k, spread = VILLAGE
        for o in self.houses + [self.church] + [t for t in self.trees if t["D"] > 100]:
            o["az"] = az0 + (o["az"] - az0) / spread
            o["D"] *= k
        self.road_az = az0 + (self.road_az - az0) / spread
        self.road_len = 330.0 * k
        self.stacks = [
            {"az": 14.0, "D": 110.0, "w": 5.0, "h": 3.6},
            {"az": 33.0, "D": 165.0, "w": 5.5, "h": 3.8},
            {"az": 352.0, "D": 240.0, "w": 5.0, "h": 3.4},
            {"az": 47.0, "D": 300.0, "w": 5.0, "h": 3.5},
        ]

    def geometry(self, az, alt, px_rad, cols=None):
        self._cols = cols
        n = az.shape[0]
        depth = np.full(n, np.inf)
        base = np.zeros((n, 3))
        kind = np.zeros(n, np.int8)
        wid = np.full(n, -1, np.int16)
        lid = np.zeros(n, np.int8)
        ln = np.zeros((n, 3))
        lg = np.zeros((n, 3))
        la = np.zeros(n)
        lm = np.ones(n)
        azr = np.radians(az)
        altr = np.radians(alt)

        def put(idx, mask, d, color, k=SOLID, house=-1, lit=None):
            sel = idx[mask]
            dd = d[mask] if isinstance(d, np.ndarray) else np.full(sel.shape, float(d))
            closer = dd < depth[sel]
            sel = sel[closer]
            depth[sel] = dd[closer]
            if isinstance(color, np.ndarray) and color.ndim == 2:
                base[sel] = color[mask][closer]
            else:
                base[sel] = color
            kind[sel] = k
            wid[sel] = house
            lid[sel] = 0 if lit is None else lit[0]
            if lit is not None:
                for dst, src in zip((ln, lg, la, lm), lit[1:]):
                    dst[sel] = src[mask][closer]

        g = np.nonzero(alt < 0)[0]
        if g.size:
            d = np.minimum(EYE / np.tan(-altr[g]), 1e6)
            sa, ca = np.abs(np.sin(azr[g])), np.abs(np.cos(azr[g]))
            fx = px_rad * (d * ca + d * d / EYE * sa)
            fy = px_rad * (d * sa + d * d / EYE * ca)
            put(g, np.ones(g.size, bool), d, self._ground(d * np.sin(azr[g]), d * np.cos(azr[g]), fx, fy,
                                                          azr[g], altr[g], px_rad))

        c = np.nonzero((alt < 7.5) & (alt > -0.5))[0]
        if c.size:
            h = az_window(az[c], 285.0, 165.0, 30.0) * (1.6 + 5.0 * self.far.ridged(azr[c]) ** 1.4) - 0.25
            shade = (0.8 + 0.35 * np.clip(alt[c] / np.maximum(h, 1e-3), 0, 1))[:, None]
            snow = (h - alt[c] < 0.5) & (h > 4.2)
            col = np.where(snow[:, None], np.array([200, 205, 215], float), FAR_COL * shade)
            put(c, alt[c] < h, 22000.0, col, FAR)

        c = np.nonzero((alt < 3.2) & (alt > -0.4) & (az > 92) & (az < 173))[0]
        if c.size:
            a = azr[c]
            p = self.belt_p
            bumps = (0.5 * np.abs(np.sin(168 * a + p[0])) + 0.3 * np.abs(np.sin(241 * a + p[1]))
                     + 0.2 * np.abs(np.sin(97 * a + p[2])))
            top = (8.0 + 6.0 * bumps + 2.0 * np.sin(11 * a + p[3])) * az_window(az[c], 95.0, 75.0, 2.5)
            h = np.degrees(np.arctan((top - EYE) / 330.0))
            col = FOREST * (0.75 + 0.35 * bumps)[:, None]
            put(c, (alt[c] < h) & (top > 0.5), 330.0, col)

        for s in self.stacks:
            self._stack(s, az, alt, altr, put)
        for i, hs in enumerate(self.houses):
            self._house(i, hs, az, alt, altr, put)
        self._church(az, alt, altr, px_rad, put)
        for t in self.trees:
            if t["kind"] == "spruce":
                self._spruce(t, az, alt, altr, px_rad, put)
            else:
                self._tree(t, az, alt, altr, px_rad, put)
        return Geo(base, depth, kind, wid, lid, ln, lg, la, lm)

    def _local(self, az, alt, altr, az_c, d, half_w, top, bottom=None):
        hw = math.degrees(math.atan(half_w / d)) * 1.05 + 0.01
        ta = math.degrees(math.atan((top - EYE) / d)) + 0.01
        ba = math.degrees(math.atan((-EYE if bottom is None else bottom - EYE) / d)) - 0.01
        c = np.nonzero((alt < ta) & (alt > ba))[0]
        dz = wrap180(az[c] - az_c)
        keep = np.abs(dz) < hw
        c, dz = c[keep], dz[keep]
        dx = d * np.tan(np.radians(dz))
        hy = EYE + d * np.tan(altr[c])
        return c, dx, hy

    def _stack(self, s, az, alt, altr, put):
        c, dx, hy = self._local(az, alt, altr, s["az"], s["D"], s["w"] / 2, s["h"])
        if not c.size:
            return
        u = 2 * dx / s["w"]
        inside = (np.abs(u) < 1) & (hy >= 0) & (hy < s["h"] * np.sqrt(np.clip(1 - u * u, 0, 1)))
        shade = (0.7 + 0.35 * hy / s["h"] - 0.1 * u)[:, None]
        put(c, inside, s["D"], HAY * shade)

    def _house(self, i, h, az, alt, altr, put):
        w, wall, roof, d = h["w"], h["wall"], h["roof"], h["D"]
        c, dx, hy = self._local(az, alt, altr, h["az"], d, w / 2 + 0.3, wall + roof + 0.8)
        if not c.size:
            return
        adx = np.abs(dx)
        body = (adx < w / 2) & (hy >= 0) & (hy < wall)
        rf = (adx < w / 2 + 0.3) & (hy >= wall) & (hy < wall + roof * (1 - adx / (w / 2 + 0.3)))
        col = np.where(rf[:, None], h["roof_col"], h["wall_col"])
        put(c, body | rf, d, col)
        if h["chimney"] is not None:
            cx = h["chimney"]
            line = wall + roof * (1 - adx / (w / 2 + 0.3))
            top = wall + roof * (1 - abs(cx) / (w / 2 + 0.3)) + 0.9
            ch = (np.abs(dx - cx) < 0.35) & (hy > line - 0.2) & (hy < top)
            put(c, ch, d - 0.1, np.array([132, 72, 58], float))
        for x0, x1 in h["win"]:
            wm = (dx > x0) & (dx < x1) & (hy > 1.0) & (hy < 2.0)
            put(c, wm, d - 0.1, GLASS, WINDOW, i)

    def _church(self, az, alt, altr, px_rad, put):
        d = self.church["D"]
        th = max(0.12, 0.45 * px_rad * d)
        c, dx, hy = self._local(az, alt, altr, self.church["az"], d, 14.0, 23.5)
        if not c.size:
            return
        x = dx + 5.0
        body = (x > -1) & (x < 13) & (hy >= 0) & (hy < 7.0)
        roof = (x > -1) & (x < 13.4) & (hy >= 7.0) & (hy < 7.0 + 2.6 * (1 - np.abs(x - 6.2) / 7.2))
        tower = (np.abs(x) < 2.2) & (hy >= 0) & (hy < 14.0)
        arch = (np.abs(x) < 0.8) & (hy > 10.6) & (hy < 12.8)
        spire = (hy >= 14.0) & (np.abs(x) < np.maximum(2.6 * (1 - (hy - 14.0) / 8.0), th))
        put(c, body | tower, d, WHITE)
        put(c, roof | (spire & (hy < 22.0)), d - 0.2, np.array([80, 96, 84], float))
        put(c, arch, d - 0.3, DARK)

    def _snap(self, c, dx, px):
        lim = 0.6 * px
        if not self._cols:
            off = dx[np.argmin(np.abs(dx))]
            return dx - off if abs(off) <= lim else dx
        row = c // self._cols
        order = np.lexsort((np.abs(dx), row))
        r = row[order]
        first = np.r_[True, r[1:] != r[:-1]]
        off = dx[order][first][np.cumsum(first) - 1]
        off = np.where(np.abs(off) <= lim, off, 0.0)
        out = np.empty_like(dx)
        out[order] = dx[order] - off
        return out

    def _clumps(self, t, rng):
        r = t["r"]
        n = int(rng.integers(8, 12))
        ang = rng.uniform(0, 2 * math.pi, n)
        rad = r * 0.62 * np.sqrt(rng.uniform(0.15, 1.0, n))
        cx = np.append(rad * np.cos(ang) * 1.05, 0.0)
        cy = np.append(rad * np.sin(ang) * 0.9 + r * 0.08, 0.0)
        cr = np.append(r * rng.uniform(0.36, 0.52, n), r * 0.62)
        cz = np.append(rng.uniform(-0.25, 0.35, n) * r, 0.0)
        t["clumps"] = (cx, cy, cr, cz)
        side = np.argsort(np.abs(cx[:-1]) - 0.3 * cy[:-1])[::-1][:3]
        t["branches"] = [(cx[k] * 0.8, cy[k] * 0.8) for k in side]

    def _lit(self, t, kind, n, a, g=None, m=None):
        """Нормали из системы дерева (вправо, вверх, к зрителю) в мировую (восток, север, верх)."""
        ra = math.radians(t["az"])
        frame = np.array([[math.cos(ra), -math.sin(ra), 0.0], [0.0, 0.0, 1.0], [-math.sin(ra), -math.cos(ra), 0.0]])

        def world(v):
            v = np.stack(v, axis=-1)
            return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9) @ frame

        zero = np.zeros((a.size, 3))
        return (kind, world(n), zero if g is None else world(g), a, np.ones(a.size) if m is None else m)

    def _tree(self, t, az, alt, altr, px_rad, put):
        d, r, hc, tw = t["D"], t["r"], t["hc"], t["tw"]
        c, dx, hy = self._local(az, alt, altr, t["az"], d, r * 1.5, hc + r * 1.3)
        if not c.size:
            return
        p = t["p"]
        px = px_rad * d
        dx = self._snap(c, dx, px)
        detail = float(smoothstep(2.5, 7.0, r / max(px, 1e-6)))
        leaf = 0.32
        h1 = (fbm(dx / leaf, hy / leaf, p[0]) - 0.5) * 1.6 + 0.5
        h2 = vnoise(dx / leaf, hy / leaf, p[1])

        top = hc - 0.1 * r
        rel = np.clip(hy / top, 0, 1)
        hw = tw / 2 * (1 - 0.4 * rel) * (1 + 0.9 * np.exp(-np.maximum(hy, 0) / 0.35))
        hw = np.maximum(hw, 0.45 * px)
        wood = (np.abs(dx) < hw) & (hy >= 0) & (hy < top)
        u = dx / hw
        for bx, by in t["branches"]:
            x0, y0, x1, y1 = 0.0, hc - 0.45 * r, bx, hc + by
            vx, vy = x1 - x0, y1 - y0
            s = np.clip(((dx - x0) * vx + (hy - y0) * vy) / (vx * vx + vy * vy), 0, 1)
            dist = np.hypot(dx - x0 - s * vx, hy - y0 - s * vy)
            bw = tw * 0.22 * (1 - 0.6 * s) + 0.02
            br = dist < bw
            wood |= br
            u = np.where(br & ~(np.abs(dx) < hw), 0.3, u)
        grain = 0.12 * detail * (fbm(dx / 0.1, hy / 0.8, p[2]) - 0.5) * 1.6
        bark = TRUNK * (0.62 + 0.42 * np.clip(0.5 - 0.5 * u, 0, 1) + grain)[:, None]
        u = np.clip(u, -1, 1)
        put(c, wood, d + 0.2, bark, lit=self._lit(t, BARK_LIT, (u, 0 * u, np.sqrt(1 - u * u)), 0.55 + grain))

        cx, cy, cr, cz = t["clumps"]
        near = np.hypot(dx, hy - hc) < r * 1.35
        c, dx, hy, h1, h2 = c[near], dx[near], hy[near], h1[near], h2[near]
        X = dx[:, None] - cx[None, :]
        Y = (hy - hc)[:, None] - cy[None, :]
        q = (X * X + Y * Y) / (cr * cr)[None, :]
        jag = 1 + 0.1 * detail * (h1 - 0.5)[:, None] + 0.06 * np.sin(np.arctan2(Y, X) * 7 + p[3])
        inside = q < jag * jag
        zs = np.where(inside, cz[None, :] + cr[None, :] * np.sqrt(np.clip(1 - q, 0, 1)), -np.inf)
        k = np.argmax(zs, axis=1)
        rows = np.arange(k.size)
        crown = np.isfinite(zs[rows, k])
        qk = q[rows, k]
        crown &= ~((qk > 0.78) & (h2 < 0.2 * detail))
        crown &= ~(h2 < 0.012 * detail)
        nx = X[rows, k] / cr[k]
        ny = Y[rows, k] / cr[k]
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        lam = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0, 1)
        glob = np.clip(0.5 + 0.5 * (-0.5 * dx / r + 0.85 * (hy - hc) / r), 0, 1)
        m = 0.72 + 0.28 * nz
        tex = detail * (0.2 * (h1 - 0.5) + 0.08 * np.sin(dx * 3.1 + p[4]) * np.sin(hy * 2.7))
        b = (0.12 + 0.55 * lam + 0.38 * glob) * m + tex
        gx, gy = dx / r, (hy - hc) / r
        gz = np.sqrt(np.clip(1 - gx * gx - gy * gy, 0, 1))
        lit = self._lit(t, LEAF_LIT, (nx, ny, nz), 0.24 * m + tex, (gx, gy, gz), m)
        put(c, crown, d, ramp(np.clip(b, 0, 1.05), LEAF_STOPS, LEAF_COLS), lit=lit)

    def _spruce(self, t, az, alt, altr, px_rad, put):
        d, H, R, b0, tw = t["D"], t["h"], t["r"], t["b"], t["tw"]
        c, dx, hy = self._local(az, alt, altr, t["az"], d, R * 1.2, H + 0.4)
        if not c.size:
            return
        p = t["p"]
        px = px_rad * d
        dx = self._snap(c, dx, px)
        detail = float(smoothstep(2.0, 6.0, R / max(px, 1e-6)))
        h1 = (fbm(dx / 0.2, hy / 0.2, p[0]) - 0.5) * 1.6 + 0.5
        thw = np.maximum(tw / 2 * (1 + 0.6 * np.exp(-np.maximum(hy, 0) / 0.3)), 0.45 * px)
        trunk = (np.abs(dx) < thw) & (hy >= 0) & (hy < H * 0.85)
        u = np.clip(dx / thw, -1, 1)
        put(c, trunk, d + 0.2, TRUNK * (0.7 + 0.3 * (dx < 0))[:, None],
            lit=self._lit(t, BARK_LIT, (u, 0 * u, np.sqrt(1 - u * u)), np.full(u.size, 0.55)))
        f = np.clip((H - hy) / (H - b0), 0, 1)
        n = t["tiers"]
        tier = f * n
        sfr = tier % 1.0
        grow = np.minimum(1.0, 0.35 + 0.65 * np.floor(tier + 1) / n)
        w = R * f ** 0.9 * (0.55 + 0.45 * sfr ** 1.6) * (0.85 + 0.15 * grow)
        w = w * detail + R * f * 0.8 * (1 - detail)
        w *= 1 + detail * (0.18 * (h1 - 0.5) + 0.06 * np.sin(hy * 9 + p[1]))
        w = np.maximum(w, np.maximum(0.12, 0.45 * px) * (hy < H))
        foliage = (np.abs(dx) < w) & (hy >= b0 - 0.3 * sfr) & (hy <= H)
        u = dx / np.maximum(w, 1e-6)
        rest = 0.25 * (1 - sfr) - 0.3 * sfr ** 3 + 0.12 * (1 - f) + detail * 0.22 * (h1 - 0.5)
        bright = 0.55 - 0.32 * u + rest
        uc = np.clip(u, -1, 1)
        lit = self._lit(t, NEEDLE_LIT, (0.9 * uc, np.full(uc.size, 0.4), 0.9 * np.sqrt(1 - uc * uc)), rest - 0.05)
        put(c, foliage, d, ramp(np.clip(bright, 0, 1.0), NEEDLE_STOPS, NEEDLE_COLS), lit=lit)

    def _oct(self, x, y, foot, wl, seed, sx=1.0, sy=1.0):
        w = smoothstep(1.2, 3.5, wl / foot)
        out = np.zeros(x.shape)
        m = w > 0.01
        if m.any():
            out[m] = (vnoise(x[m] / (wl * sx), y[m] / (wl * sy), seed) * 2 - 1) * w[m]
        return out

    def _ground(self, x, y, fx, fy, azr, altr, px_rad):
        """Земля плоскими зонами: луг, поле, пашня, дорога. Края сглажены по размеру пикселя на земле."""
        foot = np.maximum(np.maximum(fx, fy), 1e-4)
        edge = np.maximum(np.minimum(fx, fy) * 0.5, 1e-4)

        def inside(dist, f=edge):
            return smoothstep(-0.5, 0.5, dist / f)[:, None]

        patch = 0.5 + 0.35 * self._oct(x, y, foot, 45, 1.1) + 0.15 * self._oct(x, y, foot, 14, 2.3)
        k = np.clip(foot / 30.0, 0.02, 0.5)
        dry = smoothstep(0.52 - k, 0.52 + k, patch)[:, None]
        col = MEADOW * (1 - dry) + MEADOW_DRY * dry
        col = col * (0.82 + 0.18 * smoothstep(2.0, 40.0, np.hypot(x, y)))[:, None]

        for t in self.trees:
            if t["D"] > 120:
                continue
            ra = math.radians(t["az"])
            rad = t["r"] * (1.6 if t["kind"] == "leaf" else 2.0)
            dist = np.hypot(x - t["D"] * math.sin(ra), y - t["D"] * math.cos(ra))
            col = col * (1 - 0.22 * np.exp(-(dist / rad) ** 2))[:, None]

        k = 0.55 * (y - 6.0)
        area = np.minimum(y - 6.0, np.minimum(x + 140.0 + k, 260.0 + k - x) / math.hypot(1.0, 0.55))
        m = inside(area)
        col = col * (1 - m) + FIELD_EDGE * m
        m = inside(area - 1.5)
        col = col * (1 - m) + WHEAT * m

        ra = math.radians(self.road_az)
        along = x * math.sin(ra) + y * math.cos(ra)
        across = np.abs(x * math.cos(ra) - y * math.sin(ra))
        near = (along > -6) & (along < self.road_len + 2) & (across < 2.4 + foot)
        if near.any():
            a, al, f = across[near], along[near], edge[near]
            half = 1.7 + 0.15 * np.sin(al / 9.0 + 1.3) + 0.1 * np.sin(al / 3.7)
            road = inside(np.minimum(half - a, np.minimum(al + 4, self.road_len - al)), f)
            col[near] = col[near] * (1 - road) + ROAD * road
        return col

    def lit(self, geo, sun, sun_alt):
        """Базовые цвета кадра: листва, хвоя и кора освещены текущим Солнцем."""
        base = geo.base.copy()
        w = float(smoothstep(-6.0, 2.0, sun_alt))
        s = np.array([sun[0], sun[1], max(sun[2], 0.05)])
        light = w * s / np.linalg.norm(s) + (1 - w) * np.array([0.0, 0.0, 1.0])
        light /= np.linalg.norm(light)
        idx = np.nonzero(geo.lid)[0]
        if idx.size:
            k, a, m = geo.lid[idx], geo.la[idx], geo.lm[idx]
            nl, gl = geo.ln[idx] @ light, geo.lg[idx] @ light
            wrap = np.clip((nl + 0.4) / 1.4, 0, 1)
            glow = 0.16 * np.clip(-gl, 0, 1) * (1.6 - m)
            b = np.where(k == LEAF_LIT, a + m * (0.55 * wrap + 0.19 * gl) + glow,
                         np.where(k == NEEDLE_LIT, a + 0.7 * (0.5 + 0.5 * nl), a + 0.4 * np.clip(nl, 0, 1)))
            col = np.where((k == LEAF_LIT)[:, None], ramp(np.clip(b, 0, 1.05), LEAF_STOPS, LEAF_COLS),
                           np.where((k == NEEDLE_LIT)[:, None], ramp(np.clip(b, 0, 1.0), NEEDLE_STOPS, NEEDLE_COLS),
                                    TRUNK * b[:, None]))
            base[idx] = col
        return base

    def window_lights(self):
        out = []
        for i, h in enumerate(self.houses):
            for x0, x1 in h["win"]:
                xm = (x0 + x1) / 2
                az = h["az"] + math.degrees(math.atan(xm / h["D"]))
                alt = math.degrees(math.atan((1.5 - EYE) / h["D"]))
                out.append((az, alt, h["D"], i))
        return out

    def lights_on(self, solar_hour, dark):
        hh = solar_hour if solar_hour >= 12 else solar_hour + 24
        res = []
        for h in self.houses:
            on = dark and bool(h["win"]) and (hh < h["bed"] or (h["wake"] + 24 <= hh < h["wake"] + 25.3))
            res.append(on)
        return np.array(res)

    def shade(self, base, depth, kind, wid, amb, haze_col, lit):
        dist = np.where(kind == FAR, depth, np.minimum(depth, 1000.0))
        haze = (0.82 * (1 - np.exp(-(dist / 1500.0) ** 0.5)))[:, None]
        col = base * amb * (1 - haze) + haze_col * haze
        win = (kind == WINDOW) & (wid >= 0)
        if win.any():
            on = lit[np.clip(wid, 0, None)] & win
            col[on] = LAMP * (1 - 0.5 * haze[on])
        return col
