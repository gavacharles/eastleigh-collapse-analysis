"""Shared 3-D scene builders (PyVista) for the rendered animations.

Local coordinates: metres east (x) / north (y) of the site, z up (UTM 37S)."""
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
from pyproj import Transformer

HERE = Path(__file__).resolve().parent
RAW = HERE / "gis" / "raw"
TILES = RAW / "esri"
TILES.mkdir(parents=True, exist_ok=True)
SITE_LL = tuple(json.load(open(RAW / "meta.json"))["site"])
TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32737", always_xy=True)
SX, SY = TO_UTM.transform(SITE_LL[1], SITE_LL[0])
UA = "building-collapse-forensics/1.0 (research)"
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def local(lon, lat):
    x, y = TO_UTM.transform(lon, lat)
    return np.asarray(x) - SX, np.asarray(y) - SY


def tile_xy(lat, lon, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def tile_lonlat(x, y, z):
    n = 2 ** z
    return x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))


def esri_mosaic(half_deg, z, url=ESRI, tag="esri"):
    """Satellite mosaic around the site -> (RGB image, lon_w, lat_n, lon_e, lat_s)."""
    lat, lon = SITE_LL
    x0, y0 = tile_xy(lat + half_deg, lon - half_deg, z)
    x1, y1 = tile_xy(lat - half_deg, lon + half_deg, z)
    xs, ys = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    rows = []
    for ty in ys:
        row = []
        for tx in xs:
            f = TILES / f"{tag}_{z}_{ty}_{tx}.jpg"
            if not f.exists():
                subprocess.run(["curl", "-sS", "-f", "-m", "60", "-A", UA, "-o", str(f),
                                url.format(z=z, y=ty, x=tx)], check=True)
            row.append(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
        rows.append(np.hstack(row))
    img = np.vstack(rows)
    lw, ln = tile_lonlat(xs[0], ys[0], z)
    le, ls = tile_lonlat(xs[-1] + 1, ys[-1] + 1, z)
    return img, (lw, ln, le, ls)


def textured_plane(img, bounds, z=0.0):
    lw, ln, le, ls = bounds
    (x0, x1), (y1, y0) = local(np.array([lw, le]), np.array([ln, ls]))[0], local(np.array([lw, le]), np.array([ln, ls]))[1]
    plane = pv.Plane(center=((x0 + x1) / 2, (y0 + y1) / 2, z), direction=(0, 0, 1),
                     i_size=x1 - x0, j_size=y1 - y0, i_resolution=1, j_resolution=1)
    plane.texture_map_to_plane(inplace=True)
    return plane, pv.Texture(np.ascontiguousarray(img))


def terrain(img, bounds, exag=3.0, step=3):
    """DEM surface draped with imagery (z relative to site elevation, exaggerated)."""
    d = np.load(RAW / "dem.npz")
    elev, (dw, dn, de, ds) = d["elev"][::step, ::step], d["bounds"]
    ny, nx = elev.shape
    lons = np.linspace(dw, de, nx)
    lats = np.linspace(dn, ds, ny)
    LON, LAT = np.meshgrid(lons, lats)
    X, Y = local(LON, LAT)
    z0 = float(np.median(elev))
    Z = (elev - z0) * exag
    grid = pv.StructuredGrid(X, Y, Z)
    lw, ln, le, ls = bounds
    u = (LON - lw) / (le - lw)
    v = (LAT - ls) / (ln - ls)
    grid.active_texture_coordinates = np.c_[u.ravel(order="F"), v.ravel(order="F")]
    return grid, pv.Texture(np.ascontiguousarray(img)), z0


def osm_buildings(radius=320.0, default_levels=4, exclude=None):
    gis = json.load(open(HERE / "gis" / "web_layers.json"))
    tagged, untagged = [], []
    for b in gis["buildings"]:
        pts = np.array(b["xy"])
        if len(pts) < 4 or np.hypot(*pts.mean(0)) > radius or np.hypot(*pts.mean(0)) < 14:
            continue
        if exclude is not None and exclude(pts.mean(0)):
            continue
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        h = (b["lv"] or default_levels) * 3.0
        poly = pv.PolyData(np.c_[pts, np.full(len(pts), 0.3)], faces=np.r_[len(pts), np.arange(len(pts))])
        try:
            m = poly.extrude((0, 0, h), capping=True).triangulate()
        except Exception:
            continue
        (tagged if b["lv"] else untagged).append(m)
    return (pv.merge(tagged) if tagged else None), (pv.merge(untagged) if untagged else None), gis


def ninth_axis(gis):
    best = None
    for r in gis["roads"]:
        if r.get("name") != "Ninth Street":
            continue
        for (x1, y1), (x2, y2) in zip(r["xy"][:-1], r["xy"][1:]):
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 0.1:
                continue
            d = abs((x2 - x1) * y1 - (y2 - y1) * x1) / L
            if best is None or d < best[0]:
                best = (d, math.atan2(y2 - y1, x2 - x1))
    return best[1] if best else 0.0


def roads_mesh(gis, width=None):
    W = dict(secondary=12, tertiary=9, unclassified=7, residential=6, service=4, primary=14, trunk=16)
    meshes = []
    for r in gis["roads"]:
        w = W.get((r["hw"] or "").replace("_link", ""), 5)
        xy = np.array(r["xy"])
        if len(xy) < 2:
            continue
        line = pv.lines_from_points(np.c_[xy, np.full(len(xy), 0.4)])
        meshes.append(line.tube(radius=w / 2, n_sides=4).scale((1, 1, 0.02), point=(0, 0, 0.4)))
    return pv.merge(meshes)


def ring(r, z=0.6, width=1.2):
    return pv.Disc(center=(0, 0, z), inner=r - width / 2, outer=r + width / 2, c_res=128)


def rot(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = np.asarray(pts, float)
    out = pts.copy()
    out[..., 0] = c * pts[..., 0] - s * pts[..., 1]
    out[..., 1] = s * pts[..., 0] + c * pts[..., 1]
    return out


# ======================================================================== collapse state engine
class Collapse:
    """Time-resolved state of the collapsing building, shared by all renders.

    Kinematics: crush-up model fitted to the video (out/animation_timeline.json).
    Column-failure order and DCRs: FE-twin scenario S6 (out/fe/fe_anim.json).
    Geometry in the building frame: x along the Ninth Street frontage (corner end at
    x = -W/2), y into the plot, z up; rotated by AXIS and centred on the site."""
    SCEN = "S6 as-built f'c 15, two corner-end columns lost"

    def __init__(self, axis):
        tl = json.load(open(HERE / "out" / "animation_timeline.json"))
        self.rows = np.array([[r[0], r[1], r[2], np.nan if r[3] is None else r[3], r[4]] for r in tl["rows"]])
        self.lam, self.T0 = tl["lam"], tl["t0"]
        fe = json.load(open(HERE / "out" / "fe" / "fe_anim.json"))
        g = fe["geo"]
        self.xs = np.array(g["xs"]) - g["xs"][-1] / 2
        self.ys = np.array(g["ys"]) - g["ys"][-1] / 2
        self.zs = np.array(g["zs"])
        self.W, self.D = self.xs[-1] - self.xs[0], self.ys[-1] - self.ys[0]
        self.N = len(self.zs) - 1
        self.axis = axis
        sc = fe["anim"][self.SCEN]
        steps = sc["steps"]
        # failure times: initial pair over the first 0.25 s, then each cascade step
        t_steps = [0.0, 0.40, 0.60, 0.70, 0.75]
        self.fail_t = {}
        init = [tuple(c) for c in steps[0]["removed"]]
        for k, c in enumerate(init):
            self.fail_t[c] = 0.25 * k
        for s, st in enumerate(steps[1:], start=1):
            for c in st.get("new_failures", []):
                self.fail_t.setdefault(tuple(c), t_steps[min(s, len(t_steps) - 1)])
        self.dcr_steps = [(0.0 if s == 0 else t_steps[min(s, len(t_steps) - 1)] - 0.05,
                           {tuple(int(v) for v in k.strip("()").split(",")): val for k, val in st["dcr"].items()})
                          for s, st in enumerate(steps)]
        self.u_after = np.array(sc["u_after"]) if sc.get("u_after") else None
        self.nx, self.ny = len(self.xs), len(self.ys)

    def kin(self, t):
        r = self.rows
        z = float(np.interp(t, r[:, 0], r[:, 1]))
        v = float(np.interp(t, r[:, 0], r[:, 2]))
        rho = float(np.interp(t, r[:, 0], r[:, 3])) if t > self.T0 else np.nan
        sag = float(np.interp(t, r[:, 0], r[:, 4]))
        return max(z, 0.0), max(v, 0.0), rho, sag

    def state(self, t):
        z, v, rho, sag = self.kin(t)
        c = z / (1 - self.lam) if z > 0 else 0.0
        rubble = self.lam * c
        failed = {k: t >= tf for k, tf in self.fail_t.items()}
        dcr = self.dcr_steps[0][1]
        for ts, d in self.dcr_steps:
            if t >= ts:
                dcr = d
        return dict(t=t, z=z, v=v, rho=rho, sag=sag, c=c, rubble=rubble, failed=failed, dcr=dcr,
                    n_failed=sum(failed.values()), storeys_crushed=int(sum(self.zs[1:] < c)))

    def sag_at(self, x, sag):
        """Corner-bay sag: full on the corner column line, zero by mid-frontage."""
        return sag * np.clip(1 - (x - self.xs[0]) / (self.W / 2), 0, 1)

    def to_world(self, pts):
        return rot(pts, self.axis)

    def building_meshes(self, st, facade=True):
        """Slabs, facade panels and L0 columns for state st -> list of (mesh, kind)."""
        out = []
        z, c, sag = st["z"], st["c"], st["sag"]
        for k in range(1, self.N + 1):
            y0 = self.zs[k]
            if y0 < c - 1e-6:
                continue
            dz = -z - self.sag_at(self.xs[0] + self.W / 4, sag)
            slab = pv.Box(bounds=(self.xs[0] - 0.3, self.xs[-1] + 0.3, self.ys[0] - 0.3, self.ys[-1] + 0.3,
                                  y0 + dz - 0.12, y0 + dz + 0.12))
            out.append((slab, "slab"))
            if facade and k < self.N:
                h = self.zs[k + 1] - y0
                box = pv.Box(bounds=(self.xs[0], self.xs[-1], self.ys[0], self.ys[-1], y0 + dz + 0.12, y0 + dz + h - 0.12))
                out.append((box, "facade"))
        if c < self.zs[1]:
            for j, yy in enumerate(self.ys):
                for i, xx in enumerate(self.xs):
                    top = max(self.zs[1] - z - float(self.sag_at(xx, sag)), st["rubble"], 0.05)
                    f = st["failed"].get((i, j), False)
                    col = pv.Box(bounds=(xx - 0.3, xx + 0.3, yy - 0.3, yy + 0.3, 0, top))
                    out.append((col, "col_failed" if f else "col"))
        if st["rubble"] > 0.05:
            mound = pv.ParametricEllipsoid(self.W * 0.62, self.D * 0.68, st["rubble"])
            mound = mound.clip(normal=(0, 0, -1), origin=(0, 0, 0))
            out.append((mound, "rubble"))
        return [(m.points.__setitem__(slice(None), self.to_world(m.points)) or m, k) for m, k in out]

    def dust_points(self, t, n=3000, seed=3):
        if t <= 1.0:
            return None
        rng = np.random.default_rng(seed)
        a, b, cc, d = rng.random((4, n))
        age = t - 1.0
        grow = min(age / 3, 1)
        reach = (15 + 260 * grow) * np.sqrt(b)
        canyon = a < 0.6
        along = np.where(cc < 0.5, -1, 1) * reach
        across = (d - 0.5) * 16
        x = np.where(a < 0.35, along, self.W / 2 + 8 + across)
        y = np.where(a < 0.35, -(self.D / 2 + 8) + across, along)
        ang = cc * 2 * np.pi
        r = self.W / 2 + 6 + reach * 0.35
        x = np.where(canyon, x, np.cos(ang) * r)
        y = np.where(canyon, y, np.sin(ang) * r)
        zz = self.state(t)["rubble"] + (3 + 70 * grow ** 2) * d ** 1.6 * np.where(canyon, 0.45, 1)
        return self.to_world(np.c_[x, y, zz])
