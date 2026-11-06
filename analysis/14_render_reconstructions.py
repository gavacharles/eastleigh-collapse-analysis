"""
Step 14 - Frame-by-frame 3-D reconstructions of the collapse (MP4 + GIF + filmstrip).

  R1 eyewitness  approximate viewpoint of the phone camera across the street;
                 a local-only version sits the real video beside the reconstruction
  R2 structural  FE digital-twin cutaway: L0 columns coloured by demand/capacity,
                 failure order (scenario S6), bridging deformation, storey-by-storey
                 crush-up, with live kinematics
  R3 aerial      the collapse in its satellite-textured OSM street block, with
                 exposure rings and street-canyon dust

Every frame carries the time stamp, phase, sources and the credit line.
Usage: python 14_render_reconstructions.py [R1] [R2] [R3]
"""
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import scene3d as S
from credit import CREDIT

HERE = Path(__file__).resolve().parent
import os
OUT = Path(os.environ.get("RENDER_OUT", HERE / "out")) / "reconstructions"   # render locally, copy when done
OUT.mkdir(parents=True, exist_ok=True)
VIDEO = HERE.parent / "VID-20261023-WA0009.mp4"
pv.OFF_SCREEN = True

T_END, SLOW, FPS = 6.0, 0.25, 30            # event seconds, playback factor, video fps
WIN = (1280, 720)
FE = json.load(open(HERE / "out" / "fe" / "fe_results.json"))
PHYS = json.load(open(HERE / "out" / "physics_results.json"))

COL = dict(slab="#cfd3d6", facade="#9aa3a8", col="#2b3238", col_failed="#e5383b", rubble="#8e8676",
           tagged="#d08c3c", untagged="#7b8794", road="#46525c")


def phase(st, T0):
    t = st["t"]
    if t < 0.40:
        return "1  LOCAL FAILURE - ground-storey columns lose capacity at the street-corner end"
    if t < T0:
        return "2  ZIPPER - load shifts, neighbouring L0 columns overload and fail in turn"
    if st["z"] < 3:
        return "3  GLOBAL ONSET - the whole superstructure drops as one rigid block"
    if st["v"] > 0.05:
        return "4  CRUSH-UP - storeys crushed one by one against the growing rubble"
    return "5  AT REST - compact debris pile, dust surge along the streets"


def overlay(img, st, title, sources, T0, extra=None):
    """Caption bar, readouts, credit/sources strip (constant geometry)."""
    h, w = img.shape[:2]
    bar = np.full((64, w, 3), 18, np.uint8)
    cv2.putText(bar, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(bar, phase(st, T0), (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (90, 110, 255), 1, cv2.LINE_AA)
    rd = (f"t = {st['t']:4.2f} s   descent {st['z']:4.1f} m   v {st['v']:4.1f} m/s   "
          f"L0 columns lost {st['n_failed']}/20   storeys crushed {st['storeys_crushed']}/18")
    info = np.full((34, w, 3), 30, np.uint8)
    cv2.putText(info, rd, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    if extra:
        cv2.putText(info, extra, (w - 14 - cv2.getTextSize(extra, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0], 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cred = np.full((28, w, 3), 18, np.uint8)
    cv2.putText(cred, sources + "   |   " + CREDIT, (14, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, img, info, cred])


def plot_panel(width, height, C):
    """Pre-rendered kinematics panel; the cursor is drawn per frame."""
    r = C.rows
    fig, ax = plt.subplots(3, 1, figsize=(width / 100, height / 100), dpi=100, sharex=True)
    fig.patch.set_facecolor("#121619")
    for a in ax:
        a.set_facecolor("#121619"); a.tick_params(colors="#bbbbbb", labelsize=8)
        for s in a.spines.values():
            s.set_color("#444")
        a.axvspan(2.25, 3.75, color="#ffffff", alpha=0.05)
    ax[0].plot(r[:, 0], r[:, 1], color="#5aa9e6", lw=2)
    ax[0].plot(r[:, 0], np.where(r[:, 0] > C.T0, 0.5 * 9.81 * (r[:, 0] - C.T0) ** 2, 0), ":", color="#888")
    ax[0].set_ylim(-1, 40); ax[0].set_ylabel("descent [m]", color="#ddd", fontsize=9)
    ax[1].plot(r[:, 0], r[:, 2], color="#f0a43a", lw=2)
    ax[1].set_ylim(0, 14); ax[1].set_ylabel("velocity [m/s]", color="#ddd", fontsize=9)
    ax[2].plot(r[:, 0], r[:, 3], color="#c77dff", lw=2); ax[2].axhline(1, color="#777", lw=0.8)
    ax[2].set_ylim(0, 1.6); ax[2].set_ylabel("resistance/\nweight", color="#ddd", fontsize=9)
    ax[2].set_xlabel("time [s]", color="#ddd", fontsize=9); ax[2].set_xlim(0, T_END)
    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    x0 = ax[0].transData.transform((0, 0))[0]
    x1 = ax[0].transData.transform((T_END, 0))[0]
    plt.close(fig)
    return img, x0, x1


def cursor(panel, t, x0, x1):
    p = panel.copy()
    x = int(x0 + (x1 - x0) * t / T_END)
    cv2.line(p, (x, 0), (x, p.shape[0]), (229, 56, 59), 2)
    return p


class Writer:
    """Streams MP4 and GIF to disk; keeps only the frames needed for the filmstrip."""
    FILM_T = (0.0, 0.3, 0.6, 1.0, 1.6, 2.2, 3.0, 4.0, 5.0, 6.0)

    def __init__(self, name, gif_every=6, gif_w=480):
        self.mp4 = imageio.get_writer(OUT / f"{name}.mp4", fps=FPS, codec="libx264", quality=7, macro_block_size=1)
        self.gif = imageio.get_writer(OUT / f"{name}.gif", mode="I", duration=gif_every / FPS, loop=0)
        self.every, self.gw, self.name, self.n = gif_every, gif_w, name, 0
        self.film = {}

    def add(self, rgb, st=None):
        self.mp4.append_data(rgb)
        if self.n % self.every == 0:
            h = int(rgb.shape[0] * self.gw / rgb.shape[1]) // 2 * 2
            self.gif.append_data(cv2.resize(rgb, (self.gw, h), interpolation=cv2.INTER_AREA))
        if st is not None:
            for ft in self.FILM_T:
                dt = abs(st["t"] - ft)
                if ft not in self.film or dt < self.film[ft][0]:
                    self.film[ft] = (dt, cv2.resize(rgb, (480, int(rgb.shape[0] * 480 / rgb.shape[1]))))
        self.n += 1

    def close(self):
        self.mp4.close()
        self.gif.close()
        if self.film:
            tiles = [self.film[ft][1] for ft in self.FILM_T]
            rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
            cv2.imwrite(str(OUT / f"{self.name}_filmstrip.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("wrote", self.name, self.n, "frames")


def city_context(p, half_deg=0.0028, z=18, buildings=True, rings=False, exclude=None):
    img, b = S.esri_mosaic(half_deg, z)
    plane, tex = S.textured_plane(img, b)
    p.add_mesh(plane, texture=tex, ambient=0.35, diffuse=0.75)
    tagged, untagged, gis = S.osm_buildings(exclude=exclude)
    if buildings:
        if untagged is not None:
            p.add_mesh(untagged, color=COL["untagged"], opacity=0.92, smooth_shading=False)
        if tagged is not None:
            p.add_mesh(tagged, color=COL["tagged"], smooth_shading=False)
    if rings:
        p.add_mesh(S.ring(55), color="#e5383b", opacity=0.85)
        p.add_mesh(S.ring(110), color="#f0a43a", opacity=0.85)
    return gis


def draw_building(p, C, st, actors, facade_op=0.8):
    for a in actors:
        p.remove_actor(a)
    actors.clear()
    for m, kind in C.building_meshes(st):
        op = facade_op if kind == "facade" else 1.0
        actors.append(p.add_mesh(m, color=COL[kind], opacity=op, smooth_shading=kind == "rubble",
                                 ambient=0.45 if kind == "rubble" else 0.15))
    pts = C.dust_points(st["t"])
    if pts is not None:
        actors.append(p.add_points(pts, color="#d8cbb3", point_size=5, opacity=0.33, render_points_as_spheres=False))


def times():
    n = int(T_END / SLOW * FPS)
    return np.linspace(0, T_END, n)


# ------------------------------------------------------------------ R1 eyewitness
def render_R1():
    p = pv.Plotter(off_screen=True, window_size=WIN)
    p.set_background("#b9c9d6", top="#6f93b5")
    axis = S.ninth_axis(json.load(open(HERE / "gis" / "web_layers.json")))
    cam_l = np.array([22.5 * 0.9, -48.0])
    cam_w = S.rot(cam_l[None], axis)[0]

    def in_sightline(c, cam=cam_w):
        # distance from footprint centroid to the camera-building segment
        d = cam / np.linalg.norm(cam)
        tproj = np.clip(np.dot(c, d), 0, np.linalg.norm(cam) + 10)
        return np.linalg.norm(c - tproj * d) < 26
    gis = city_context(p, exclude=in_sightline)
    C = S.Collapse(axis)
    # approximate phone position: across Ninth Street, ~45 m out, offset towards the
    # non-corner end, ~14 m up (balcony); looking at the lower half of the frontage
    cam_local = np.array([C.W * 0.9, -48.0, 14.0])
    foc_local = np.array([-C.W * 0.15, 0.0, 22.0])
    cam = S.rot(cam_local[None], C.axis)[0]
    foc = S.rot(foc_local[None], C.axis)[0]
    p.camera_position = [tuple(cam), tuple(foc), (0, 0, 1)]
    p.camera.view_angle = 50
    cap = cv2.VideoCapture(str(VIDEO))            # frames read on demand (low memory)
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cur = {"i": -1, "f": None}

    def video_frame(i):
        while cur["i"] < i:
            ok, f = cap.read()
            if not ok:
                break
            cur["i"] += 1
            cur["f"] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        return cur["f"]
    w3 = Writer("R1_eyewitness_3d")
    wsbs = imageio.get_writer(OUT / "R1_eyewitness_side_by_side_LOCAL.mp4", fps=FPS, codec="libx264", quality=7,
                              macro_block_size=1)
    actors = []
    for t in times():
        st = C.state(t)
        draw_building(p, C, st, actors)
        img = np.ascontiguousarray(p.screenshot(return_img=True))
        frame = overlay(img, st, "R1  Eyewitness-view reconstruction (approximate camera position)",
                        "Data: video kinematics; crush-up/PINN; FE twin S6; OSM; Esri imagery. Site: LR 36/II/40 (position approx.)", C.T0)
        w3.add(frame, st)
        vf = video_frame(min(int(round(t * 29.97)), nvid - 1))
        vf = cv2.resize(vf, (int(vf.shape[1] * img.shape[0] / vf.shape[0]), img.shape[0]))
        sbs = np.hstack([vf, img])
        sbs = overlay(sbs, st, "R1  Eyewitness video (left) vs 3-D reconstruction (right) - LOCAL ONLY",
                      "Data: eyewitness video VID-20261023-WA0009; reconstruction (this analysis)", C.T0)
        wsbs.append_data(sbs)
    wsbs.close()
    w3.close()
    p.close()


# ------------------------------------------------------------------ R2 structural
def render_R2():
    p = pv.Plotter(off_screen=True, window_size=(860, 720))
    p.set_background("#14191d")
    C = S.Collapse(0.0)                      # building frame = world frame (no city)
    ground = pv.Plane(center=(0, 0, -0.05), i_size=120, j_size=120, i_resolution=24, j_resolution=24)
    p.add_mesh(ground, color="#2a3137", show_edges=True, edge_color="#3a444c")
    fe = json.load(open(HERE / "out" / "fe" / "fe_anim.json"))
    geo = fe["geo"]
    xs = np.array(geo["xs"]) - geo["xs"][-1] / 2
    ys = np.array(geo["ys"]) - geo["ys"][-1] / 2
    zs = np.array(geo["zs"])
    nx, ny = len(xs), len(ys)
    nodes = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    u_after = np.array(fe["anim"][C.SCEN]["u_after"])
    u_int = np.array(fe["anim"][C.SCEN]["u_intact"])
    du = u_after - u_int
    beams = np.array(geo["beams"])
    cols = geo["cols"]
    cmap = plt.get_cmap("RdYlGn_r")
    panel, px0, px1 = plot_panel(420, 720, C)
    sat = FE["scenarios"][C.SCEN]
    pd_poor = FE["pushdown"]["poor (3T16 top / 2T12 bottom)"]
    notes = {1: f"Corner pair lost: load re-routes through beams (Vierendeel); quarter period of local mode ~{FE['modal']['zipper_stage_time']:.2f} s",
             2: f"Sudden loss -> dynamic demand ~2x the static increment; neighbours exceed capacity (DCR > 1)",
             3: f"L0 reserve sum P0/W = {sat['sumP0_over_W']:.2f}; once L0 is gone, crushing force < weight -> unstoppable",
             4: f"Measured resistance ~{PHYS['rho_free_early'][0]:.2f} W early, rising as the moving mass falls (crush-up)",
             5: "Debris pile ~25-35 % of original height; dust expelled along the street canyons"}
    w = Writer("R2_structural_twin")
    actors = []
    p.camera_position = [(-58, -62, 40), (0, 0, 22), (0, 0, 1)]
    for fi, t in enumerate(times()):
        for a in actors:
            p.remove_actor(a)
        actors.clear()
        st = C.state(t)
        # deformation amplification ramps in during the local phase, then rigid descent
        amp = 40.0 * min(t / 0.6, 1.0) if t < 1.2 else 40.0 * max(0.0, 1 - (t - 1.2) / 0.6)
        disp = np.zeros_like(nodes)
        disp[:, 2] -= st["z"]
        disp += du * amp
        P = nodes + disp
        alive_z = zs >= st["c"] - 1e-6
        lines, colors = [], []
        for (a, b) in beams:
            ka, kb = a // (nx * ny), b // (nx * ny)
            if alive_z[ka] and alive_z[kb]:
                lines.append((P[a], P[b]))
        for (n, storey, ij, size) in cols:
            a, b = n
            if storey == 1:
                continue
            ka, kb = a // (nx * ny), b // (nx * ny)
            if alive_z[ka] and alive_z[kb]:
                lines.append((P[a], P[b]))
        if lines:
            seg = np.array(lines).reshape(-1, 3)
            mesh = pv.line_segments_from_points(seg)
            actors.append(p.add_mesh(mesh, color="#c9d1d6", line_width=1.6))
        # L0 columns: colour by DCR, red dashed (thin) when failed
        if st["c"] < zs[1]:
            for j in range(ny):
                for i in range(nx):
                    top_node = nodes[(1 * ny + j) * nx + i] + disp[(1 * ny + j) * nx + i]
                    top = max(top_node[2], st["rubble"], 0.05)
                    failed = st["failed"].get((i, j), False)
                    d = st["dcr"].get((i, j), 0.0)
                    if failed:
                        m = pv.Cylinder(center=(xs[i], ys[j], top / 2), direction=(0, 0, 1), radius=0.18, height=top)
                        actors.append(p.add_mesh(m, color="#ff4d4f"))
                        actors.append(p.add_mesh(pv.Sphere(radius=0.7, center=(xs[i], ys[j], top / 2)), color="#ff4d4f"))
                    else:
                        m = pv.Cylinder(center=(xs[i], ys[j], top / 2), direction=(0, 0, 1), radius=0.35, height=top)
                        actors.append(p.add_mesh(m, color=cmap(min(d, 1.2) / 1.2)[:3]))
        # crushing front flash
        if st["c"] > zs[1] and st["v"] > 0.5:
            zc = st["rubble"]
            band = pv.Box(bounds=(xs[0] - 1, xs[-1] + 1, ys[0] - 1, ys[-1] + 1, zc, zc + 1.2))
            actors.append(p.add_mesh(band, color="#ff8c42", opacity=0.55))
        if st["rubble"] > 0.05:
            mound = pv.ParametricEllipsoid(C.W * 0.62, C.D * 0.68, st["rubble"]).clip(normal=(0, 0, -1), origin=(0, 0, 0))
            actors.append(p.add_mesh(mound, color=COL["rubble"], smooth_shading=True))
        pts = C.dust_points(t, n=1500)
        if pts is not None:
            actors.append(p.add_points(pts, color="#d8cbb3", point_size=4, opacity=0.25))
        p.camera.azimuth = -0.06                           # slow orbit
        img = np.ascontiguousarray(p.screenshot(return_img=True))
        stage = 1 if t < 0.4 else 2 if t < C.T0 else 3 if st["z"] < 3 else 4 if st["v"] > 0.05 else 5
        cv2.putText(img, notes[stage][:95], (12, img.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        # colour key
        for k in range(100):
            c = (np.array(cmap(k / 100 * 1.0)[:3]) * 255).astype(int).tolist()
            cv2.line(img, (12 + k * 2, 14), (12 + k * 2, 26), c, 2)
        cv2.putText(img, "L0 column demand/capacity 0 ... 1.2", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, f"deformation x{amp:.0f}" if amp > 1 else "", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        frame = np.hstack([img, cursor(panel, t, px0, px1)])
        frame = overlay(frame, st, "R2  Structural reconstruction - finite-element digital twin (scenario S6)",
                        "Data: FE twin (assumed sections), crush-up/PINN fit to video, Izzuddin pushdown", C.T0)
        w.add(frame, st)
    w.close()
    p.close()


# ------------------------------------------------------------------ R3 aerial
def render_R3():
    p = pv.Plotter(off_screen=True, window_size=WIN)
    p.set_background("#9fb3c4", top="#4f6f8f")
    gis = city_context(p, rings=True)
    C = S.Collapse(S.ninth_axis(gis))
    w = Writer("R3_aerial_gis")
    actors = []
    ts = times()
    for fi, t in enumerate(ts):
        st = C.state(t)
        draw_building(p, C, st, actors)
        ang = np.radians(200 + 70 * fi / len(ts))
        r, hgt = 240 - 70 * fi / len(ts), 190
        p.camera_position = [(r * np.cos(ang), r * np.sin(ang), hgt), (0, 0, 22), (0, 0, 1)]
        img = np.ascontiguousarray(p.screenshot(return_img=True))
        frame = overlay(img, st, "R3  Aerial reconstruction in the real street block (55 m / 110 m exposure rings)",
                        "Data: OSM (untagged = 4 storeys), Esri imagery, crush-up/PINN, FE twin S6. Site: LR 36/II/40 (approx.)", C.T0)
        w.add(frame, st)
    w.close()
    p.close()


if __name__ == "__main__":
    which = sys.argv[1:] or ["R2"]          # R1/R3 retired: viewpoint/occlusion did not render well
    for wname in which:
        {"R1": render_R1, "R2": render_R2, "R3": render_R3}[wname]()
