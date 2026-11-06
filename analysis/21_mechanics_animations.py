"""
Step 21 - Mechanics animations and explainer graphics.

  V3  anim_pushdown_hinges.mp4/.gif   corner-bay pushdown: plastic hinges appear in order,
                                      synced with the capacity curve (static + energy-based dynamic)
  V9  anim_column_crushing.mp4/.gif   L0 column crushing: elastic -> cover spalling -> bar
                                      buckling -> core crushing -> rubble, synced with force-shortening
  V6  fig_optical_flow.jpg            how the video was measured: camera-compensated flow field,
                                      tracked patches, storey ruler  (LOCAL: contains footage)
  V10 fig_timeline.jpg                five-stage timeline graphic for social media
Usage: python 21_mechanics_animations.py [V3] [V9] [V6] [V10]
"""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit import CREDIT, SRC_TWIN, footer, stamp

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "explainers"
OUT.mkdir(parents=True, exist_ok=True)
FPS = 30


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Out:
    def __init__(self, name, gif_every=3, gif_w=720):
        self.mp4 = imageio.get_writer(OUT / f"{name}.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        self.gif = imageio.get_writer(OUT / f"{name}.gif", mode="I", duration=gif_every / FPS, loop=0)
        self.n, self.every, self.gw = 0, gif_every, gif_w

    def add(self, fig):
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        self.mp4.append_data(img)
        if self.n % self.every == 0:
            h = int(img.shape[0] * self.gw / img.shape[1]) // 2 * 2
            self.gif.append_data(cv2.resize(img, (self.gw, h), interpolation=cv2.INTER_AREA))
        self.n += 1

    def close(self):
        self.mp4.close(); self.gif.close()


# ------------------------------------------------------------------ V3 pushdown hinges
def v3_pushdown():
    FE = load("fe", "10_fe_twin.py")
    np.seterr(all="ignore")
    T20, T16, T12 = 314e-6, 201e-6, 113e-6
    poor = FE.pushdown(3 * T16, 2 * T12)
    design = FE.pushdown(4 * T20, 3 * T16)
    p = FE.PARAMS
    xs = np.r_[0, np.cumsum(p["bays_x"])]
    zs = np.r_[0, p["h_ground"] + p["h_typ"] * np.arange(p["storeys"])]
    curve = np.array(poor["curve"])
    hinges = poor["hinges"]
    exag = 4.0
    span = p["bays_x"][0]

    def sag(x, u):
        return u * np.clip(1 - x / span, 0, 1) * exag

    fig = plt.figure(figsize=(13.5, 7.6), dpi=100)
    ax = fig.add_axes([0.03, 0.08, 0.42, 0.84])
    ac = fig.add_axes([0.53, 0.14, 0.44, 0.74])
    out = Out("anim_pushdown_hinges")
    n_ev = len(hinges)
    per = 9
    frames = [(k, s / per) for k in range(n_ev) for s in range(per)] + [(n_ev - 1, 1.0)] * 45
    for k, frac in frames:
        i0 = min(k + 1, len(curve) - 1); i1 = min(k + 2, len(curve) - 1)
        lam = curve[i0, 0] + (curve[i1, 0] - curve[i0, 0]) * frac
        u = curve[i0, 1] + (curve[i1, 1] - curve[i0, 1]) * frac
        ax.clear(); ac.clear()
        # frame (removed corner column at L0)
        xx = np.linspace(0, xs[-1], 200)
        for z in zs[1:]:
            ax.plot(xx, z - sag(xx, u), color="0.25", lw=1.2)
        for x in xs:
            z0 = 0 if x > 0 else zs[1]
            if x == 0:
                ax.plot([0, 0], [0, zs[1] - 0.6], color="#c62f2f", lw=1.5, ls=":")
                ax.plot(0, zs[1] / 2 - 0.5, "x", color="#c62f2f", ms=12, mew=3)
            ax.plot([x, x], [z0 - (sag(x, u) if x == 0 else 0), zs[-1] - sag(x, u)], color="0.25", lw=1.6)
        ax.axvspan(0, span, color="#1f6aa5", alpha=0.06)
        for h in hinges[:k + 1]:
            ax.plot(h["x"], h["z"] - sag(h["x"], u), "o", color="#e5383b", ms=6, mec="k", mew=0.4)
        ax.fill_between([-2, xs[-1] + 2], -1, 0, color="0.7")
        ax.set_xlim(-3, xs[-1] + 3); ax.set_ylim(-1.5, zs[-1] + 2); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"Frontage frame over the lost corner column\nplastic hinges: {k + 1}/{n_ev}   (sag x{exag:.0f})", fontsize=11)
        # capacity curve
        dc = np.array(design["curve"])
        ac.plot(dc[:, 1] * 1000, dc[:, 0], color="#1f6aa5", lw=1.2, alpha=0.45, label="design reinforcement (static)")
        ac.plot(np.array(design["u"]) * 1000, design["lam_dyn"], color="#1f6aa5", lw=1, ls="--", alpha=0.45,
                label="design reinforcement (dynamic)")
        m = curve[:, 1] <= u + 1e-9
        ac.plot(curve[m, 1] * 1000, curve[m, 0], color="#c62f2f", lw=2.4, label="poor reinforcement (static)")
        uu = np.array(poor["u"]); ld = np.array(poor["lam_dyn"])
        ac.plot(uu[uu <= u] * 1000, ld[uu <= u], color="#c62f2f", lw=1.6, ls="--", label="poor reinforcement (dynamic, energy)")
        ac.plot(u * 1000, lam, "o", color="#c62f2f", ms=9)
        ac.axhline(1, color="k", lw=1)
        ac.text(980, 1.03, "load to carry (D + 0.25L)", ha="right", fontsize=9)
        ac.set(xlim=(0, 1000), ylim=(0, 1.4), xlabel="sag over the lost corner column [mm]",
               ylabel="load factor carried by bridging", title="Pushdown capacity: poor bridging stays below the load")
        ac.legend(fontsize=8, loc="lower right")
        fig.texts.clear()
        footer(fig, SRC_TWIN, "event-to-event plastic hinges; dynamic capacity per Izzuddin et al. (2008)", fontsize=8)
        out.add(fig)
    out.close()
    fig.savefig(OUT / "fig_pushdown_hinges_final.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ V9 column crushing
def v9_column():
    FE = load("fe", "10_fe_twin.py")
    a, L, cov = 0.60, 4.0, 0.045
    d15, F15 = FE.column_crush(a=a, fc=15)
    d30, F30 = FE.column_crush(a=a, fc=30)
    lz = min(1.5 * a, L)
    def eps_of(delta):
        return delta / L if delta < 0.004 * L else 0.004 + (delta - 0.004 * L) / (lz * 4)
    n = 480
    tau = np.linspace(0, 1, n)
    deltas = np.where(tau < 0.45, 0.06 * (tau / 0.45) ** 2, 0.06 + (2.0 - 0.06) * ((tau - 0.45) / 0.55) ** 1.6)
    rng = np.random.default_rng(4)
    chips = [(rng.uniform(-1, 1), rng.uniform(0.15, 0.95), rng.uniform(0.5, 1.5)) for _ in range(40)]
    fig = plt.figure(figsize=(13.5, 7.6), dpi=100)
    ax = fig.add_axes([0.04, 0.08, 0.36, 0.84])
    ac = fig.add_axes([0.48, 0.14, 0.49, 0.74])
    out = Out("anim_column_crushing")
    for fi, delta in enumerate(deltas):
        e = eps_of(delta)
        ax.clear(); ac.clear()
        crush_h = lz if delta <= 0.004 * L else max(lz - (delta - 0.004 * L), 0.12)
        top_shift = delta
        bulge = 0.0 if e < 0.003 else min(0.10, 0.6 * (e - 0.003))
        # upper (intact) part
        y0 = crush_h
        ax.add_patch(plt.Rectangle((-a / 2, y0), a, L - lz, color="#c9cdd0", ec="0.3"))
        # crush zone: core
        w = a - 2 * cov + 2 * bulge
        ax.add_patch(plt.Rectangle((-w / 2, 0), w, crush_h, color="#9aa3a8" if e < 0.006 else "#7d7466", ec="0.3"))
        # cover in crush zone, present until spalling
        if e < 0.003:
            for sgn in (-1, 1):
                ax.add_patch(plt.Rectangle((sgn * (a / 2) - (cov if sgn > 0 else 0), 0), cov, crush_h, color="#c9cdd0", ec="0.3"))
        else:
            age = min((e - 0.003) / 0.02, 1.0)
            for (sx, sy, sp) in chips:
                ax.add_patch(plt.Rectangle((np.sign(sx) * (a / 2 + 0.05 + 0.6 * age * sp * abs(sx)),
                                            max(sy * crush_h - 1.2 * age ** 2 * sp, 0.02)), 0.05, 0.04, color="#b0b5b8"))
        # bars
        for sgn in (-1, 1):
            xb = sgn * (a / 2 - cov - 0.01)
            yy = np.linspace(0, crush_h, 200)
            amp = 0.0 if e < 0.004 else min(0.22, 2.5 * (e - 0.004))
            s_t = 0.30
            bow = amp * np.abs(np.sin(np.pi * yy / s_t)) * sgn
            ax.plot(xb + bow, yy, color="#2b2b2b", lw=2.2)
            ax.plot([xb, xb], [y0, L - lz + y0], color="#2b2b2b", lw=2.2)
        for yt in np.arange(0.15, L - top_shift, 0.30):
            ax.plot([-a / 2 + cov, a / 2 - cov], [yt, yt], color="#555", lw=0.8)
        if delta > 0.1:
            r = min(delta / 2.0, 1) * 0.5
            ax.add_patch(plt.Polygon([[-a / 2 - 0.5 * r - 0.2, 0], [a / 2 + 0.5 * r + 0.2, 0], [0.1, r]], color="#8e8676"))
        ax.fill_between([-1.2, 1.2], -0.1, 0, color="0.6")
        ax.annotate("", (0, L - top_shift + 0.05), (0, L - top_shift + 0.55), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#c62f2f"))
        ax.text(0.08, L - top_shift + 0.35, "weight of the\nbuilding above", fontsize=9, color="#c62f2f")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.1, L + 0.8); ax.set_aspect("equal"); ax.axis("off")
        stage = ("1  elastic: concrete and steel share the load" if e < 0.002 else
                 "2  peak: concrete near its strength" if e < 0.003 else
                 "3  cover spalls off" if e < 0.004 else
                 "4  bars buckle between widely spaced ties" if e < 0.02 else
                 "5  core crushes; rubble carries only friction")
        ax.set_title(f"Ground-storey column, 600 x 600 mm\n{stage}", fontsize=11)
        # curve
        F15s, F30s = F15 / 1000, F30 / 1000
        ac.plot(d30 * 1000, F30s, color="#1f6aa5", lw=1.4, alpha=0.6, label="f'c 30 MPa")
        m = d15 <= delta
        ac.plot(d15 * 1000, F15s, color="#c62f2f", lw=1, alpha=0.25)
        ac.plot(d15[m] * 1000, F15s[m], color="#c62f2f", lw=2.4, label="f'c 15 MPa (animated)")
        ac.plot(delta * 1000, np.interp(delta, d15, F15s), "o", color="#c62f2f", ms=9)
        ac.set_xscale("symlog", linthresh=20)
        ac.set(xlim=(0, 2100), ylim=(0, 14.5), xlabel="shortening [mm]  (linear to 20 mm, log beyond)",
               ylabel="axial force [MN]", title="Force-shortening: the peak is brief, the crushed residue is small")
        ac.axhline(np.interp(2.0, d15, F15s), color="#c62f2f", ls=":", lw=1)
        ac.legend(fontsize=9)
        fig.texts.clear()
        footer(fig, "column model: Mander concrete + bar buckling (this analysis); illustrative geometry", fontsize=8)
        out.add(fig)
    out.close()
    plt.close(fig)


# ------------------------------------------------------------------ V6 optical flow (LOCAL)
def v6_flow():
    src = Path(sys.argv[0]).resolve().parents[3] / "all"     # scratch/all frame dump
    if not src.exists():
        src = HERE.parent.parent / "all"
    pairs = [(6, 14, "0.2-0.47 s: corner end starts to drop"), (20, 28, "0.67-0.93 s: drop spreads"),
             (44, 50, "1.47-1.67 s: block descends; dust jets"), (56, 62, "1.87-2.07 s: rigid-block descent")]
    P = {"U": (262, 238, 312, 270), "L": (255, 444, 307, 472), "LL": (140, 440, 200, 478), "G": (300, 522, 378, 556)}
    tiles = []
    for a_, b_, cap in pairs:
        f0 = cv2.imread(str(src / f"{a_:03d}.png")); f1 = cv2.imread(str(src / f"{b_:03d}.png"))
        g0, g1 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 21, 5, 7, 1.5, 0)
        H, Wd = g0.shape
        yy, xx = np.mgrid[0:H, 0:Wd]
        stat = np.zeros_like(g0, bool); stat[0:860, 400:480] = True; stat[600:860, 0:120] = True
        A = np.c_[xx[stat], yy[stat], np.ones(stat.sum())]
        cx = np.linalg.lstsq(A, flow[..., 0][stat], rcond=None)[0]
        cy = np.linalg.lstsq(A, flow[..., 1][stat], rcond=None)[0]
        res_x = flow[..., 0] - (cx[0] * xx + cx[1] * yy + cx[2])
        res_y = flow[..., 1] - (cy[0] * xx + cy[1] * yy + cy[2])
        img = f0.copy()
        k = 9
        rx = cv2.blur(res_x, (k, k)); ry = cv2.blur(res_y, (k, k))
        for y in range(70, 560, 34):
            for x in range(130, 395, 34):
                dx, dy = rx[y, x], ry[y, x]
                mag = np.hypot(dx, dy)
                if mag < 0.6 or mag > 12:
                    continue
                down = dy > 0.6 and abs(dx) < 1.5 * dy
                c = (40, 40, 255) if down else (220, 200, 120)
                cv2.arrowedLine(img, (x, y), (int(x + 4 * dx), int(y + 4 * dy)), c, 2 if down else 1, tipLength=0.3)
        for k, (x0, y0, x1, y1) in P.items():
            cv2.rectangle(img, (x0, y0), (x1, y1), (120, 255, 120) if k == "G" else (0, 220, 255), 1)
            cv2.putText(img, k, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.line(img, (316, 254), (316, 458), (255, 0, 255), 2)
        cv2.putText(img, "ruler: 6 storeys", (322, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)
        cv2.rectangle(img, (0, 0), (480, 30), (0, 0, 0), -1)
        cv2.putText(img, cap, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(img[:700])
    grid = np.hstack(tiles)
    leg = np.full((40, grid.shape[1], 3), 20, np.uint8)
    cv2.putText(leg, "Arrows: smoothed motion after removing camera pan/zoom (fitted on static buildings), x4. Red = downward. "
                "Boxes: tracked patches (U, L, LL facade; G static wall) - the quantitative measurements; dense flow is qualitative.", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    grid = stamp(np.vstack([grid, leg]), "eyewitness video VID-20261023-WA0009; Farneback optical flow (this analysis)",
                 "LOCAL - shows bystanders", scale=0.5)
    cv2.imwrite(str(OUT / "fig_optical_flow_LOCAL.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ------------------------------------------------------------------ V10 timeline graphic
def v10_timeline():
    import pyvista as pv
    import scene3d as S
    pv.OFF_SCREEN = True
    C = S.Collapse(0.0)
    stages = [(-0.1, "BEFORE", "Shops occupied at ground level,\nno occupation certificate;\nrubble and missing blockwork at L0"),
              (0.35, "0 - 0.4 s", "Corner columns lose capacity;\ncorner drops ~0.6 m"),
              (0.75, "0.4 - 0.8 s", "Neighbours overload and fail:\nall 20 ground-storey columns"),
              (2.3, "0.8 - 2.3 s", "Whole building drops as one block;\nresistance only ~0.4-0.5 x weight"),
              (4.3, "2.3 - 5.5 s", "Crushed storey by storey at up to\n12 m/s; 1.3-1.7 GJ released")]
    cols = {"slab": "#e8eaec", "facade": "#a7b0b6", "col": "#2b3238", "col_failed": "#ff2d2d", "rubble": "#8e8676"}
    tiles = []
    for t, head, body in stages:
        p = pv.Plotter(off_screen=True, window_size=(520, 760))
        p.set_background("#f4f5f6")
        p.add_mesh(pv.Plane(i_size=90, j_size=90), color="#d9dcdf")
        st = C.state(t)
        st = dict(st, sag=st["sag"] * 3)
        for m, kind in C.building_meshes(st):
            p.add_mesh(m, color=cols[kind], opacity=0.85 if kind == "facade" else 1.0)
        pts = C.dust_points(t, n=1500)
        if pts is not None:
            p.add_points(pts, color="#c8b99f", point_size=4, opacity=0.35)
        p.camera_position = [(-62, -78, 40), (0, 0, 22), (0, 0, 1)]
        im = np.ascontiguousarray(p.screenshot(return_img=True)); p.close()
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        band = np.full((170, im.shape[1], 3), 255, np.uint8)
        cv2.putText(band, head, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (44, 67, 181), 2, cv2.LINE_AA)
        for k, ln in enumerate(body.split("\n")):
            cv2.putText(band, ln, (16, 80 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(np.vstack([band, im]))
    sep = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    row = tiles[0]
    for t_ in tiles[1:]:
        row = np.hstack([row, sep, t_])
    title = np.full((90, row.shape[1], 3), 255, np.uint8)
    cv2.putText(title, "How the Eastleigh building came down (preliminary reconstruction)", (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (20, 20, 20), 2, cv2.LINE_AA)
    img = stamp(np.vstack([title, row]), "eyewitness video kinematics; crush-up/PINN; FE twin (scenario S6); press reports",
                "corner sag x3 for visibility", scale=0.6)
    cv2.imwrite(str(OUT / "fig_timeline.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    which = sys.argv[1:] or ["V3", "V9", "V6", "V10"]
    for w in which:
        {"V3": v3_pushdown, "V9": v9_column, "V6": v6_flow, "V10": v10_timeline}[w]()
        print("done", w)
