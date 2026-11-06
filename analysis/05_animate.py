"""
Step 5 - Animated reconstruction, synchronised with the eyewitness video.

Left : eyewitness frame at the same video time
Middle: schematic street elevation driven by the identified physics
        - 0 to t0: L0 column zipper from the street-corner end (sequence from
          the digital-twin cascade, timing schematic); the column line above a
          lost column sags by the *measured* left-end drop (alternate-path
          deformation, not rigid rotation)
        - t0 onward: tapered crush-up ODE (step 2): block descends z(t); storeys
          whose original elevation < c = z/(1-lam) are crushed into a rubble
          layer of height lam*c
Right : z(t), v(t), rho(t) with CV measurements and a moving cursor

Writes out/collapse_reconstruction.mp4 (with video, 0.5x speed),
       out/collapse_reconstruction.gif (schematic + plots, real time),
       out/animation_timeline.json (for the interactive page)
"""
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from credit import SRC_CV, SRC_MODEL, SRC_TWIN, SRC_VIDEO, footer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
VIDEO = HERE.parent / "VID-20261023-WA0009.mp4"
G = 9.81

phys = json.load(open(OUT / "physics_results.json"))
K, BETA, T0 = phys["k_taper"], phys["beta_taper"], phys["t0_taper"]
N, H_TYP, H_GND, LAM = 18, 3.0, 4.0, 0.25
BAYS = 5
W = 22.5
XS = np.linspace(0, W, BAYS + 1)
FLOORS = np.r_[H_GND + H_TYP * np.arange(N)]          # slab elevations L1..roof
H_TOT = FLOORS[-1]
# L0 zipper (digital-twin S4 sequence; times schematic but bounded by video)
COL_FAIL_T = np.array([-0.2, 0.40, 0.60, 0.60, 0.75, 0.75])
T_END = 6.0


# ------------------------------------------------------------------ kinematics
def crushup(t_grid):
    def rhs(_, y):
        n = y[0] / ((1 - LAM) * H_TYP)
        rho = K * max(1 - BETA * n / N, 0.05) / max(N - n, 0.5)
        return [y[1], G * (1 - rho) if (y[1] > 0 or rho < 1) else 0.0]
    z = np.zeros_like(t_grid)
    v = np.zeros_like(t_grid)
    m = t_grid > T0
    sol = solve_ivp(rhs, (T0, t_grid[-1] + 1e-6), [0.05, 0.0], t_eval=t_grid[m], max_step=0.01)
    z[m], v[m] = sol.y
    v = np.maximum(v, 0)
    rho = np.where(m, K * np.maximum(1 - BETA * (z / ((1 - LAM) * H_TYP)) / N, 0.05)
                   / np.maximum(N - z / ((1 - LAM) * H_TYP), 0.5), np.nan)
    return z, v, rho


tg = np.linspace(0, T_END, 1201)
zg, vg, rhog = crushup(tg)
raw = np.genfromtxt(OUT / "kinematics.csv", delimiter=",", names=True, dtype=None, encoding=None)
tilt = raw[raw["segment"] == "A_left_minus_centre"]
tilt_t = np.r_[tilt["t_s"]]
tilt_z = np.maximum.accumulate(np.clip(np.r_[tilt["z_m"]], 0, None))


def left_drop(t):
    return float(np.interp(t, np.r_[0, tilt_t], np.r_[0, tilt_z]))


def state(t):
    z = float(np.interp(t, tg, zg))
    c = z / (1 - LAM) if z > 0 else 0.0          # original height already crushed
    rubble = LAM * c
    sag = left_drop(t)
    # sag profile: full at col line 0, zero by mid-frontage (alternate-path)
    sag_x = sag * np.clip(1 - XS / (W / 2), 0, 1)
    failed = t >= COL_FAIL_T
    return z, c, rubble, sag_x, failed


# ------------------------------------------------------------------ drawing
def draw_building(ax, t):
    z, c, rubble, sag_x, failed = state(t)
    ax.clear()
    ax.set_xlim(-14, W + 14)
    ax.set_ylim(-1, H_TOT + 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.fill_between([-14, W + 14], -1, 0, color="#8d8d8d")
    # neighbours (static)
    ax.add_patch(plt.Rectangle((-13, 0), 11, 3 + 3 * 13, color="#e9e2d0", ec="#b9ad90"))
    ax.add_patch(plt.Rectangle((W + 2, 0), 11, 3 + 3 * 7, color="#d8c9b5", ec="#a89478"))
    # rubble
    if rubble > 0:
        xs = np.linspace(-1.5, W + 1.5, 40)
        prof = rubble * (1 - 0.15 * np.cos(np.linspace(0, 2 * np.pi, 40))) * np.clip(
            1 - (np.abs(xs - W / 2) - W / 2) / 1.5, 0, 1)
        ax.fill_between(xs, 0, prof, color="#9b8f7d", zorder=3)
    # L0 storey (before crush)
    col_top_L0 = H_GND - (z if z > 0 else 0) - sag_x
    for i, x in enumerate(XS):
        if c < H_GND:
            top = max(col_top_L0[i], rubble)
            if failed[i]:
                ax.plot([x, x], [0, top], color="#d1495b", lw=2.2, ls=(0, (2, 2)), zorder=4)
                ax.plot(x, top / 2, "x", color="#d1495b", ms=7, mew=2, zorder=5)
            else:
                ax.plot([x, x], [0, top], color="#3d3d3d", lw=3, zorder=4)
    # upper block: floors with original elevation >= c survive, shifted down by z
    alive = FLOORS[FLOORS >= c - 1e-9]
    for j, y0 in enumerate(alive):
        y = y0 - z - sag_x
        ax.plot(XS, y, color="#555", lw=1.6, zorder=6)
        if y0 < H_TOT:
            y_up = y0 + H_TYP - z - sag_x
            for i, x in enumerate(XS):
                ax.plot([x, x], [y[i], y_up[i]], color="#777", lw=1.4, zorder=5)
            ax.fill_between(XS, y, y_up, color="#c7c2b8", alpha=0.5, zorder=4, lw=0)
    # dust jets from L0 while it is being crushed and during crush-up
    if t > 1.0:
        rng = np.random.default_rng(int(t * 1000) % 2**31)
        s = min((t - 1.0) / 2.0, 1.0)
        for side in (-1, 1):
            for _ in range(10):
                d = rng.uniform(0.3, 1) * (3 + 10 * s)
                xx = (W / 2 + side * (W / 2 + d))
                yy = rubble + rng.uniform(0, 3 + 6 * s)
                ax.add_patch(plt.Circle((xx, yy), rng.uniform(0.8, 2.2) * (0.6 + s),
                                        color="#cdbfa6", alpha=0.25 * (0.6 + 0.4 * s), zorder=7, lw=0))
    phase = ("Local failure at L0 corner" if t < 0.4 else
             "Zipper along L0 columns" if t < T0 else
             "Global onset: rigid block drops" if z < 3 else
             "Crush-up from the base")
    ax.set_title(f"t = {t:4.2f} s   |   {phase}", fontsize=9, loc="left")
    ax.text(W / 2, H_TOT + 2.5, "street elevation (schematic, 18 storeys, physics-driven)",
            ha="center", fontsize=7, color="0.4")
    ax.text(-1, 0.6, "street-corner end", ha="right", fontsize=7, color="#d1495b")


def setup_plots(fig, gs):
    axz = fig.add_subplot(gs[0])
    axv = fig.add_subplot(gs[1], sharex=axz)
    axr = fig.add_subplot(gs[2], sharex=axz)
    segA, segB = raw[raw["segment"] == "A_centre_M"], raw[raw["segment"] == "B_roof"]
    zA0 = np.interp(segA["t_s"][0], tg, zg) - segA["z_m"][0]
    zB0 = np.interp(segB["t_s"][0], tg, zg) - segB["z_m"][0]
    axz.plot(tg, zg, color="#2e86ab", lw=1.6, label="crush-up model")
    axz.plot(segA["t_s"], segA["z_m"] + zA0, ".", color="#f18f01", ms=3, label="CV (video)")
    axz.plot(segB["t_s"], segB["z_m"] + zB0, ".", color="#f18f01", ms=3)
    axz.plot(tg, np.where(tg > T0, 0.5 * G * (tg - T0) ** 2, 0), ":", color="0.5", label="free fall")
    axz.set(ylim=(-1, 40), ylabel="descent z [m]")
    axz.legend(fontsize=6, loc="upper left")
    axv.plot(tg, vg, color="#2e86ab", lw=1.6)
    axv.plot(tg, np.where(tg > T0, G * (tg - T0), 0), ":", color="0.5")
    axv.set(ylim=(0, 20), ylabel="v [m/s]")
    axr.plot(tg, rhog, color="#a23b72", lw=1.6)
    axr.axhline(1, color="k", lw=0.6)
    axr.set(ylim=(0, 1.6), ylabel="resistance /\nweight", xlabel="video time [s]", xlim=(0, T_END))
    for a in (axz, axv, axr):
        a.axvspan(2.25, 3.75, color="0.92", zorder=0)
        a.tick_params(labelsize=7)
        a.yaxis.label.set_size(8)
    axr.xaxis.label.set_size(8)
    cursors = [a.axvline(0, color="#d1495b", lw=1) for a in (axz, axv, axr)]
    return cursors


def render(with_video, speed, path, fps_out=30, size=(13.5, 6.2)):
    cap = cv2.VideoCapture(str(VIDEO))
    vframes = []
    while with_video:
        ok, f = cap.read()
        if not ok:
            break
        vframes.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    vfps = 29.97
    ncols = 3 if with_video else 2
    fig = plt.figure(figsize=size if with_video else (10, 6.2), dpi=100)
    widths = [0.8, 1.2, 1.1] if with_video else [1.2, 1.1]
    outer = fig.add_gridspec(1, ncols, width_ratios=widths, wspace=0.08)
    axvid = fig.add_subplot(outer[0]) if with_video else None
    axb = fig.add_subplot(outer[-2])
    gs = outer[-1].subgridspec(3, 1, hspace=0.15)
    cursors = setup_plots(fig, gs)
    fig.suptitle("Eastleigh, Nairobi - 23 Oct 2026: reconstructed failure sequence "
                 "(CV + PINN/ODE + digital-twin seed; preliminary)", fontsize=10)
    footer(fig, SRC_VIDEO, SRC_CV, SRC_MODEL, SRC_TWIN, fontsize=7)
    nfr = int(T_END / speed * fps_out)
    writer = imageio.get_writer(path, fps=fps_out, **({"codec": "libx264", "quality": 7,
                                                        "macro_block_size": 1}
                                                       if path.suffix == ".mp4" else {"loop": 0}))
    for k in range(nfr):
        t = k * speed / fps_out
        draw_building(axb, t)
        for cur in cursors:
            cur.set_xdata([t, t])
        if axvid is not None:
            axvid.clear()
            axvid.axis("off")
            fi = min(int(round(t * vfps)), len(vframes) - 1)
            axvid.imshow(vframes[fi])
            axvid.set_title(f"eyewitness video  t = {fi / vfps:4.2f} s", fontsize=9, loc="left")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)
    writer.close()
    plt.close(fig)
    print("wrote", path)


def export_timeline():
    ts = np.round(np.arange(0, T_END + 1e-9, 1 / 60), 4)
    rows = []
    for t in ts:
        z, c, rubble, sag_x, failed = state(t)
        rows.append([float(t), round(z, 3), round(float(np.interp(t, tg, vg)), 3),
                     None if not np.isfinite(np.interp(t, tg, rhog)) else round(float(np.interp(t, tg, rhog)), 3),
                     round(sag_x[0], 3)])
    meta = dict(fields=["t", "z", "v", "rho", "sag_left"], N=N, h_typ=H_TYP, h_gnd=H_GND,
                lam=LAM, W=W, bays=BAYS, col_fail_t=COL_FAIL_T.tolist(), t0=T0, k=K, beta=BETA,
                rows=rows)
    json.dump(meta, open(OUT / "animation_timeline.json", "w"))
    print("wrote", OUT / "animation_timeline.json")


if __name__ == "__main__":
    export_timeline()
    render(True, 0.5, OUT / "collapse_reconstruction.mp4")
    render(False, 1.0, OUT / "collapse_reconstruction.gif", fps_out=15)
