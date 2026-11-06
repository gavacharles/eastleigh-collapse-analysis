"""
Step 20 - Engineering explainer figures for the write-up.

  V1  fig_reserve_map.png            axial load ratio P/(f'c Ag) of every column, design vs as-built
  V2  fig_sudden_loss.png            SDOF overshoot under sudden column loss + bridging mechanisms
  V4  fig_resistance_vs_weight.png   storey crushing force vs the weight of the falling block
  V5  fig_height_crosscheck.png      measured descent vs crush-up model for 12-18 storeys
  V7  fig_failure_envelope_map.png   FE-twin failure envelope as a map
  V8  fig_bayes_sensitivity.png      which assumed probabilities move the ranking
  V11 fig_debris_footprint.png       expected debris footprint and pile height on imagery
All geometry/sections in the twin are assumed (see 10_fe_twin.py).
"""
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from credit import SITE_LABEL, SRC_TWIN, footer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "explainers"
OUT.mkdir(parents=True, exist_ok=True)


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FE = load("fe", "10_fe_twin.py")
FER = json.load(open(HERE / "out" / "fe" / "fe_results.json"))
PHYS = json.load(open(HERE / "out" / "physics_results.json"))


# ------------------------------------------------------------------ V1 reserve map
def v1_reserve_map():
    np.seterr(all="ignore")
    out = {}
    for fc in (30, 15):
        p = deepcopy(FE.PARAMS); p["fc"] = fc
        fr = FE.Frame3D(p)
        _, res = fr.solve(())
        nx, ny, ns = fr.nx, fr.ny, p["storeys"]
        n = np.zeros((ns, nx * ny))
        for (storey, (i, j)), r in res.items():
            a = next(s for lo, hi, s in p["col_bands"] if lo <= storey <= hi)
            n[storey - 1, j * nx + i] = r["P"] / (fc * 1e3 * a * a)
        out[fc] = (n, res, nx, ny)
    fig = plt.figure(figsize=(15, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9], wspace=0.28)
    cmap = plt.get_cmap("RdYlGn_r")
    for k, fc in enumerate((30, 15)):
        ax = fig.add_subplot(gs[k])
        n = out[fc][0]
        im = ax.imshow(n, origin="lower", cmap=cmap, vmin=0, vmax=1.2, aspect="auto",
                       extent=(0.5, n.shape[1] + 0.5, 0.5, n.shape[0] + 0.5))
        ax.set(xlabel="column (grid order, corner = 1)", ylabel="storey", title=f"Axial load ratio P/(f'c Ag), f'c = {fc} MPa")
        ax.axhline(1.5, color="k", lw=0.8)
        ax.text(n.shape[1] + 0.3, 1, "L0", va="center", fontsize=8)
    cb = fig.colorbar(im, ax=fig.axes[:2], fraction=0.03, pad=0.02, location="right")
    cb.set_label("P / (f'c Ag)   (sound design ~0.3-0.4; >1 means steel carries the excess)")
    ax = fig.add_subplot(gs[2])
    n15, res15, nx, ny = out[15]
    vals = n15[0].reshape(ny, nx)
    im2 = ax.imshow(vals, cmap=cmap, vmin=0, vmax=1.2, origin="lower")
    for j in range(ny):
        for i in range(nx):
            r = res15[(1, (i, j))]
            ax.text(i, j, f"{vals[j, i]:.2f}\n{r['P0'] / r['P']:.1f}x", ha="center", va="center", fontsize=8,
                    color="k" if vals[j, i] < 0.8 else "w")
    ax.set_xticks(range(nx)); ax.set_yticks(range(ny))
    ax.set(xlabel="frontage (Ninth Street) ->", ylabel="depth", title="Ground storey, f'c 15 MPa: load ratio / reserve P0/P\n(blue box: street-corner column, least loaded)")
    ax.add_patch(mpatches.Rectangle((-0.5, -0.5), 1, 1, fill=False, ec="#1f6aa5", lw=3))

    footer(fig, SRC_TWIN, "gravity D + 0.25L, cracked stiffness; 18 storeys", fontsize=7.5)
    fig.savefig(OUT / "fig_reserve_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ V2 sudden loss
def v2_sudden_loss():
    T = FER["modal"]["T_local_vertical"]
    w = 2 * np.pi / T
    t = np.linspace(0, 2.5 * T, 600)
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    ax.plot(t, np.where(t > 0, 1.0, 0), color="0.5", ls=":", label="gradual (static) loading")
    for xi, col in ((0.0, "#c62f2f"), (0.05, "#1f6aa5")):
        wd = w * np.sqrt(1 - xi ** 2)
        u = 1 - np.exp(-xi * w * t) * (np.cos(wd * t) + xi / np.sqrt(1 - xi ** 2) * np.sin(wd * t))
        ax.plot(t, u, color=col, lw=2, label=f"sudden loss, damping {xi * 100:.0f} %  (peak {u.max():.2f}x)")
    ax.axhline(2, color="#c62f2f", lw=0.6, ls="--")
    ax.axvline(T / 2, color="0.4", lw=0.8)
    ax.text(T / 2 + 0.02, 2.05, f"peak at T/2 = {T / 2:.2f} s", fontsize=8)
    ax.set(xlabel="time after column loss [s]", ylabel="response / static response",
           title=f"Sudden column loss overshoots: up to 2x (local mode T = {T:.2f} s)")
    ax.set_ylim(0, 2.75)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    # schematics
    def frame(ax, kind):
        xs = np.array([0, 5.6, 11.2, 16.8]); levels = np.array([0, 4, 7, 10])
        sag = lambda x: 1.2 * np.clip(1 - x / 5.6, 0, 1)
        for zl in levels[1:]:
            if kind == "vierendeel":
                xx = np.linspace(0, 16.8, 100)
                ax.plot(xx, zl - sag(xx) - 0.25 * np.sin(np.pi * np.clip(xx / 5.6, 0, 1)) * (xx < 5.6), color="k", lw=2)
            else:
                xx = np.linspace(0, 16.8, 100)
                ax.plot(xx, zl - sag(xx) * 1.4, color="#c62f2f" if zl == levels[1] else "k", lw=2)
        for x in xs:
            if x == 0:
                ax.plot([x, x], [levels[1] - sag(0) * (1.4 if kind == "catenary" else 1), levels[-1] - sag(0) * (1.4 if kind == "catenary" else 1)],
                        color="k", lw=2)
                ax.plot([x, x], [0, levels[1] - 1.5], color="#c62f2f", lw=2, ls=":")
                ax.plot(x, 1.5, "x", color="#c62f2f", ms=14, mew=3)
            else:
                ax.plot([x, x], [0, levels[-1]], color="k", lw=2)
        ax.fill_between([-1, 18], -0.4, 0, color="0.7")
        ax.set_xlim(-1.5, 18); ax.set_ylim(-0.6, 11); ax.set_aspect("equal"); ax.axis("off")
    a1 = fig.add_subplot(gs[1]); frame(a1, "vierendeel")
    a1.set_title("Bridging by frame (Vierendeel) action\nbeams bend in double curvature over 2 spans", fontsize=10)
    a1.annotate("plastic hinges form at\nbeam ends (pushdown)", (5.6, 6.4), (8, 9.2), fontsize=8, arrowprops=dict(arrowstyle="->"))
    a2 = fig.add_subplot(gs[2]); frame(a2, "catenary")
    a2.set_title("Catenary action at large sag\nneeds continuous bottom bars + ties", fontsize=10)
    a2.annotate("beam becomes a tension tie;\nfails if bars are lapped\nat the column", (3, 2.1), (7, 0.9), fontsize=8, arrowprops=dict(arrowstyle="->"))
    footer(fig, "single-degree-of-freedom dynamics; local mode period from the FE twin (this analysis)")
    fig.savefig(OUT / "fig_sudden_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ V4 resistance vs weight
def v4_resistance():
    span = 0.75 * 3.0
    W = FER["crushing"]["W_kN"]
    N = 18
    fig, ax = plt.subplots(figsize=(11.5, 6))
    for fc, col in ((30, "#1f6aa5"), (15, "#c62f2f")):
        d, F = FE.column_crush(fc=fc)
        Fs = F * 20 / W
        m = d <= span
        ax.plot(d[m] * 1000, Fs[m], color=col, lw=2, label=f"storey crushing force, f'c {fc} MPa")
        mean = np.trapezoid(Fs[m], d[m]) / span
        ax.hlines(mean, 0, span * 1000, colors=col, linestyles="--", lw=1.4, label=f"mean over the crush, f'c {fc} MPa: {mean:.2f} W")
    for k, ls in ((0, "-"), (6, "-."), (12, ":")):
        wk = (N - k) / N
        ax.axhline(wk, color="k", lw=1.2, ls=ls)
        ax.text(span * 1000 * 1.01, wk, f"weight of falling block\nafter {k} storeys crushed", fontsize=8, va="center")
    r0 = PHYS["rho_free_early"][0]
    ax.axhspan(r0 - 0.03, r0 + 0.08, color="#c77700", alpha=0.25)
    ax.text(300, r0 + 0.12, "resistance measured from the video (0.38-0.46 W)", color="#8a5300", fontsize=9)
    ax.fill_between([0, span * 1000], 0, 1, color="#c62f2f", alpha=0.05)
    ax.text(300, 0.83, "mean crushing resistance < weight of the falling block  ->  it keeps accelerating", fontsize=10, color="#c62f2f")
    ax.set(xlim=(0, span * 1000), ylim=(0, 3.6), xlabel="shortening of the crushing storey [mm]",
           ylabel="force / weight of the full building above L0",
           title="Why the collapse could not stop: peak capacity is high, crushed capacity is not")
    ax.legend(fontsize=8.5, loc="upper right")
    footer(fig, "column model: Mander concrete + bar buckling (20 L0 columns, 600 mm square)", "measured resistance: crush-up/PINN fit to video",
           "twin sections assumed")
    fig.savefig(OUT / "fig_resistance_vs_weight.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ V5 height cross-check
def v5_height():
    G, LAM, H = 9.81, 0.25, 3.0
    raw = np.genfromtxt(HERE / "out" / "kinematics.csv", delimiter=",", names=True, dtype=None, encoding=None)
    rest = raw[raw["segment"] == "A_centre_L"]
    segs = [raw[raw["segment"] == s] for s in ("A_centre_M", "B_roof")]

    def sim(k, t0, t, N):
        def rhs(_, y):
            rho = k / max(N - y[0] / ((1 - LAM) * H), 0.5)
            return [y[1], G * (1 - rho) if (y[1] > 0 or rho < 1) else 0.0]
        t = np.asarray(t); z = np.zeros_like(t, float); m = t > t0
        if m.any():
            z[m] = solve_ivp(rhs, (t0, t[m].max() + 1e-6), [0.05, 0], t_eval=t[m], max_step=0.01).y[0]
        return z

    def resid(p, N):
        k, t0, cA, cB = p
        r = [(sim(k, t0, rest["t_s"], N) - rest["z_m"]) / rest["sigma_m"]]
        for d, c in zip(segs, (cA, cB)):
            r.append((sim(k, t0, d["t_s"], N) + c - d["z_m"]) / d["sigma_m"])
        return np.concatenate(r)

    fig, axs = plt.subplots(1, 4, figsize=(17, 4.8), sharey=True)
    tt = np.linspace(0, 5.6, 400)
    for ax, N in zip(axs, (12, 14, 16, 18)):
        b = min((least_squares(resid, [k0, 0.9, 0, -10], args=(N,), bounds=([0, 0.3, -5, -60], [40, 1.6, 5, 10]))
                 for k0 in (5, 8, 12)), key=lambda r: r.cost)
        k, t0, cA, cB = b.x
        chi = 2 * b.cost / (len(b.fun) - 4)
        ax.plot(tt, sim(k, t0, tt, N), color="#1f6aa5", lw=2, label="crush-up model")
        ax.plot(rest["t_s"], rest["z_m"], ".", color="k", ms=3)
        for d, c, col in zip(segs, (cA, cB), ("#c77700", "#c62f2f")):
            ax.plot(d["t_s"], d["z_m"] - c, ".", color=col, ms=4)
        ax.axvspan(2.25, 3.75, color="0.92", zorder=0)
        good = chi < 2
        ax.set_title(f"{N} storeys:  chi2/dof = {chi:.1f}", color="#2a9d8f" if good else "#c62f2f", fontsize=11)
        ax.set_xlabel("video time [s]")
        ax.set_ylim(-1, 40)
    axs[0].set_ylabel("descent of the building [m]")
    axs[0].legend(["crush-up model", "measured (CV)"], fontsize=8, loc="upper left")
    fig.suptitle("The motion constrains the height: 12 storeys cannot reproduce the measured descent; 16-18 can", fontsize=12)
    footer(fig, "eyewitness video, optical-flow tracking; crush-up model fitted per storey count (this analysis)")
    fig.savefig(OUT / "fig_height_crosscheck.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ V7 failure envelope map
def v7_envelope():
    env = FER["envelope"]
    scales = sorted({e["scale"] for e in env}); fcs = sorted({e["fc"] for e in env})
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    cmap = ListedColormap(["#2a9d8f", "#c62f2f", "#9a9a9a"])
    for ax, nl in zip(axs, (1, 2)):
        M = np.zeros((len(fcs), len(scales)))
        for e in env:
            if e["n_lost"] != nl:
                continue
            i, j = fcs.index(e["fc"]), scales.index(e["scale"])
            M[i, j] = 2 if e["max_dcr"] > 1 else 1 if e["collapsed"] else 0
            ax.text(j, i, f"{e['reserve']:.2f}", ha="center", va="center", fontsize=9, color="w")
        ax.imshow(M, cmap=cmap, vmin=0, vmax=2, origin="lower", aspect="auto")
        ax.set_xticks(range(len(scales))); ax.set_xticklabels([f"{s:.0%}" for s in scales])
        ax.set_yticks(range(len(fcs))); ax.set_yticklabels([f"{f:g}" for f in fcs])
        ax.set(xlabel="column size relative to design", title=f"{nl} corner-end column{'s' if nl > 1 else ''} lost at L0")
    axs[0].set_ylabel("concrete strength f'c [MPa]")
    handles = [mpatches.Patch(color="#2a9d8f", label="holds"),
               mpatches.Patch(color="#c62f2f", label="stands intact, then full L0 zipper (matches the video)"),
               mpatches.Patch(color="#9a9a9a", label="could not have stood even intact")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Failure envelope of the digital twin (cell labels: L0 capacity reserve, sum P0 / W)", fontsize=12)
    fig.subplots_adjust(bottom=0.22)
    footer(fig, SRC_TWIN, "3-D alternate-path cascade, DAF 2")
    fig.savefig(OUT / "fig_failure_envelope_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ V8 Bayesian sensitivity
def v8_bayes():
    B = load("bayes", "13_bayes_causes.py")
    base, _ = B.posterior(B.PRIOR, B.CPT)
    targets = ["A", "M", "X"]
    rows = []
    for src, d in (("prior", B.PRIOR), ("cpt", B.CPT)):
        for k, v in d.items():
            deltas = {}
            for f in (0.5, 1.5):
                dd = dict(d); dd[k] = min(max(v * f, 0.005), 0.995)
                post, _ = B.posterior(dd if src == "prior" else B.PRIOR, dd if src == "cpt" else B.CPT)
                deltas[f] = {t: post[t] - base[t] for t in targets}
            rows.append((f"{'prior ' if src == 'prior' else ''}{k}", deltas))
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.5))
    for ax, t in zip(axs, targets):
        rs = sorted(rows, key=lambda r: -max(abs(r[1][0.5][t]), abs(r[1][1.5][t])))[:12][::-1]
        y = np.arange(len(rs))
        ax.barh(y, [r[1][0.5][t] for r in rs], color="#1f6aa5", label="parameter x0.5")
        ax.barh(y, [r[1][1.5][t] for r in rs], color="#c77700", alpha=0.85, label="parameter x1.5")
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rs], fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set(xlabel="change in posterior", title=f"{B.SHORT[t]}  (base {base[t]:.2f})")
    axs[0].legend(fontsize=8)
    fig.suptitle("Bayesian sensitivity: which assumed probabilities move each leading cause (one at a time, x0.5 / x1.5)", fontsize=12)
    footer(fig, "Bayesian network with expert-elicited probabilities (13_bayes_causes.py)")
    fig.savefig(OUT / "fig_bayes_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ V11 debris footprint
def v11_debris():
    import scene3d as S
    img, b = S.esri_mosaic(0.0012, 19)
    (x0, x1), (y1, y0) = S.local(np.array([b[0], b[2]]), np.array([b[1], b[3]]))
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 0.8], wspace=0.15, hspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, extent=(x0, x1, y0, y1))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(100 * np.cos(th), 100 * np.sin(th), color="w", ls="--", lw=1.5, label="plot lies within ~100 m (no public parcel data)")
    ax.plot(0, 0, marker="+", color="#ff3b30", ms=18, mew=3, label="reference: Ninth St x 2nd Ave Eastleigh")
    ax.set_xlim(-125, 125); ax.set_ylim(-125, 125); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title(f"Location: {SITE_LABEL}", fontsize=10)
    # plan schematic
    W, D, m = 22.5, 20.0, 8.0
    a2 = fig.add_subplot(gs[0, 1])
    a2.add_patch(mpatches.Rectangle((-W / 2 - m, -D / 2 - m), W + 2 * m, D + 2 * m, color="#f0a43a", alpha=0.35, label="debris spill zone (+8 m)"))
    a2.add_patch(mpatches.Rectangle((-W / 2, -D / 2), W, D, color="#e5383b", alpha=0.45, label="building footprint (generic 22.5 x 20 m)"))
    a2.add_patch(mpatches.Rectangle((-3, -2.5), 6, 5, color="#5a3e2b", alpha=0.8, label="possible core remnant / void zone"))
    a2.plot([-W / 2 - m - 6, W / 2 + m + 6], [-D / 2 - m - 3] * 2, color="0.5", lw=6, alpha=0.6)
    a2.text(0, -D / 2 - m - 5.5, "street frontage", ha="center", fontsize=8, color="0.3")
    a2.set_xlim(-30, 30); a2.set_ylim(-26, 22); a2.set_aspect("equal")
    a2.set_xlabel("m"); a2.set_title("Plan (to scale): expected debris footprint", fontsize=10)
    a2.legend(fontsize=7.5, loc="upper right")
    # section of the pile
    a3 = fig.add_subplot(gs[1, 1])
    xs = np.linspace(-W / 2 - m, W / 2 + m, 200)
    for H, col, lab in ((4 + 15 * 3, "#1f6aa5", "16 storeys"), (4 + 17 * 3, "#c62f2f", "18 storeys")):
        for lam, ls in ((0.25, "-"), (0.35, "--")):
            h = lam * H
            prof = h * np.clip(1 - np.maximum(np.abs(xs) - W / 2 + 4, 0) / (m + 4), 0, 1)
            a3.plot(xs, prof, color=col, ls=ls, label=f"{lab}, compaction {lam:.0%}: {h:.0f} m")
    a3.axvspan(-W / 2, W / 2, color="#e5383b", alpha=0.08)
    a3.set(xlabel="m across the frontage", ylabel="pile height [m]", title="Section: expected pile height (crush-up model)")
    a3.legend(fontsize=7.5, ncol=2, loc="upper right")
    a3.set_ylim(0, 24)
    fig.text(0.53, 0.015, "Post-event imagery: none in the public archive yet (latest Wayback acquisition 16 Jan 2026). "
             "Vertical, in-footprint collapse per the video.", fontsize=8.5, color="0.3")
    footer(fig, "Esri World Imagery (pre-event, Jan 2026)", "pile height from crush-up model; footprint generic")
    fig.savefig(OUT / "fig_debris_footprint.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (v1_reserve_map, v2_sudden_loss, v4_resistance, v5_height, v7_envelope, v8_bayes, v11_debris):
        f(); print("done", f.__name__)
