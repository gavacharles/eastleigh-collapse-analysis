"""
Step 3 - Digital-twin seed: can a ground-storey defect at the street-corner end
cascade across the whole ground storey (the sequence seen in the video)?

A parametric 2-D RC plane frame (street elevation) with linear-elastic direct
stiffness analysis and a GSA-style alternate-path cascade:
  * remove the initiating element(s)
  * dynamic demand = before + DAF x (after - before), DAF = 2 (sudden loss)
  * any ground-storey column with DCR > 1 fails and is removed; repeat.
DCR = max(P/P0, M/Mcap(P)), Mcap from a tri-linear P-M envelope through
(P0, 0), (Pb = 0.4 P0, Mb), (0, Mpure); expected strengths, cracked stiffness
(0.7 Ig columns, 0.35 Ig beams, as ACI 318 / ASCE 41).

ALL GEOMETRY, SECTIONS AND LOADS BELOW ARE ASSUMPTIONS, not survey data. The
point is the *mechanism* and its sensitivity, not a number for this building.
Replace PARAMS with surveyed drawings/cores once available - that is what turns
this seed into an actual digital twin.
"""
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit import SRC_TWIN, footer

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

PARAMS = dict(
    bays=[4.5] * 5,                     # m, street frontage ~22 m (from video, approx.)
    h_ground=4.0, h_typ=3.0, storeys=18,
    trib=5.0,                           # m, tributary width into this frame
    dead=7.0, live=2.0, live_ground=4.0,  # kPa (slab+finishes+partitions+blockwork allowance)
    col_by_storey=[(1, 6, 0.50), (7, 12, 0.40), (13, 18, 0.30)],  # square columns, m
    beam=(0.30, 0.60),                  # b x h, m
    rho_s=0.015, fy=460.0,              # column steel ratio, MPa
    fc=30.0,                            # MPa (design intent)
    DAF=2.0,
)


def build(p):
    xs = np.r_[0, np.cumsum(p["bays"])]
    ys = np.r_[0, p["h_ground"] + p["h_typ"] * np.arange(p["storeys"])]
    nodes = [(x, y) for y in ys for x in xs]
    nx = len(xs)
    nid = lambda i, j: j * nx + i
    E = 4700 * np.sqrt(p["fc"]) * 1e3   # kPa
    els = []
    for j in range(p["storeys"]):
        a = next(s for lo, hi, s in p["col_by_storey"] if lo <= j + 1 <= hi)
        for i in range(nx):
            els.append(dict(n=(nid(i, j), nid(i, j + 1)), kind="col", storey=j + 1, bay=i,
                            A=a * a, I=0.7 * a ** 4 / 12, E=E, b=a, w=0.0))
    b, h = p["beam"]
    w_typ = (p["dead"] + 0.25 * p["live"]) * p["trib"]
    w_gnd = (p["dead"] + 0.25 * p["live_ground"]) * p["trib"]
    for j in range(1, p["storeys"] + 1):
        for i in range(nx - 1):
            els.append(dict(n=(nid(i, j), nid(i + 1, j)), kind="beam", storey=j, bay=i,
                            A=b * h, I=0.35 * b * h ** 3 / 12, E=E,
                            w=(w_gnd if j <= 2 else w_typ) + 0.3 * h * b * 24))
    supports = [nid(i, 0) for i in range(nx)]
    return np.array(nodes), els, supports, nx


def frame_solve(nodes, els, supports, settle=None):
    nd = len(nodes) * 3
    K = np.zeros((nd, nd))
    F = np.zeros(nd)
    cache = []
    for e in els:
        i, j = e["n"]
        (x1, y1), (x2, y2) = nodes[i], nodes[j]
        L = np.hypot(x2 - x1, y2 - y1)
        c, s = (x2 - x1) / L, (y2 - y1) / L
        EA, EI = e["E"] * e["A"], e["E"] * e["I"]
        kl = np.array([[EA / L, 0, 0, -EA / L, 0, 0],
                       [0, 12 * EI / L**3, 6 * EI / L**2, 0, -12 * EI / L**3, 6 * EI / L**2],
                       [0, 6 * EI / L**2, 4 * EI / L, 0, -6 * EI / L**2, 2 * EI / L],
                       [-EA / L, 0, 0, EA / L, 0, 0],
                       [0, -12 * EI / L**3, -6 * EI / L**2, 0, 12 * EI / L**3, -6 * EI / L**2],
                       [0, 6 * EI / L**2, 2 * EI / L, 0, -6 * EI / L**2, 4 * EI / L]])
        T = np.zeros((6, 6))
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T[:3, :3] = R
        T[3:, 3:] = R
        w = e["w"]     # gravity UDL (beams are horizontal -> local y = global y)
        fef = np.array([0, -w * L / 2, -w * L**2 / 12, 0, -w * L / 2, w * L**2 / 12])
        dofs = np.r_[3 * i:3 * i + 3, 3 * j:3 * j + 3]
        K[np.ix_(dofs, dofs)] += T.T @ kl @ T
        F[dofs] += T.T @ fef
        cache.append((dofs, T, kl, fef))
    # column self-weight as nodal loads
    for e in els:
        if e["kind"] == "col":
            i, j = e["n"]
            L = nodes[j][1] - nodes[i][1]
            F[3 * i + 1] -= e["A"] * L * 24 / 2
            F[3 * j + 1] -= e["A"] * L * 24 / 2
    fixed = np.r_[[np.r_[3 * n:3 * n + 3] for n in supports]].ravel()
    u = np.zeros(nd)
    if settle:
        for n, dy in settle.items():
            u[3 * n + 1] = dy
    free = np.setdiff1d(np.arange(nd), fixed)
    # drop dofs of nodes with no stiffness (orphaned after removals)
    diag = np.abs(np.diag(K))
    free = free[diag[free] > 1e-9]
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free] - K[np.ix_(free, fixed)] @ u[fixed])
    forces = []
    for dofs, T, kl, fef in cache:
        f = kl @ (T @ u[dofs]) - fef
        forces.append(f)       # local: N1,V1,M1,N2,V2,M2  (N1>0 = compression for col i->j up)
    return u, forces


def column_dcr(p, e, f):
    a = e["b"]
    fc, fy, rs = p["fc"] * 1e3, p["fy"] * 1e3, p["rho_s"]
    Ag = a * a
    P0 = 0.85 * fc * Ag * (1 - rs) + fy * rs * Ag
    Pb = 0.4 * P0
    Mb = 0.5 * rs * Ag * fy * 0.8 * a + 0.5 * Pb * (a / 2 - 0.4 * 0.45 * a)   # balanced moment
    Mpure = 0.5 * rs * Ag * fy * 0.85 * a                                     # pure bending
    P = max(f[0], 0.0)
    M = max(abs(f[2]), abs(f[5]))
    Mcap = Mb * (P0 - P) / (P0 - Pb) if P >= Pb else Mpure + (Mb - Mpure) * P / Pb
    if P >= P0:                     # axial crushing governs
        return P / P0, P, P0
    return max(P / P0, M / Mcap), P, P0


def analyse(p, removed=(), settle=None, weaken=None):
    nodes, els, sup, nx = build(p)
    weaken = weaken or {}
    for (st, bay), fac in weaken.items():
        for e in els:
            if e["kind"] == "col" and e["storey"] == st and e["bay"] == bay:
                e["b"] *= np.sqrt(fac)
                e["A"] *= fac
                e["I"] *= fac ** 2
    keep = [e for e in els if not (e["kind"] == "col" and (e["storey"], e["bay"]) in removed)]
    st_nodes = {e["n"][0] for e in keep if e["kind"] == "col" and e["storey"] == 1}
    sup2 = [s for s in sup if s in st_nodes]
    u, forces = frame_solve(nodes, keep, sup2, settle=settle)
    res = {}
    for e, f in zip(keep, forces):
        if e["kind"] == "col":
            d, P, P0 = column_dcr(p, e, f)
            res[(e["storey"], e["bay"])] = dict(dcr=d, P=P, P0=P0)
    return res, u, nodes, keep


def cascade(p, initial=(), settle=None, weaken=None, max_steps=10):
    removed = set(initial)
    base, *_ = analyse(p, (), settle, weaken)
    log = [dict(step=0, removed=sorted(removed),
                max_dcr_ground=max(v["dcr"] for k, v in base.items() if k[0] == 1))]
    prev = base
    for step in range(1, max_steps):
        cur, *_ = analyse(p, tuple(removed), settle, weaken)
        dyn = {k: prev.get(k, cur[k])["dcr"] + p["DAF"] * (cur[k]["dcr"] - prev.get(k, cur[k])["dcr"])
               for k in cur}
        fail = [k for k, d in dyn.items() if k[0] == 1 and d > 1.0 and k not in removed]
        log.append(dict(step=step, removed=sorted(removed),
                        dyn_dcr_ground={str(k[1]): round(v, 2) for k, v in dyn.items() if k[0] == 1},
                        max_dyn_dcr_storey2=round(max(v for k, v in dyn.items() if k[0] == 2), 2),
                        new_failures=sorted(fail)))
        if not fail:
            return log, False, cur, dyn
        removed |= set(fail)
        if sum(1 for k in removed if k[0] == 1) >= len(p["bays"]) + 1:
            return log, True, cur, dyn
        prev = cur
    return log, True, cur, dyn


SCEN = {
    "S0 design fc=30, intact":                     dict(fc=30, init=()),
    "S1 fc=30, corner col removed (alteration)":   dict(fc=30, init=((1, 0),)),
    "S2 fc=30, corner + 2nd col removed":          dict(fc=30, init=((1, 0), (1, 1))),
    "S3 fc=15 (poor concrete), intact":            dict(fc=15, init=()),
    "S4 fc=15, corner col removed":                dict(fc=15, init=((1, 0),)),
    "S5 fc=15, corner+2nd cols lose 50% section":  dict(fc=15, init=(), weaken={(1, 0): 0.5, (1, 1): 0.5}),
    "S6 fc=30, corner footing settles 50 mm":      dict(fc=30, init=(), settle_mm=50),
}


def main():
    summary = {}
    fig, axes = plt.subplots(1, len(SCEN), figsize=(3.0 * len(SCEN), 7.5), sharey=True)
    for ax, (name, sc) in zip(axes, SCEN.items()):
        p = deepcopy(PARAMS)
        p["fc"] = sc["fc"]
        nodes, els, sup, nx = build(p)
        settle = {sup[0]: -sc["settle_mm"] / 1000} if sc.get("settle_mm") else None
        log, collapsed, cur, dyn = cascade(p, sc["init"], settle, sc.get("weaken"))
        base, *_ = analyse(p, (), settle, sc.get("weaken"))
        ground = {b: round(base[(1, b)]["dcr"], 2) for b in range(nx)}
        W = sum(v["P"] for k, v in base.items() if k[0] == 1)
        cap = sum(v["P0"] for k, v in base.items() if k[0] == 1)
        summary[name] = dict(ground_dcr_intact=ground, capacity_over_weight=round(cap / W, 2),
                             global_collapse=collapsed, log=log)
        print(f"{name:45s} intact ground DCR max {max(ground.values()):.2f}  "
              f"sumP0/W={cap / W:.2f}  -> {'CASCADE -> full ground-storey loss' if collapsed else 'arrested'}"
              f"  steps={[l.get('new_failures', l['removed']) for l in log]}")
        # draw final state
        _, u, nodes2, keep = analyse(p, tuple(log[-1]["removed"]), settle, sc.get("weaken"))
        for e in keep:
            i, j = e["n"]
            (x1, y1), (x2, y2) = nodes2[i], nodes2[j]
            col = "0.6"
            if e["kind"] == "col" and e["storey"] <= 2:
                d = dyn.get((e["storey"], e["bay"]), 0)
                col = plt.cm.RdYlGn_r(min(d, 1.5) / 1.5)
            ax.plot([x1, x2], [y1, y2], color=col, lw=2.5 if e["kind"] == "col" else 1)
        failed = [tuple(k) for k in log[-1]["removed"]] + [tuple(k) for k in log[-1].get("new_failures", [])]
        for k in failed:
            x = np.r_[0, np.cumsum(p["bays"])][k[1]]
            ax.plot([x, x], [0, p["h_ground"]], "k:", lw=1)
            ax.plot(x, p["h_ground"] / 2, "kx", ms=9, mew=2)
        ax.set_title(name.replace(", ", ",\n", 1) + ("\n=> CASCADE" if collapsed else "\n=> holds"),
                     fontsize=8, color="tab:red" if collapsed else "tab:green")
        ax.set_aspect("equal")
        ax.set_xticks([])
    axes[0].set_ylabel("height [m]")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1.5))
    fig.colorbar(sm, ax=axes, shrink=0.5, label="ground/1st storey column dynamic DCR")
    fig.suptitle("Digital-twin seed: alternate-path cascade of ground-storey columns "
                 "(ASSUMED sections/loads - screening only)", fontsize=10)
    footer(fig, SRC_TWIN, "scenario logic per GSA alternate-path guidance")
    fig.savefig(OUT / "fig_twin_cascade.png", dpi=140, bbox_inches="tight")
    json.dump(summary, open(OUT / "twin_results.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
