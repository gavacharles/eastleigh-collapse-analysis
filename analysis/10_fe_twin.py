"""
Step 10 - Finite-element digital twin (parametric 3-D RC frame) and advanced mechanics.

  1. 3-D linear-elastic frame FE (12-dof beam-columns, sparse direct solver),
     gravity D + 0.25L, cracked stiffness; column DCR from axial + biaxial bending
  2. Modal analysis (lumped mass): global periods, and the local vertical mode
     of the corner bay after column loss (sets the time scale of the zipper)
  3. Alternate-path cascade in 3-D (GSA-style, DAF = 2) for each candidate cause
  4. Nonlinear pushdown of the frontage frame over the lost corner column
     (event-to-event plastic hinges in beams) + energy-based dynamic capacity
     (Izzuddin et al. 2008)
  5. Axial crushing of an L0 column (Mander unconfined concrete + bar buckling):
     peak vs mean crushing force, compared with the resistance measured from video

ALL GEOMETRY, SECTIONS, REINFORCEMENT AND LOADS ARE ASSUMED (no drawings yet).
The twin shows mechanisms and sensitivities; replace PARAMS with surveyed data.
"""
import json
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from credit import SRC_TWIN, footer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "fe"
OUT.mkdir(parents=True, exist_ok=True)
G = 9.81

PARAMS = dict(
    bays_x=[5.625] * 4,          # frontage along Ninth Street (22.5 m), typical local spans
    bays_y=[6.667] * 3,          # depth (20 m)
    h_ground=4.0, h_typ=3.0, storeys=18,
    col_bands=[(1, 6, 0.60), (7, 12, 0.50), (13, 18, 0.40)],   # square columns [m]
    beam=(0.30, 0.60),           # b x h [m]
    slab_t=0.20,
    dead_extra=2.7,              # kPa finishes + blockwork partitions
    live=2.0, live_ground=4.0,   # kPa
    fc=30.0, fy=460.0, rho_col=0.015,
    DAF=2.0,
)

SCENARIOS = {
    "S1 design f'c 30, corner column lost":           dict(fc=30, remove=[(0, 0)]),
    "S2 as-built f'c 15, corner column lost":         dict(fc=15, remove=[(0, 0)]),
    "S3 design f'c 30, two corner-end columns lost":  dict(fc=30, remove=[(0, 0), (1, 0)]),
    "S4 f'c 20, corner + next column lose 50% area":  dict(fc=20, remove=[], weaken={(0, 0): 0.5, (1, 0): 0.5}),
    "S5 designed for 12 storeys, built 18, corner lost": dict(fc=25, remove=[(0, 0)], designed_for=12),
    "S6 as-built f'c 15, two corner-end columns lost": dict(fc=15, remove=[(0, 0), (1, 0)]),
}


def sweep(p):
    """Failure envelope: which (concrete strength, column size, initiating damage) reproduce
    the observed behaviour (corner loss -> full L0 cascade)?"""
    out = []
    for n_lost, init in ((1, [(0, 0)]), (2, [(0, 0), (1, 0)])):
        for scale in (0.8, 0.9, 1.0, 1.1):
            for fc in (10, 12.5, 15, 20, 25, 30):
                q = deepcopy(p); q["fc"] = fc
                q["col_bands"] = [(lo, hi, a * scale) for lo, hi, a in p["col_bands"]]
                r, _ = cascade(q, init)
                out.append(dict(n_lost=n_lost, scale=scale, fc=fc, collapsed=r["collapsed"],
                                reserve=r["sumP0_over_W"], max_dcr=r["max_dcr_intact"]))
    return out


# ======================================================================== model
class Frame3D:
    def __init__(self, p, weaken=None, designed_for=None):
        self.p = p
        xs = np.r_[0, np.cumsum(p["bays_x"])]
        ys = np.r_[0, np.cumsum(p["bays_y"])]
        zs = np.r_[0, p["h_ground"] + p["h_typ"] * np.arange(p["storeys"])]
        self.xs, self.ys, self.zs = xs, ys, zs
        self.nx, self.ny, self.nz = len(xs), len(ys), len(zs)
        self.nodes = np.array([(x, y, z) for z in zs for y in ys for x in xs])
        E = 4700 * np.sqrt(p["fc"]) * 1e3        # kPa
        Gm = E / 2.4
        self.E = E
        els = []
        scale = 1.0
        if designed_for:                          # columns sized for fewer storeys
            scale = np.sqrt(designed_for / p["storeys"])
        weaken = weaken or {}
        for k in range(p["storeys"]):
            a = next(s for lo, hi, s in p["col_bands"] if lo <= k + 1 <= hi) * scale
            for j in range(self.ny):
                for i in range(self.nx):
                    f = weaken.get((i, j), 1.0) if k == 0 else 1.0
                    A = a * a * f
                    I = 0.7 * (a * a) ** 2 / 12 * f ** 2
                    els.append(dict(n=(self.nid(i, j, k), self.nid(i, j, k + 1)), kind="col", storey=k + 1,
                                    ij=(i, j), A=A, Iy=I, Iz=I, J=2.25 * (a * np.sqrt(f)) ** 4 * 0.141,
                                    E=E, G=Gm, b=a * np.sqrt(f), w=0.0))
        b, h = p["beam"]
        q_slab = 25 * p["slab_t"] + p["dead_extra"]
        for k in range(1, p["storeys"] + 1):
            live = p["live_ground"] if k <= 2 else p["live"]
            q = q_slab + 0.25 * live                     # kPa, collapse combination
            for j in range(self.ny):
                for i in range(self.nx - 1):            # beams along x
                    trib = (p["bays_y"][j - 1] / 2 if j > 0 else 0) + (p["bays_y"][j] / 2 if j < self.ny - 1 else 0)
                    els.append(self._beam(self.nid(i, j, k), self.nid(i + 1, j, k), b, h, E, Gm,
                                          q * trib / 2 + 25 * b * (h - p["slab_t"]), k, (i, j, "x")))
            for i in range(self.nx):
                for j in range(self.ny - 1):            # beams along y
                    trib = (p["bays_x"][i - 1] / 2 if i > 0 else 0) + (p["bays_x"][i] / 2 if i < self.nx - 1 else 0)
                    els.append(self._beam(self.nid(i, j, k), self.nid(i, j + 1, k), b, h, E, Gm,
                                          q * trib / 2 + 25 * b * (h - p["slab_t"]), k, (i, j, "y")))
        self.els = els
        self.supports = [self.nid(i, j, 0) for j in range(self.ny) for i in range(self.nx)]

    def nid(self, i, j, k):
        return (k * self.ny + j) * self.nx + i

    @staticmethod
    def _beam(n1, n2, b, h, E, Gm, w, k, tag):
        return dict(n=(n1, n2), kind="beam", storey=k, ij=tag, A=b * h, Iy=0.35 * b * h ** 3 / 12,
                    Iz=0.35 * h * b ** 3 / 12, J=0.2 * b ** 3 * h, E=E, G=Gm, b=b, w=w)

    # ---------------------------------------------------------------- element
    def _elem(self, e):
        i, j = e["n"]
        d = self.nodes[j] - self.nodes[i]
        L = np.linalg.norm(d)
        ex = d / L
        ref = np.array([1.0, 0, 0]) if abs(ex[2]) > 0.99 else np.array([0, 0, 1.0])
        ey = np.cross(ref, ex); ey /= np.linalg.norm(ey)
        ez = np.cross(ex, ey)
        R = np.vstack([ex, ey, ez])
        T = np.kron(np.eye(4), R)
        E, Gm, A, Iy, Iz, J = e["E"], e["G"], e["A"], e["Iy"], e["Iz"], e["J"]
        k = np.zeros((12, 12))
        k[0, 0] = k[6, 6] = E * A / L; k[0, 6] = k[6, 0] = -E * A / L
        k[3, 3] = k[9, 9] = Gm * J / L; k[3, 9] = k[9, 3] = -Gm * J / L
        a, b_, c, d_ = 12 * E * Iz / L**3, 6 * E * Iz / L**2, 4 * E * Iz / L, 2 * E * Iz / L
        for (r, s_, v) in [(1, 1, a), (1, 5, b_), (1, 7, -a), (1, 11, b_), (5, 5, c), (5, 7, -b_), (5, 11, d_),
                           (7, 7, a), (7, 11, -b_), (11, 11, c)]:
            k[r, s_] = k[s_, r] = v
        a, b_, c, d_ = 12 * E * Iy / L**3, 6 * E * Iy / L**2, 4 * E * Iy / L, 2 * E * Iy / L
        for (r, s_, v) in [(2, 2, a), (2, 4, -b_), (2, 8, -a), (2, 10, -b_), (4, 4, c), (4, 8, b_), (4, 10, d_),
                           (8, 8, a), (8, 10, b_), (10, 10, c)]:
            k[r, s_] = k[s_, r] = v
        # consistent load of a downward UDL w (global -Z) on a horizontal member: local z = global Z
        f = np.zeros(12)
        if e["w"]:
            w = e["w"]
            f[2] = f[8] = -w * L / 2
            f[4], f[10] = w * L**2 / 12, -w * L**2 / 12
        return T, k, f, L

    # ---------------------------------------------------------------- solve
    def solve(self, removed=(), settle=None):
        nn = len(self.nodes)
        nd = 6 * nn
        rows, cols, vals = [], [], []
        F = np.zeros(nd)
        cache = []
        keep = [e for e in self.els if not (e["kind"] == "col" and e["storey"] == 1 and e["ij"] in removed)]
        for e in keep:
            T, k, f, L = self._elem(e)
            dofs = np.r_[6 * e["n"][0]:6 * e["n"][0] + 6, 6 * e["n"][1]:6 * e["n"][1] + 6]
            kg = T.T @ k @ T
            r, c = np.meshgrid(dofs, dofs, indexing="ij")
            rows.append(r.ravel()); cols.append(c.ravel()); vals.append(kg.ravel())
            F[dofs] += T.T @ f
            if e["kind"] == "col":                       # self-weight lumped to ends
                wcol = e["A"] * L * 25
                F[6 * e["n"][0] + 2] -= wcol / 2
                F[6 * e["n"][1] + 2] -= wcol / 2
            cache.append((e, dofs, T, k, f))
        K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nd, nd))
        sup = [s for s in self.supports
               if not any(s == self.nid(i, j, 0) for (i, j) in removed)]
        fixed = np.r_[[np.arange(6 * s, 6 * s + 6) for s in sup]].ravel()
        u = np.zeros(nd)
        if settle:
            for n, dz in settle.items():
                u[6 * n + 2] = dz
        free = np.setdiff1d(np.arange(nd), fixed)
        diag = K.diagonal()
        free = free[diag[free] > 1e-9]
        Kff = K[free][:, free]
        rhs = F[free] - K[free][:, fixed] @ u[fixed]
        u[free] = spla.spsolve(Kff.tocsc(), rhs)
        self.K, self.F, self.free, self.u = K, F, free, u
        res = {}
        for e, dofs, T, k, f in cache:
            fl = k @ (T @ u[dofs]) - f
            if e["kind"] == "col":
                res[(e["storey"], e["ij"])] = self.col_check(e, fl)
        return u, res

    def col_check(self, e, fl):
        p = self.p
        a = e["b"]
        fc, fy, rs = p["fc"] * 1e3, p["fy"] * 1e3, p["rho_col"]
        Ag = e["A"]
        P0 = 0.85 * fc * Ag * (1 - rs) + fy * rs * Ag
        Pb = 0.4 * P0
        Mb = 0.5 * rs * Ag * fy * 0.8 * a + 0.5 * Pb * (a / 2 - 0.18 * a)
        Mpure = 0.5 * rs * Ag * fy * 0.85 * a
        P = max(fl[0], 0.0)
        M = max(np.hypot(fl[4], fl[5]), np.hypot(fl[10], fl[11]))
        if P >= P0:
            dcr = P / P0
        else:
            Mcap = Mb * (P0 - P) / (P0 - Pb) if P >= Pb else Mpure + (Mb - Mpure) * P / Pb
            dcr = max(P / P0, M / Mcap)
        return dict(dcr=float(dcr), P=float(P), P0=float(P0), M=float(M))

    # ---------------------------------------------------------------- modal
    def modal(self, removed=(), nmodes=12):
        self.solve(removed)
        K, free = self.K, self.free
        m = np.zeros(K.shape[0])
        loads = -self.F.copy()
        for n in range(len(self.nodes)):
            mz = max(loads[6 * n + 2], 0) / G               # t (kN / g)
            m[6 * n:6 * n + 3] = mz
            m[6 * n + 3:6 * n + 6] = mz * 1e-3              # small rotary inertia
        Kf = K[free][:, free].tocsc()
        Mf = sp.diags(m[free] + 1e-9)
        w2, phi = spla.eigsh(Kf, k=nmodes, M=Mf, sigma=0, which="LM")
        order = np.argsort(w2)
        w2, phi = w2[order], phi[:, order]
        full = np.zeros((K.shape[0], nmodes))
        full[free] = phi
        return 2 * np.pi / np.sqrt(np.abs(w2)), full


# ======================================================================== analyses
def cascade(p, init, weaken=None, designed_for=None, max_steps=8):
    fr = Frame3D(p, weaken, designed_for)
    u0, base = fr.solve(())
    removed = list(init)
    steps = [dict(step=0, removed=list(removed),
                  dcr={str(k[1]): round(v["dcr"], 3) for k, v in base.items() if k[0] == 1})]
    prev = base
    collapsed = False
    disp = {}
    for s in range(1, max_steps):
        u, cur = fr.solve(tuple(removed))
        if s == 1:
            disp["after_initial"] = u.reshape(-1, 6)[:, :3].tolist()
        dyn = {k: prev.get(k, cur[k])["dcr"] + p["DAF"] * (cur[k]["dcr"] - prev.get(k, cur[k])["dcr"])
               for k in cur}
        fail = sorted(k[1] for k, d in dyn.items() if k[0] == 1 and d > 1.0 and k[1] not in removed)
        steps.append(dict(step=s, removed=list(removed), new_failures=fail,
                          dcr={str(k[1]): round(v, 3) for k, v in dyn.items() if k[0] == 1},
                          max_dcr_L1=round(max(v for k, v in dyn.items() if k[0] == 2), 3)))
        if not fail:
            break
        removed += fail
        if len(removed) >= fr.nx * fr.ny - 2:
            collapsed = True
            break
        prev = cur
    sumP0 = sum(v["P0"] for k, v in base.items() if k[0] == 1)
    W = sum(v["P"] for k, v in base.items() if k[0] == 1)
    return dict(collapsed=collapsed, steps=steps, sumP0_over_W=sumP0 / W, W_kN=W,
                max_dcr_intact=max(v["dcr"] for k, v in base.items() if k[0] == 1),
                u_intact=u0.reshape(-1, 6)[:, :3].tolist(), **disp), fr


# ------------------------------------------------------------------ 2-D pushdown with plastic hinges
def pushdown(beam_top_As, beam_bot_As, fy=460e3, theta_u=0.02, trib=2.5, nsub=4, p=PARAMS):
    """Frontage frame (2-D) with corner L0 column removed. Beams subdivided; hinges at
    any beam node when |M| reaches Mp (hogging uses top steel, sagging bottom steel).
    Load: (D + 0.25L) everywhere at lambda = 1 on the bays over the removed column,
    others fixed at 1. Returns lambda vs vertical displacement of the corner node."""
    bx = p["bays_x"]
    xs = np.r_[0, np.cumsum(bx)]
    zs = np.r_[0, p["h_ground"] + p["h_typ"] * np.arange(p["storeys"])]
    E = 4700 * np.sqrt(p["fc"]) * 1e3
    b, h = p["beam"]
    d = h - 0.05
    Mp_neg, Mp_pos = beam_top_As * fy * 0.9 * d, beam_bot_As * fy * 0.9 * d
    q = (25 * p["slab_t"] + p["dead_extra"] + 0.25 * p["live"]) * trib + 25 * b * (h - p["slab_t"])
    # nodes: column lines at every level + internal beam nodes
    nodes, nid = [], {}
    def add(x, z):
        key = (round(x, 4), round(z, 4))
        if key not in nid:
            nid[key] = len(nodes); nodes.append((x, z))
        return nid[key]
    els = []
    for k in range(len(zs) - 1):
        a = next(s for lo, hi, s in p["col_bands"] if lo <= k + 1 <= hi)
        for i, x in enumerate(xs):
            if k == 0 and i == 0:
                continue                                 # removed corner column
            els.append(dict(n=(add(x, zs[k]), add(x, zs[k + 1])), A=a * a, I=0.7 * a**4 / 12, beam=False))
    loads = {}
    for k in range(1, len(zs)):
        for i in range(len(xs) - 1):
            pts = [add(xs[i] + bx[i] * s / nsub, zs[k]) for s in range(nsub + 1)]
            for a_, b2 in zip(pts[:-1], pts[1:]):
                els.append(dict(n=(a_, b2), A=b * h, I=0.35 * b * h**3 / 12, beam=True, bay=i))
            for s, n in enumerate(pts):
                wgt = 0.5 if s in (0, nsub) else 1.0
                loads.setdefault(n, [0.0, 0.0])
                loads[n][0 if i == 0 else 1] += q * bx[i] / nsub * wgt
    nodes = np.array(nodes)
    nn = len(nodes)
    base = [nid[(round(x, 4), 0.0)] for x in xs[1:]]
    corner = nid[(0.0, round(zs[1], 4))]
    for e in els:
        e["rel"] = [False, False]
    P_var = np.zeros(3 * nn); P_fix = np.zeros(3 * nn)
    for n, (lv, lf) in loads.items():
        P_var[3 * n + 1] -= lv; P_fix[3 * n + 1] -= lf

    def kmat(e):
        (x1, z1), (x2, z2) = nodes[e["n"][0]], nodes[e["n"][1]]
        L = np.hypot(x2 - x1, z2 - z1); c, s = (x2 - x1) / L, (z2 - z1) / L
        EA, EI = E * e["A"], E * e["I"]
        k = np.array([[EA / L, 0, 0, -EA / L, 0, 0],
                      [0, 12 * EI / L**3, 6 * EI / L**2, 0, -12 * EI / L**3, 6 * EI / L**2],
                      [0, 6 * EI / L**2, 4 * EI / L, 0, -6 * EI / L**2, 2 * EI / L],
                      [-EA / L, 0, 0, EA / L, 0, 0],
                      [0, -12 * EI / L**3, -6 * EI / L**2, 0, 12 * EI / L**3, -6 * EI / L**2],
                      [0, 6 * EI / L**2, 2 * EI / L, 0, -6 * EI / L**2, 4 * EI / L]])
        rel = [2 * 0 + 2 if e["rel"][0] else None, 5 if e["rel"][1] else None]
        rel = [r for r in rel if r is not None]
        if rel:                                          # static condensation of released rotations
            keep = [i for i in range(6) if i not in rel]
            kcc = k[np.ix_(rel, rel)]; kcr = k[np.ix_(rel, keep)]
            kred = k[np.ix_(keep, keep)] - kcr.T @ np.linalg.solve(kcc, kcr)
            k2 = np.zeros((6, 6)); k2[np.ix_(keep, keep)] = kred
            k = k2
        T = np.zeros((6, 6)); R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]); T[:3, :3] = R; T[3:, 3:] = R
        return T, k, L

    def solve(P):
        K = np.zeros((3 * nn, 3 * nn)); store = []
        for e in els:
            T, k, L = kmat(e)
            dofs = np.r_[3 * e["n"][0]:3 * e["n"][0] + 3, 3 * e["n"][1]:3 * e["n"][1] + 3]
            K[np.ix_(dofs, dofs)] += T.T @ k @ T
            store.append((e, dofs, T, k))
        fixed = np.r_[[np.arange(3 * n, 3 * n + 3) for n in base]].ravel()
        free = np.setdiff1d(np.arange(3 * nn), fixed)
        free = free[np.abs(np.diag(K)[free]) > 1e-9]
        Kf = K[np.ix_(free, free)]
        if np.linalg.cond(Kf) > 1e13:
            return None, None
        u = np.zeros(3 * nn); u[free] = np.linalg.solve(Kf, P[free])
        M = [(e, (k @ (T @ u[dofs]))[[2, 5]]) for e, dofs, T, k in store]
        return u, M

    # initial state: fixed part + lambda = 0 on variable part, then events
    u_acc = np.zeros(3 * nn)
    M_acc = {id(e): np.zeros(2) for e in els}
    lam, curve, hinges = 0.0, [(0.0, 0.0)], []
    loads_now = P_fix.copy()
    first = True
    for step in range(200):
        P = loads_now if first else P_var
        u, M = solve(P)
        if u is None:
            break
        # moments are increments (first step: full fixed load); find scale to next hinge
        scale = 1.0 if first else np.inf
        cands = []
        for e, m in M:
            if not e["beam"]:
                continue
            for end in (0, 1):
                if e["rel"][end]:
                    continue
                cur = M_acc[id(e)][end]
                # sign: end moment at node j acts opposite; hogging at supports = negative bending
                for mi, cap in ((m[end], None),):
                    tot_sign = np.sign(cur + mi) if (cur + mi) != 0 else 1
                    Mp = Mp_neg if (tot_sign * (1 if end == 0 else -1)) > 0 else Mp_pos
                    if first:
                        continue
                    if abs(mi) < 1e-9:
                        continue
                    s_ = (np.sign(mi) * Mp - cur) / mi if np.sign(mi) * Mp != 0 else np.inf
                    if s_ > 1e-9:
                        cands.append((s_, e, end))
        if not first:
            if not cands:
                break
            scale, e_h, end_h = min(cands, key=lambda c: c[0])
        u_acc += scale * u
        for e, m in M:
            M_acc[id(e)] += scale * m
        if not first:
            lam += scale
            e_h["rel"][end_h] = True
            hinges.append(dict(lam=lam, x=float(nodes[e_h["n"][end_h]][0]), z=float(nodes[e_h["n"][end_h]][1])))
        curve.append((lam, -u_acc[3 * corner + 1]))
        first = False
        if -u_acc[3 * corner + 1] > 1.0 or lam > 6:
            break
    curve = np.array(curve)
    # mechanism: displacement grows at constant lambda -> extend to 1.0 m at last lambda
    if curve[-1, 1] < 1.0:
        curve = np.vstack([curve, [curve[-1, 0], 1.0]])
    uu = curve[:, 1]
    lam_s = curve[:, 0]
    # energy-based dynamic capacity (Izzuddin 2008): lambda_d(u) = (1/u) * integral lambda_s du
    u_f = np.linspace(1e-4, uu.max(), 400)
    ls = np.interp(u_f, uu, lam_s)
    ld = np.cumsum(ls * np.gradient(u_f)) / u_f
    # limit by plastic rotation: approx theta_p ~ u / span_at_corner
    u_lim = theta_u * 2 * bx[0]
    return dict(Mp_neg=Mp_neg, Mp_pos=Mp_pos, curve=curve.tolist(), u=u_f.tolist(), lam_static=ls.tolist(),
                lam_dyn=ld.tolist(), lam_dyn_at_ulim=float(np.interp(u_lim, u_f, ld)), u_lim=u_lim,
                lam_static_max=float(lam_s.max()), n_hinges=len(hinges), hinges=hinges[:40])


# ------------------------------------------------------------------ column crushing mechanics
def column_crush(a=0.60, fc=30.0, rho=0.015, fy=460.0, L=4.0, s_ties=0.30):
    """Axial load - shortening of an L0 column. Mander unconfined concrete (cover + weakly
    confined core); bars yield then buckle once the cover spalls; post-peak softening
    from localisation over a crush zone ~ 1.5 a. Returns force [kN] vs shortening [m]."""
    Ec = 4700 * np.sqrt(fc)
    eps_c0 = 0.002
    eps = np.linspace(0, 0.5, 3000)                   # average strain over the column
    # local strain in crush zone after localisation
    lz = min(1.5 * a, L)
    def mander(e, fcc, ecc):
        r = Ec / (Ec - fcc / ecc)
        x = e / ecc
        return fcc * x * r / (r - 1 + x ** r)
    Ag = a * a
    As = rho * Ag
    # weak confinement from widely spaced ties: ke small
    fl = 0.02 * fc
    fcc = fc * (1 + 2.0 * fl / fc)
    ecc = eps_c0 * (1 + 5 * (fcc / fc - 1))
    core = mander(eps, fcc, ecc)
    core = np.where(eps > 0.006, core * np.exp(-(eps - 0.006) / 0.03) + 0.15 * fc, core)   # crushing + rubble friction
    cover_frac = 1 - ((a - 0.08) / a) ** 2
    cover = np.where(eps < 0.004, mander(np.minimum(eps, 0.004), fc, eps_c0), 0.0)
    steel = np.minimum(eps * 200000, fy)
    steel = np.where(eps > 0.004, fy * np.maximum(0.2, np.exp(-(eps - 0.004) / 0.01)), steel)  # buckling
    sigma_c = core * (1 - cover_frac) + cover * cover_frac
    F = (sigma_c * Ag * (1 - rho) + steel * As) * 1e3        # MPa*m2*1e3 = kN
    delta = np.where(eps < 0.004, eps * L, 0.004 * L + (eps - 0.004) * lz * 4)   # localisation then progressive
    return delta, F


# ======================================================================== main
def main():
    np.seterr(all="ignore")          # macOS Accelerate emits spurious matmul warnings
    p = deepcopy(PARAMS)
    results = {}
    # ---- modal (design model, intact and after corner loss)
    fr = Frame3D(p)
    T_int, phi = fr.modal(())
    T_dmg, phi_d = fr.modal(((0, 0),), nmodes=30)
    node = fr.nid(0, 0, 1)
    vert = [abs(phi_d[6 * node + 2, m]) / (np.abs(phi_d[2::6, m]).max() + 1e-12) for m in range(30)]
    mloc = int(np.argmax([v * (1 if T_dmg[m] < 1.0 else 0) for m, v in enumerate(vert)]))
    results["modal"] = dict(T_intact=T_int[:6].round(3).tolist(), T_local_vertical=round(float(T_dmg[mloc]), 3),
                            zipper_stage_time=round(float(T_dmg[mloc]) / 4, 3))
    print("periods intact [s]:", results["modal"]["T_intact"], " local vertical mode after corner loss:",
          results["modal"]["T_local_vertical"], "s -> quarter period", results["modal"]["zipper_stage_time"], "s")

    # ---- cascades
    results["scenarios"] = {}
    anim = {}
    for name, sc in SCENARIOS.items():
        q = deepcopy(p); q["fc"] = sc["fc"]
        r, frs = cascade(q, sc["remove"], sc.get("weaken"), sc.get("designed_for"))
        seq = [s.get("new_failures", s["removed"]) for s in r["steps"]]
        print(f"{name:50s} sumP0/W={r['sumP0_over_W']:.2f}  max DCR intact={r['max_dcr_intact']:.2f}  "
              f"-> {'CASCADE' if r['collapsed'] else 'holds'}  seq={seq}")
        anim[name] = dict(steps=r["steps"], collapsed=r["collapsed"],
                          u_after=r.get("after_initial"), u_intact=r["u_intact"])
        results["scenarios"][name] = {k: v for k, v in r.items() if k not in ("u_intact", "after_initial")}
    geo = dict(xs=fr.xs.tolist(), ys=fr.ys.tolist(), zs=fr.zs.tolist(),
               cols=[(e["n"], e["storey"], e["ij"], e["b"]) for e in fr.els if e["kind"] == "col"],
               beams=[e["n"] for e in fr.els if e["kind"] == "beam"])
    json.dump(dict(geo=geo, anim=anim), open(OUT / "fe_anim.json", "w"))

    # ---- failure envelope
    env = sweep(p)
    results["envelope"] = env
    print("envelope: collapse cases", sum(e["collapsed"] for e in env), "of", len(env))

    # ---- pushdown (reinforcement scenarios)
    T20, T16, T12 = 314e-6, 201e-6, 113e-6
    pd_design = pushdown(4 * T20, 3 * T16)
    pd_poor = pushdown(3 * T16, 2 * T12)
    results["pushdown"] = {"design (4T20 top / 3T16 bottom)": {k: pd_design[k] for k in ("lam_static_max", "lam_dyn_at_ulim", "n_hinges", "Mp_neg", "Mp_pos")},
                           "poor (3T16 top / 2T12 bottom)": {k: pd_poor[k] for k in ("lam_static_max", "lam_dyn_at_ulim", "n_hinges", "Mp_neg", "Mp_pos")}}
    print("pushdown:", json.dumps(results["pushdown"], indent=0))

    # ---- crushing
    phys = json.load(open(HERE / "out" / "physics_results.json"))
    d30, F30 = column_crush(fc=30)
    d15, F15 = column_crush(fc=15)
    span = 0.75 * 3.0                                   # (1 - lam) h, descent per storey crushed
    def mean_over(d, F, s):
        m = d <= s
        return float(np.trapezoid(F[m], d[m]) / s)
    W = results["scenarios"][list(SCENARIOS)[0]]["W_kN"]
    ncol = fr.nx * fr.ny            # L0 columns in the twin grid
    results["crushing"] = dict(peak_fc30=float(F30.max()), mean_fc30=mean_over(d30, F30, span),
                               peak_fc15=float(F15.max()), mean_fc15=mean_over(d15, F15, span),
                               W_kN=W, measured_rho_early=phys["rho_free_early"][0],
                               k_storeys=phys["k_classical"])
    c = results["crushing"]
    c["mean_storey_over_W_fc30"] = c["mean_fc30"] * ncol / W
    c["mean_storey_over_W_fc15"] = c["mean_fc15"] * ncol / W
    c["peak_storey_over_W_fc30"] = c["peak_fc30"] * ncol / W
    c["peak_storey_over_W_fc15"] = c["peak_fc15"] * ncol / W
    print("crushing:", {k: round(v, 3) if isinstance(v, float) else v for k, v in c.items()})
    json.dump(results, open(OUT / "fe_results.json", "w"), indent=1, default=str)

    # ---- figures
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.2))
    for pdr, lab, col in ((pd_design, "design reinforcement", "#1f6aa5"), (pd_poor, "poor reinforcement", "#c62f2f")):
        cv = np.array(pdr["curve"])
        ax[0].plot(cv[:, 1] * 1000, cv[:, 0], color=col, lw=2, label=f"static pushdown - {lab}")
        ax[0].plot(np.array(pdr["u"]) * 1000, pdr["lam_dyn"], color=col, ls="--", label=f"dynamic (energy) - {lab}")
    ax[0].axhline(1, color="k", lw=0.8)
    ax[0].axvline(pd_design["u_lim"] * 1000, color="0.5", ls=":")
    ax[0].text(pd_design["u_lim"] * 1000 + 5, 0.1, "rotation limit\n(0.02 rad)", fontsize=8, color="0.4")
    ax[0].set(xlim=(0, 1000), ylim=(0, max(2.5, pd_design["lam_static_max"] * 1.1)), xlabel="vertical displacement over lost corner column [mm]",
              ylabel="load factor on (D + 0.25L)", title="Corner-column loss: pushdown + dynamic capacity")
    ax[0].legend(fontsize=7)
    for (d, F, lab, col) in ((d30, F30, "f'c 30 MPa", "#1f6aa5"), (d15, F15, "f'c 15 MPa", "#c62f2f")):
        ax[1].plot(d * 1000, F / 1000, color=col, lw=2, label=lab)
        mm = mean_over(d, F, span)
        ax[1].hlines(mm / 1000, 0, span * 1000, colors=col, linestyles="--")
    ax[1].hlines(phys["rho_free_early"][0] * W / ncol / 1000, 0, span * 1000, colors="#c77700", lw=3, alpha=0.6,
                 label="resistance measured from video (per column)")
    ax[1].set(xlim=(0, span * 1000), xlabel="shortening [mm]", ylabel="axial force per L0 column [MN]",
              title="L0 column crushing: peak vs mean resistance")
    ax[1].legend(fontsize=8)
    names = list(SCENARIOS)
    vals = [results["scenarios"][n]["max_dcr_intact"] for n in names]
    cols_ = ["#c62f2f" if results["scenarios"][n]["collapsed"] else "#2a9d8f" for n in names]
    ax[2].barh(range(len(names)), [results["scenarios"][n]["sumP0_over_W"] for n in names], color=cols_)
    ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels([n.replace(", ", ",\n", 1) for n in names], fontsize=7.5)
    ax[2].axvline(1, color="k", lw=0.8)
    ax[2].set(xlabel="L0 axial capacity / weight above  (sum P0 / W)", title="3-D cascade: red = full L0 loss")
    ax[2].invert_yaxis()
    for n_lost, mk in ((1, "o"), (2, "s")):
        sub = [e for e in env if e["n_lost"] == n_lost]
        ax[3].scatter([e["reserve"] for e in sub], [e["fc"] + (0.6 if n_lost == 2 else -0.6) for e in sub],
                      c=["#9a9a9a" if e["max_dcr"] > 1 else "#c62f2f" if e["collapsed"] else "#2a9d8f" for e in sub],
                      marker=mk, s=46, edgecolor="k", lw=0.4,
                      label=f"{n_lost} corner-end column{'s' if n_lost > 1 else ''} lost")
    ax[3].set(xlabel="L0 capacity reserve (sum P0 / W)", ylabel="concrete strength f'c [MPa]",
              title="Failure envelope\nred: stands, then L0 zipper | grey: could not stand | green: holds", fontsize=9)
    ax[3].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    footer(fig, SRC_TWIN, "pushdown: event-to-event plastic hinges; dynamic capacity per Izzuddin et al. (2008)",
           "crushing: Mander concrete model + bar buckling", "measured resistance: crush-up fit to video")
    fig.savefig(OUT / "fig_fe_mechanics.png", dpi=150)
    print("saved", OUT / "fig_fe_mechanics.png")


if __name__ == "__main__":
    main()
