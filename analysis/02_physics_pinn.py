"""
Step 2 - Physics: what resistance did the structure offer while it came down?

Equation of motion of the descending upper block (z downward, m = moving mass):
        m(z) z'' = m(z) g - R(t)            rho(t) := R / (m g)
        z''      = g (1 - rho(t))
rho = 0 -> free fall; rho = 1 -> constant velocity; rho > 1 -> arrest.

Bazant-type crush-up (base-first) closure: the block loses its bottom storey each
time it descends (1 - lam) h, while each storey offers a mean dynamic crushing
force F_c = k * w_storey:
        n(z) = z / ((1 - lam) h),     rho(z) = k / (N - n(z))

Three estimators, which should agree if the data are physically consistent:
  (1) classical ODE least-squares fit of the crush-up model   (k, t0, offsets)
  (2) PINN-A: free-form inverse PINN, rho(t) is its own network (no closure)
  (3) PINN-B: crush-up-constrained PINN, single learnable scalar k
The camera gap 2.2-3.8 s is not observed; the PINNs bridge it with physics.
Each measured segment has an unknown displacement origin -> learnable offset.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from credit import SRC_CV, SRC_MODEL, SRC_VIDEO, footer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
G = 9.81
N_STOREYS = 18          # reported height
H = 3.0                 # m storey height (same assumption as step 1)
LAM = 0.25              # rubble compaction ratio (crushed storey keeps 25 % of h)
SEGS = ["A_centre_M", "B_roof"]

# ---------------------------------------------------------------- data
raw = np.genfromtxt(OUT / "kinematics.csv", delimiter=",", names=True, dtype=None, encoding=None)
rest = raw[raw["segment"] == "A_centre_L"]
data = {s: raw[raw["segment"] == s] for s in SEGS}
tilt = raw[raw["segment"] == "A_left_minus_centre"]


# ---------------------------------------------------------------- (1) classical fit
def simulate(k, t0, t, N=N_STOREYS, h=H, lam=LAM, beta=0.0, z0=0.05, v0=0.0, with_v=False):
    """Crush-up ODE from onset t0 (small seed z0 for the failed ground storey).
    beta: linear taper of storey crushing capacity with height (columns get
    smaller up the building): F_c(n) = k w (1 - beta n / N)."""
    def rhs(_, y):
        n = y[0] / ((1 - lam) * h)
        rho = k * max(1 - beta * n / N, 0.05) / max(N - n, 0.5)
        return [y[1], G * (1 - rho) if y[1] > 0 or rho < 1 else 0.0]
    tt = np.asarray(t)
    z = np.zeros_like(tt, dtype=float)
    v = np.zeros_like(tt, dtype=float)
    m = tt > t0
    if m.any():
        sol = solve_ivp(rhs, (t0, tt[m].max() + 1e-6), [z0, v0], t_eval=tt[m],
                        max_step=0.01, rtol=1e-7)
        z[m], v[m] = sol.y[0], sol.y[1]
    return (z, v) if with_v else z


def resid(p, N=N_STOREYS, h=H, lam=LAM):
    k, t0, cA, cB = p
    r = [(simulate(k, t0, rest["t_s"], N, h, lam) - rest["z_m"]) / rest["sigma_m"]]
    for s, c in zip(SEGS, (cA, cB)):
        d = data[s]
        r.append((simulate(k, t0, d["t_s"], N, h, lam) + c - d["z_m"]) / d["sigma_m"])
    return np.concatenate(r)


def fit(N=N_STOREYS, h=H, lam=LAM):
    best = None
    for k0 in (6, 9, 12, 15):
        for t00 in (0.8, 1.0, 1.2):
            r = least_squares(resid, [k0, t00, 0.0, -10.0], args=(N, h, lam),
                              bounds=([0, 0.3, -5, -60], [40, 1.6, 5, 10]))
            if best is None or r.cost < best.cost:
                best = r
    return best


fit0 = fit()
k_hat, t0_hat, cA_hat, cB_hat = fit0.x
chi2 = 2 * fit0.cost / (len(fit0.fun) - 4)

# constant-acceleration (constant rho) alternative, for model comparison
def resid_const(p):
    rho, t0, cA, cB = p
    def z(t):
        return np.where(t > t0, 0.5 * G * (1 - rho) * (t - t0) ** 2, 0.0)
    r = [(z(rest["t_s"]) - rest["z_m"]) / rest["sigma_m"]]
    for s, c in zip(SEGS, (cA, cB)):
        r.append((z(data[s]["t_s"]) + c - data[s]["z_m"]) / data[s]["sigma_m"])
    return np.concatenate(r)
fc = least_squares(resid_const, [0.5, 1.0, 0, -10], bounds=([0, 0.3, -5, -80], [1.5, 1.6, 5, 10]))
chi2_const = 2 * fc.cost / (len(fc.fun) - 4)

# sensitivity: storey height, compaction, storey count
sens = []
for h in (2.7, 3.0, 3.3):
    for lam in (0.15, 0.25, 0.35):
        for N in (17, 18, 19):
            r = fit(N, h, lam)
            sens.append(dict(h=h, lam=lam, N=N, k=r.x[0], t0=r.x[1],
                             chi2=2 * r.cost / (len(r.fun) - 4)))
ks = np.array([s["k"] for s in sens])
# tapered-capacity crush-up (k0, beta)
def resid_taper(p):
    k, beta, t0, cA, cB = p
    r = [(simulate(k, t0, rest["t_s"], beta=beta) - rest["z_m"]) / rest["sigma_m"]]
    for s, c in zip(SEGS, (cA, cB)):
        d = data[s]
        r.append((simulate(k, t0, d["t_s"], beta=beta) + c - d["z_m"]) / d["sigma_m"])
    return np.concatenate(r)
ft = min((least_squares(resid_taper, [k_hat, b0, t0_hat, cA_hat, cB_hat],
                        bounds=([0, 0, 0.3, -5, -60], [40, 0.95, 1.6, 5, 10])) for b0 in (0.1, 0.4, 0.7)),
         key=lambda r: r.cost)
kT, betaT, t0T = ft.x[:3]
chi2_taper = 2 * ft.cost / (len(ft.fun) - 5)
print(f"[classical taper] k0 = {kT:.2f}, beta = {betaT:.2f}, t0 = {t0T:.2f}, chi2/dof = {chi2_taper:.2f}")

print(f"[classical] k = {k_hat:.2f} (sens. range {ks.min():.1f}-{ks.max():.1f}), "
      f"t0 = {t0_hat:.2f} s, chi2/dof = {chi2:.2f}; const-rho model chi2/dof = {chi2_const:.2f} "
      f"(rho={fc.x[0]:.2f})")


# ---------------------------------------------------------------- (2)(3) PINNs
torch.manual_seed(0)
T_LO, T_HI = 0.0, 5.6
def tn(t):  # normalise time
    return (t - T_LO) / (T_HI - T_LO) * 2 - 1


class MLP(torch.nn.Module):
    def __init__(self, width=48, depth=4, out_act=None):
        super().__init__()
        layers, d = [], 1
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        layers += [torch.nn.Linear(d, 1)]
        self.f = torch.nn.Sequential(*layers)
        self.out_act = out_act

    def forward(self, t):
        y = self.f(tn(t))
        return self.out_act(y) if self.out_act else y


def tensors(arr):
    return tuple(torch.tensor(np.ascontiguousarray(arr[c], dtype=np.float32))[:, None]
                 for c in ("t_s", "z_m", "sigma_m"))


def train_pinn(mode, seed, iters=6000):
    torch.manual_seed(seed)
    # z(t) = softplus-gated network so z ~ 0 well before onset
    znet = MLP()
    off = torch.nn.Parameter(torch.tensor([0.0, cB_hat], dtype=torch.float32))
    params = list(znet.parameters()) + [off]
    if mode == "free":
        rnet = MLP(width=24, depth=3, out_act=lambda y: 2.0 * torch.sigmoid(y))
        params += list(rnet.parameters())
    else:
        logk = torch.nn.Parameter(torch.tensor(np.log(8.0), dtype=torch.float32))
        params += [logk]
    opt = torch.optim.Adam(params, lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, 2000, 0.4)
    tr, zr, sr = tensors(rest)
    segs = [tensors(data[s]) for s in SEGS]
    tc = torch.linspace(1.35, 5.0, 300)[:, None].requires_grad_(True)  # physics domain
    for it in range(iters):
        opt.zero_grad()
        loss_d = (((znet(tr) - zr) / sr) ** 2).mean()
        for j, (t, z, s) in enumerate(segs):
            loss_d = loss_d + (((znet(t) + off[j] - z) / s) ** 2).mean()
        z = znet(tc)
        dz = torch.autograd.grad(z.sum(), tc, create_graph=True)[0]
        d2z = torch.autograd.grad(dz.sum(), tc, create_graph=True)[0]
        if mode == "free":
            rho = rnet(tc)
            drho = torch.autograd.grad(rho.sum(), tc, create_graph=True)[0]
            reg = 1e-3 * (drho ** 2).mean()
        else:
            n = z / ((1 - LAM) * H)
            rho = torch.exp(logk) / torch.clamp(N_STOREYS - n, min=0.5)
            reg = 0.0
        loss_p = ((d2z - G * (1 - rho)) ** 2).mean() / G ** 2
        loss_v = torch.relu(-dz).pow(2).mean()          # block does not move upward
        w_p = min(1.0, it / 1500) * 50.0                  # physics weight ramp
        loss = loss_d + w_p * loss_p + 10 * loss_v + reg
        loss.backward()
        opt.step()
        sched.step()
    tg = torch.linspace(0.0, 5.6, 400)[:, None].requires_grad_(True)
    z = znet(tg)
    dz = torch.autograd.grad(z.sum(), tg, create_graph=True)[0]
    d2z = torch.autograd.grad(dz.sum(), tg)[0]
    res = dict(t=tg.detach().numpy().ravel(), z=z.detach().numpy().ravel(),
               v=dz.detach().numpy().ravel(), a=d2z.numpy().ravel(),
               off=off.detach().numpy().tolist(), loss_d=float(loss_d.detach()), loss_p=float(loss_p.detach()))
    if mode == "free":
        res["rho"] = rnet(tg).detach().numpy().ravel()
    else:
        res["k"] = float(torch.exp(logk).detach())
        n = res["z"] / ((1 - LAM) * H)
        res["rho"] = res["k"] / np.clip(N_STOREYS - n, 0.5, None)
    return res


SEEDS = range(5)
pinn_free = [train_pinn("free", s) for s in SEEDS]
pinn_cu = [train_pinn("crushup", s) for s in SEEDS]
k_pinn = np.array([r["k"] for r in pinn_cu])
print(f"[PINN-B crush-up] k = {k_pinn.mean():.2f} +/- {k_pinn.std():.2f} (5 seeds)")
win = lambda r, a, b: r["rho"][(r["t"] > a) & (r["t"] < b)].mean()
rhoA = np.array([win(r, 1.5, 2.2) for r in pinn_free])
rhoB = np.array([win(r, 3.8, 4.9) for r in pinn_free])
print(f"[PINN-A free] mean rho 1.5-2.2 s = {rhoA.mean():.2f}+/-{rhoA.std():.2f}; "
      f"3.8-4.9 s = {rhoB.mean():.2f}+/-{rhoB.std():.2f}")

# predicted end of collapse (extrapolation beyond last observation at 4.97 s)
tt = np.linspace(0, 15, 6000)
z_end = N_STOREYS * (1 - LAM) * H
t_ff = np.sqrt(2 * z_end / G)
def end_of(zz, vv, t0):
    i = np.argmax((tt > t0 + 1) & ((zz >= z_end * 0.98) | (vv < 0.05)))
    return tt[i], zz[i]
zz, v = simulate(k_hat, t0_hat, tt, with_v=True)
zT, vT = simulate(kT, t0T, tt, beta=betaT, with_v=True)
t_end, z_stop = end_of(zz, v, t0_hat)
t_endT, z_stopT = end_of(zT, vT, t0T)
print(f"[constant k] motion stops at {t_end:.1f} s after {z_stop:.0f} m "
      f"(theoretical full crush {z_end:.0f} m at lam={LAM}; equals full crush if lam~"
      f"{1 - z_stop / (N_STOREYS * H):.2f})")
print(f"[tapered]    motion stops at {t_endT:.1f} s after {z_stopT:.0f} m "
      f"(duration {t_endT - t0T:.1f} s vs free fall {t_ff:.1f} s)")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
tg = pinn_cu[0]["t"]
a0 = ax[0, 0]
a0.plot(rest["t_s"], rest["z_m"], "k.", ms=4, label="CV: centre @L1 vs ground (pre-zoom)")
for s, c, col in zip(SEGS, (cA_hat, cB_hat), ("tab:blue", "tab:red")):
    a0.plot(data[s]["t_s"], data[s]["z_m"] - c, ".", color=col, ms=5, label=f"CV: {s} (offset-aligned)")
for r in pinn_cu:
    a0.plot(tg, r["z"], color="tab:green", alpha=0.35, lw=1)
a0.plot(tt[tt < 5.6], zz[tt < 5.6], "k--", lw=1.2, label=f"crush-up ODE, constant k={k_hat:.1f}")
a0.plot(tt[tt < 5.6], zT[tt < 5.6], "-.", color="tab:brown", lw=1.2,
        label=f"crush-up ODE, tapered k0={kT:.1f}, beta={betaT:.2f}")
ff = np.where(tg > t0_hat, 0.5 * G * (tg - t0_hat) ** 2, 0)
a0.plot(tg, ff, ":", color="grey", label="free fall from same onset")
a0.axvspan(2.25, 3.75, color="0.9", zorder=0)
a0.text(3.0, 2, "camera gap\n(physics bridge)", ha="center", fontsize=8, color="0.4")
a0.set(xlim=(0, 5.6), ylim=(-1, 40), xlabel="video time [s]", ylabel="descent of upper block z [m]",
       title="Descent: CV measurements vs physics")
a0.legend(fontsize=7, loc="upper left")

a1 = ax[0, 1]
for r in pinn_cu:
    a1.plot(tg, r["v"], color="tab:green", alpha=0.35)
a1.plot(tt[tt < 5.6], v[tt < 5.6], "k--", label="crush-up ODE, constant k")
a1.plot(tt[tt < 5.6], vT[tt < 5.6], "-.", color="tab:brown", label="crush-up ODE, tapered")
a1.axvline(4.97, color="0.5", lw=0.6)
a1.text(5.0, 22, "last\nobs.", fontsize=7, color="0.4")
a1.plot(tg, np.where(tg > t0_hat, G * (tg - t0_hat), 0), ":", color="grey", label="free fall")
a1.axvspan(2.25, 3.75, color="0.9", zorder=0)
a1.set(xlim=(0, 5.6), ylim=(0, 25), xlabel="video time [s]", ylabel="velocity [m/s]",
       title="Velocity (PINN-B ensemble, green)")
a1.legend(fontsize=8)

a2 = ax[1, 0]
for r in pinn_free:
    m = (r["t"] > 1.35) & (r["t"] < 4.97)
    a2.plot(r["t"][m], r["rho"][m], color="tab:purple", alpha=0.4)
for r in pinn_cu:
    m = (r["t"] > 1.35) & (r["t"] < 4.97)
    a2.plot(r["t"][m], r["rho"][m], color="tab:green", alpha=0.4)
a2.axhline(1, color="k", lw=0.8)
a2.axhline(0, color="grey", lw=0.8, ls=":")
a2.axvspan(2.25, 3.75, color="0.9", zorder=0)
a2.plot([], [], color="tab:purple", label="PINN-A free-form rho(t)")
a2.plot([], [], color="tab:green", label="PINN-B crush-up rho(z(t))")
a2.set(xlim=(1.3, 5.0), ylim=(-0.1, 1.6), xlabel="video time [s]",
       ylabel="rho = resistance / moving weight", title="Identified structural resistance")
a2.legend(fontsize=8)

a3 = ax[1, 1]
a3.plot(tilt["t_s"], tilt["z_m"], "o-", ms=3, color="tab:orange")
a3.axvline(t0_hat, color="k", ls="--", lw=0.8)
a3.text(t0_hat + 0.03, 0.05, f"global onset\n t0={t0_hat:.2f}s", fontsize=8)
a3.set(xlabel="video time [s]", ylabel="left end minus centre [m]",
       title="Initiation: differential drop of L1 slab (street-corner end)")
fig.tight_layout(rect=(0, 0.025, 1, 1))
footer(fig, SRC_VIDEO, SRC_CV, SRC_MODEL)
fig.savefig(OUT / "fig_kinematics_pinn.png", dpi=150)

json.dump(dict(k_classical=k_hat, t0=t0_hat, chi2_crushup=chi2, chi2_const=chi2_const,
               rho_const=fc.x[0], k_sens_min=float(ks.min()), k_sens_max=float(ks.max()),
               k_pinn_mean=float(k_pinn.mean()), k_pinn_std=float(k_pinn.std()),
               rho_free_early=[float(rhoA.mean()), float(rhoA.std())],
               rho_free_late=[float(rhoB.mean()), float(rhoB.std())],
               k_taper=kT, beta_taper=betaT, t0_taper=t0T, chi2_taper=chi2_taper,
               const_k_stop_t=float(t_end), const_k_stop_z=float(z_stop),
               taper_stop_t=float(t_endT), taper_stop_z=float(z_stopT),
               crush_duration_taper=float(t_endT - t0T), z_end=z_end, t_freefall=float(t_ff),
               tilt_max=float(np.nanmax(tilt["z_m"])), sensitivity=sens),
          open(OUT / "physics_results.json", "w"), indent=1)
print("saved", OUT / "fig_kinematics_pinn.png")
