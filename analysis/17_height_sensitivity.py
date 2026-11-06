"""
Step 17 - Sensitivity to the building height.

Press reports differ (12, 16 or 18 storeys); a storey count on the white end
facade in the video gives ~16 (+/- 2). The crush-up fit (step 2) is repeated for
N = 12, 14, 16, 18 storeys and two storey heights; the resistance ratio at onset
and the fit quality show whether the conclusions depend on N.
"""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
G, LAM = 9.81, 0.25
raw = np.genfromtxt(OUT / "kinematics.csv", delimiter=",", names=True, dtype=None, encoding=None)
rest = raw[raw["segment"] == "A_centre_L"]
segs = [raw[raw["segment"] == s] for s in ("A_centre_M", "B_roof")]


def simulate(k, t0, t, N, h):
    def rhs(_, y):
        rho = k / max(N - y[0] / ((1 - LAM) * h), 0.5)
        return [y[1], G * (1 - rho) if (y[1] > 0 or rho < 1) else 0.0]
    t = np.asarray(t); z = np.zeros_like(t, float); m = t > t0
    if m.any():
        z[m] = solve_ivp(rhs, (t0, t[m].max() + 1e-6), [0.05, 0], t_eval=t[m], max_step=0.01).y[0]
    return z


def resid(p, N, h):
    k, t0, cA, cB = p
    r = [(simulate(k, t0, rest["t_s"], N, h) - rest["z_m"]) / rest["sigma_m"]]
    for d, c in zip(segs, (cA, cB)):
        r.append((simulate(k, t0, d["t_s"], N, h) + c - d["z_m"]) / d["sigma_m"])
    return np.concatenate(r)


def main():
    rows = []
    for N in (12, 14, 16, 18):
        for h in (3.0, 3.2):
            best = min((least_squares(resid, [k0, 0.9, 0, -10], args=(N, h),
                                      bounds=([0, 0.3, -5, -60], [40, 1.6, 5, 10])) for k0 in (5, 8, 12)),
                       key=lambda r: r.cost)
            k, t0 = best.x[:2]
            rows.append(dict(N=N, h=h, k=round(float(k), 2), t0=round(float(t0), 2), rho_onset=round(float(k / N), 3),
                             chi2=round(float(2 * best.cost / (len(best.fun) - 4)), 2)))
            print(rows[-1])
    json.dump(rows, open(OUT / "height_sensitivity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
