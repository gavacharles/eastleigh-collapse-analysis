"""
Step 18 - Physics of the collapse: energy and momentum budget of the crush-up.

From the fitted crush-up model (step 2): moving mass m(z) = (N - n(z)) m_s,
potential energy released, kinetic energy of the moving block, energy dissipated
by crushing, and the resistance force against the weight. Shown for 16 and 18
storeys (the video favours 16-18; see step 17).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from credit import footer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
G, LAM, H = 9.81, 0.25, 3.0
fe = json.load(open(OUT / "fe" / "fe_results.json"))
sens = {(r["N"], r["h"]): r for r in json.load(open(OUT / "height_sensitivity.json"))}
W18 = fe["scenarios"][list(fe["scenarios"])[0]]["W_kN"]          # kN, 18-storey twin


def run(N):
    r = sens[(N, H)]
    k, t0 = r["k"], r["t0"]
    ms = W18 / 18 / G * 1e3                                        # kg per storey
    def rhs(_, y):
        n = y[0] / ((1 - LAM) * H)
        rho = k / max(N - n, 0.5)
        return [y[1], G * (1 - rho) if (y[1] > 0 or rho < 1) else 0.0]
    t = np.linspace(t0, t0 + 6, 3000)
    sol = solve_ivp(rhs, (t[0], t[-1]), [0.05, 0.0], t_eval=t, max_step=0.005)
    z, v = sol.y
    n = z / ((1 - LAM) * H)
    m = np.clip(N - n, 0.5, None) * ms
    dz = np.gradient(z)
    pe = np.cumsum(m * G * dz)                                     # J released by the moving block
    ke = 0.5 * m * v ** 2
    rho = k / np.clip(N - n, 0.5, None)
    stop = np.argmax((t > t0 + 1) & (v < 0.02)) or len(t) - 1
    return dict(t=t[:stop], z=z[:stop], v=v[:stop], m=m[:stop], pe=pe[:stop], ke=ke[:stop], rho=rho[:stop], N=N, k=k)


def main():
    runs = [run(16), run(18)]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    for r, ls in zip(runs, ("--", "-")):
        lab = f"{r['N']} storeys"
        ax[0, 0].plot(r["t"], r["pe"] / 1e9, color="#1f6aa5", ls=ls, label=f"potential energy released ({lab})")
        ax[0, 0].plot(r["t"], (r["pe"] - r["ke"]) / 1e9, color="#c62f2f", ls=ls, label=f"dissipated by crushing ({lab})")
        ax[0, 0].plot(r["t"], r["ke"] / 1e9, color="#c77700", ls=ls, label=f"kinetic energy ({lab})")
        ax[0, 1].plot(r["t"], r["rho"], color="#7b2cbf", ls=ls, label=lab)
        ax[1, 0].plot(r["t"], r["m"] / 1e6, color="#2a9d8f", ls=ls, label=lab)
        ax[1, 1].plot(r["t"], r["m"] * r["v"] / 1e6, color="#264653", ls=ls, label=lab)
    ax[0, 0].set(xlabel="time [s]", ylabel="energy [GJ]", title="Energy budget of the falling block")
    ax[0, 0].legend(fontsize=7.5)
    ax[0, 1].axhline(1, color="k", lw=0.8)
    ax[0, 1].text(0.02, 0.92, "resistance < weight: block accelerates\nresistance > weight: block decelerates",
                  transform=ax[0, 1].transAxes, fontsize=8, va="top")
    ax[0, 1].set(xlabel="time [s]", ylabel="crushing resistance / moving weight", ylim=(0, 1.6),
                 title="Why it could not stop: resistance below weight")
    ax[0, 1].legend(fontsize=8)
    ax[1, 0].set(xlabel="time [s]", ylabel="moving mass [kt]", title="Moving mass shrinks as storeys are crushed (crush-up)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].set(xlabel="time [s]", ylabel="momentum [10^6 kg m/s]", title="Momentum of the moving block")
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    footer(fig, "crush-up model fitted to the eyewitness video (steps 2, 17)",
           f"storey mass from the FE twin (W = {W18 / 1000:.0f} MN for 18 storeys); compaction 25 %")
    fig.savefig(OUT / "fig_physics_energy.png", dpi=150)
    for r in runs:
        print(f"N={r['N']}: PE released {r['pe'][-1] / 1e9:.2f} GJ, peak KE {r['ke'].max() / 1e9:.2f} GJ, "
              f"dissipated {(r['pe'][-1] - r['ke'][-1]) / 1e9:.2f} GJ")


if __name__ == "__main__":
    main()
