"""
Step 13 - Bayesian ranking of causes.

Bayesian network (exact enumeration), conditioned on "the building collapsed":

  root causes (binary, priors)            mechanism nodes                 collapse mode
  A  alteration / damage of L0 elements ->  CAP  low L0 capacity reserve    BASE = INIT and CASC
  M  deficient materials / workmanship  ->      (noisy-OR of M, X, D)       BLAST, TOP, LATERAL
  X  extra storeys / vertical extension ->  INIT local loss at L0 corner
  D  design / detailing deficiency      ->      (noisy-OR of A, F, CAP)
  F  foundation failure                 ->  CASC cascade across L0
  B  explosion                              (from the FE-twin envelope:
  T  top-down construction failure           high if CAP, low otherwise)
  Q  lateral event (earthquake / wind)

Evidence (from video CV, physics, FE twin, GIS / remote sensing, press) enters as
likelihoods given the mechanism. Every conditional probability is uncertain: a
Monte Carlo draws each from a Beta distribution around its nominal value
(concentration 20) to give credible intervals. Evidence ablation gives each
item's influence.

Probabilities are expert-elicited (documented below) and are meant to be
challenged: change them in CPT / PRIOR and re-run.
"""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit import footer

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "bayes"
OUT.mkdir(parents=True, exist_ok=True)

ROOTS = ["A", "M", "X", "D", "F", "B", "T", "Q"]
NAMES = {"A": "Loss or alteration of ground-storey load-bearing elements (removal, damage, cutting)",
         "M": "Deficient materials / workmanship (low-strength concrete, poor steel)",
         "X": "Extra storeys / vertical extension beyond design",
         "D": "Design or detailing deficiency (undersized columns, weak ties)",
         "F": "Foundation failure (bearing, settlement, expansive clay, excavation)",
         "B": "Explosion (gas or deliberate)",
         "T": "Top-down construction failure (formwork / upper-floor overload)",
         "Q": "Lateral event (earthquake / wind)"}
SHORT = {"A": "L0 load-path loss", "M": "Deficient materials", "X": "Extra storeys", "D": "Design deficiency",
         "F": "Foundation failure", "B": "Explosion", "T": "Top-down construction", "Q": "Lateral event"}

# priors: base rates for a collapsing RC building under construction in Nairobi
# (informed by NBI audits: majority non-compliant; collapse inquiries frequently cite
# materials, unapproved floors and alterations; explosions/lateral events rare)
PRIOR = dict(A=0.25, M=0.50, X=0.30, D=0.30, F=0.15, B=0.01, T=0.08, Q=0.005)

CPT = dict(
    cap_M=0.60, cap_X=0.55, cap_D=0.50, cap_leak=0.05,          # noisy-OR -> CAP
    init_A=0.70, init_F=0.50, init_CAP=0.25, init_leak=0.01,    # noisy-OR -> INIT (local L0 loss)
    casc_CAP=0.85, casc_noCAP=0.15,                              # FE twin: design holds, low reserve cascades
    blast=0.70, top=0.60, lateral=0.50,                          # P(collapse mode | cause)
    # evidence likelihoods
    E1_base=0.90, E1_blast=0.50, E1_top=0.05, E1_lat=0.30,      # failure starts at base, upper block intact
    E2_A=0.70, E2_F=0.80, E2_CAP=0.30, E2_blast=0.30, E2_other=0.20,   # first motion at one CORNER
    E3_A=0.90, E3_notA=0.35,                                     # works/rubble at L0 before collapse
    E4_distress=0.60, E4_none=0.20,                              # cracks reported beforehand
    E5_blast=0.05, E5_notblast=0.95,                             # no flash/fireball, ~1 s initiation
    E6_base=0.80, E6_other=0.40,                                 # resistance ~0.4 W, crush-up fits (rho rises)
    E7_F=0.55, E7_notF=0.90,                                     # dry antecedent rain, interfluve terrain
    E8_F=0.70, E8_notF=0.50,                                     # high-clay soils nearby (SoilGrids)
    E9_X=0.80, E9_notX=0.35,                                     # lower floors occupied while building upward; no occupation cert.
    E10_lat=0.05, E10_notlat=0.95,                               # no sway / toppling, vertical descent
    E11_X=0.60, E11_notX=0.40,                                   # 18 storeys in a rapidly densifying block (imagery 2011-2026)
    E12_CAP=0.75, E12_noCAP=0.30,                                # twin: only low-reserve configurations reproduce the zipper
)
EVIDENCE = {
    "E1": "Failure started at the base; upper block intact (video CV)",
    "E2": "First motion at the street-corner end (differential tracking)",
    "E3": "Rubble, ladders and partly missing blockwork at L0 before collapse (video)",
    "E4": "Cracks reported beforehand (press, IEK)",
    "E5": "No flash or blast signature; ~1 s progressive initiation (video, audio)",
    "E6": "Measured resistance ~0.4 W rising to ~1; crush-up fits (PINN)",
    "E7": "No rain trigger; site on interfluve, far from rivers (GIS)",
    "E8": "High-clay soils nearby (SoilGrids)",
    "E9": "Ground-floor shops occupied during construction without occupation certificate (Governor)",
    "E10": "No sway or toppling; near-vertical descent (video)",
    "E11": "Tall build in a rapidly densifying block (archive imagery 2011-2026)",
    "E12": "FE twin: only low-reserve configurations reproduce the L0 zipper",
}


def noisy_or(active, weights, leak):
    q = 1 - leak
    for a, w in zip(active, weights):
        if a:
            q *= 1 - w
    return 1 - q


def posterior(prior, cpt, use=None):
    use = set(EVIDENCE) if use is None else set(use)
    post = {r: 0.0 for r in ROOTS}
    combos = {}
    Z = 0.0
    for bits in itertools.product((0, 1), repeat=len(ROOTS)):
        s = dict(zip(ROOTS, bits))
        p_root = np.prod([prior[r] if s[r] else 1 - prior[r] for r in ROOTS])
        if p_root < 1e-14:
            continue
        p_cap1 = noisy_or([s["M"], s["X"], s["D"]], [cpt["cap_M"], cpt["cap_X"], cpt["cap_D"]], cpt["cap_leak"])
        for cap in (0, 1):
            pc = p_cap1 if cap else 1 - p_cap1
            p_init1 = noisy_or([s["A"], s["F"], cap], [cpt["init_A"], cpt["init_F"], cpt["init_CAP"]], cpt["init_leak"])
            for init in (0, 1):
                pi = p_init1 if init else 1 - p_init1
                p_casc1 = (cpt["casc_CAP"] if cap else cpt["casc_noCAP"]) if init else 0.0
                for casc in (0, 1):
                    pk = p_casc1 if casc else 1 - p_casc1
                    if pk == 0:
                        continue
                    base = init and casc
                    # other modes (independent)
                    for bl, tp, lt in itertools.product((0, 1), repeat=3):
                        pb = (cpt["blast"] if bl else 1 - cpt["blast"]) if s["B"] else (1.0 if not bl else 0.0)
                        pt = (cpt["top"] if tp else 1 - cpt["top"]) if s["T"] else (1.0 if not tp else 0.0)
                        pl = (cpt["lateral"] if lt else 1 - cpt["lateral"]) if s["Q"] else (1.0 if not lt else 0.0)
                        pm = pb * pt * pl
                        if pm == 0 or not (base or bl or tp or lt):
                            continue                         # condition on collapse
                        # every mode that occurred leaves its own signature: likelihoods multiply
                        occ = [m for m, f in (("base", base), ("blast", bl), ("top", tp), ("lat", lt)) if f]
                        L = 1.0
                        if "E1" in use:
                            for m in occ:
                                L *= {"base": cpt["E1_base"], "blast": cpt["E1_blast"], "top": cpt["E1_top"], "lat": cpt["E1_lat"]}[m]
                        if "E2" in use:
                            if base:
                                L *= noisy_or([s["A"], s["F"], cap], [cpt["E2_A"], cpt["E2_F"], cpt["E2_CAP"]], 0.05)
                            if bl:
                                L *= cpt["E2_blast"]
                            if tp or lt:
                                L *= cpt["E2_other"]
                        if "E3" in use:
                            L *= cpt["E3_A"] if s["A"] else cpt["E3_notA"]
                        if "E4" in use:
                            L *= cpt["E4_distress"] if (cap or s["F"] or s["A"]) else cpt["E4_none"]
                        if "E5" in use:
                            L *= cpt["E5_blast"] if bl else cpt["E5_notblast"]
                        if "E6" in use:
                            L *= cpt["E6_base"] if (base and not (bl or tp)) else cpt["E6_other"]
                        if "E7" in use:
                            L *= cpt["E7_F"] if s["F"] else cpt["E7_notF"]
                        if "E8" in use:
                            L *= cpt["E8_F"] if s["F"] else cpt["E8_notF"]
                        if "E9" in use:
                            L *= cpt["E9_X"] if s["X"] else cpt["E9_notX"]
                        if "E10" in use:
                            L *= cpt["E10_lat"] if lt else cpt["E10_notlat"]
                        if "E11" in use:
                            L *= cpt["E11_X"] if s["X"] else cpt["E11_notX"]
                        if "E12" in use:
                            L *= cpt["E12_CAP"] if cap else cpt["E12_noCAP"]
                        w = p_root * pc * pi * pk * pm * L
                        Z += w
                        for r in ROOTS:
                            active = {"B": bl, "T": tp, "Q": lt}.get(r, s[r])   # B/T/Q: the mode actually happened
                            if active:
                                post[r] += w
                        key = "".join(r for r in ROOTS if {"B": bl, "T": tp, "Q": lt}.get(r, s[r])) or "-"
                        combos[key] = combos.get(key, 0.0) + w
    post = {r: v / Z for r, v in post.items()}
    combos = {k: v / Z for k, v in combos.items()}
    return post, combos


def jitter(d, rng, kappa=20.0):
    out = {}
    for k, v in d.items():
        v = min(max(v, 1e-3), 1 - 1e-3)
        out[k] = rng.beta(v * kappa, (1 - v) * kappa)
    return out


def main():
    post, combos = posterior(PRIOR, CPT)
    # prior-only (collapse observed, no evidence)
    post0, _ = posterior(PRIOR, CPT, use=[])
    rng = np.random.default_rng(1)
    S = 400
    mc = {r: [] for r in ROOTS}
    for _ in range(S):
        p, _c = posterior(jitter(PRIOR, rng, 40), jitter(CPT, rng))
        for r in ROOTS:
            mc[r].append(p[r])
    ci = {r: (float(np.percentile(mc[r], 5)), float(np.percentile(mc[r], 95))) for r in ROOTS}
    # evidence ablation: change in posterior when an item is removed
    abl = {}
    for e in EVIDENCE:
        p_wo, _ = posterior(PRIOR, CPT, use=[x for x in EVIDENCE if x != e])
        abl[e] = {r: post[r] - p_wo[r] for r in ROOTS}
    top = sorted(combos.items(), key=lambda kv: -kv[1])[:8]
    res = dict(prior=PRIOR, posterior_given_collapse_no_evidence=post0, posterior=post, ci90=ci,
               top_explanations=[dict(causes=k, label=" + ".join(SHORT[c] for c in k) if k != "-" else "none",
                                      prob=v) for k, v in top],
               ablation=abl, cpt=CPT, evidence=EVIDENCE, names=NAMES)
    json.dump(res, open(OUT / "bayes_results.json", "w"), indent=1)
    order = sorted(ROOTS, key=lambda r: -post[r])
    print("posterior P(cause | collapse, evidence):")
    for r in order:
        print(f"  {SHORT[r]:24s} prior {PRIOR[r]:.3f}  ->  {post[r]:.3f}   90% CI {ci[r][0]:.2f}-{ci[r][1]:.2f}")
    print("most probable combinations:")
    for t in res["top_explanations"]:
        print(f"  {t['prob']:.3f}  {t['label']}")

    # ---- figure
    fig = plt.figure(figsize=(16, 9.5))
    ax1 = fig.add_axes([0.20, 0.55, 0.33, 0.38])
    ax2 = fig.add_axes([0.66, 0.55, 0.32, 0.38])
    ax3 = fig.add_axes([0.36, 0.07, 0.55, 0.38])
    y = np.arange(len(order))
    ax1.barh(y - 0.2, [PRIOR[r] for r in order], 0.38, color="#b9c4c9", label="prior")
    ax1.barh(y + 0.2, [post[r] for r in order], 0.38, color="#1f6aa5", label="posterior")
    ax1.errorbar([post[r] for r in order], y + 0.2, xerr=[[post[r] - ci[r][0] for r in order], [ci[r][1] - post[r] for r in order]],
                 fmt="none", ecolor="k", capsize=3, lw=1)
    ax1.set_yticks(y); ax1.set_yticklabels([SHORT[r] for r in order]); ax1.invert_yaxis()
    ax1.set(xlim=(0, 1), xlabel="probability the cause was present", title="Causes: prior vs posterior (90% CI)")
    ax1.legend(fontsize=8, loc="lower right")
    labs = [t["label"] for t in res["top_explanations"]][::-1]
    vals = [t["prob"] for t in res["top_explanations"]][::-1]
    ax2.barh(range(len(vals)), vals, color="#c77700")
    ax2.set_yticks(range(len(vals))); ax2.set_yticklabels([l.replace(" + ", " +\n") for l in labs], fontsize=7.5)
    ax2.set(xlabel="joint posterior probability", title="Most probable cause combinations")
    ev = list(EVIDENCE)
    M = np.array([[abl[e][r] for r in order] for e in ev])
    im = ax3.imshow(M, cmap="RdBu_r", vmin=-0.35, vmax=0.35, aspect="auto")
    ax3.set_yticks(range(len(ev))); ax3.set_yticklabels([f"{e}  {EVIDENCE[e]}" for e in ev], fontsize=7.5)
    ax3.set_xticks(range(len(order))); ax3.set_xticklabels([SHORT[r].replace(" ", "\n", 1) for r in order], fontsize=7.5)
    for i in range(len(ev)):
        for j in range(len(order)):
            if abs(M[i, j]) > 0.02:
                ax3.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax3.set_title("Influence of each evidence item on each cause (posterior with item minus without)", fontsize=10)
    plt.colorbar(im, ax=ax3, fraction=0.03)
    footer(fig, "Bayesian network with expert-elicited probabilities (this analysis)",
           "evidence: eyewitness video CV, PINN physics, FE twin, GIS/remote sensing, SoilGrids, Open-Meteo, press reports")
    fig.savefig(OUT / "fig_bayes_causes.png", dpi=140)
    print("saved", OUT / "fig_bayes_causes.png")


if __name__ == "__main__":
    main()
