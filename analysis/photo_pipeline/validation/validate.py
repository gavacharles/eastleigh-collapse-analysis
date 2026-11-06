"""
Validate the cause ranking against past collapses with published findings (validation/benchmark.json).

    python validation/validate.py                 # v1 (frozen snapshot) vs current config
    python validation/validate.py --versions cur  # current config only

For each case and scenario (A photos only, B photos + early public record) the case evidence is run
through the same network as a live case (bn.posterior). Scores, over the causes an inquiry graded:
  top-1 / top-3   a primary cause is ranked first / in the first three
  Brier           mean (posterior - truth)^2, truth = 1 primary/contributing, 0 ruled out
  Brier skill     1 - Brier / Brier(prior alone): > 0 means the evidence improved on the priors
  AUC             P(posterior of a found cause > posterior of a ruled-out cause), pooled
Outputs go to validation/out/.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bn  # noqa: E402

CREDIT = "Credit: Gava Charles"
SOURCES = "Data: published inquiry findings and photographs (sources per case in benchmark.json)"
VERSIONS = {"v1": (HERE / "evidence_library_v1.json", HERE / "network_v1.json"),
            "cur": (HERE.parent / "config" / "evidence_library.json", HERE.parent / "config" / "network.json")}
SCEN = {"A": ["photos"], "B": ["photos", "record"]}
USE = ("high", "medium")
BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = {"A": "#2a78d6", "B": "#eb6834"}


def run_version(lib, net, bench):
    res, unencoded = {}, {}
    for cid, c in bench["cases"].items():
        res[cid] = {}
        for s, groups in SCEN.items():
            obs = [o for g in groups for o in c[g] if o["conf"] in USE]
            items, keys = [], []
            for o in obs:
                if o["item"] in lib["items"]:
                    if o["item"] not in keys:            # an item counts once however many photos show it
                        keys.append(o["item"]); items.append(lib["items"][o["item"]])
                else:
                    unencoded.setdefault(cid, set()).add(o["item"])
            res[cid][s] = {"posterior": bn.posterior(net, items), "items": keys}
    return res, {k: sorted(v) for k, v in unencoded.items()}


def score(bench, post_of, roots):
    """post_of(cid) -> {root: p}. Returns metrics and per-case ranks."""
    top1 = top3 = 0
    sq, sq0, pos, neg, per = [], [], [], [], {}
    for cid, c in bench["cases"].items():
        p, truth = post_of(cid), c["truth"]
        order = sorted(roots, key=lambda r: -p[r])
        prim = [r for r, v in truth.items() if v == "primary"]
        best = min(order.index(r) for r in prim) + 1
        top1 += best == 1
        top3 += best <= 3
        per[cid] = {"best_primary_rank": best, "ranking": order[:3]}
        for r, v in truth.items():
            y = 0.0 if v == "ruled_out" else 1.0
            sq.append((p[r] - y) ** 2)
            (neg if y == 0 else pos).append(p[r])
    n = len(bench["cases"])
    auc = float(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg]))
    return {"n_cases": n, "top1": top1 / n, "top3": top3 / n, "brier": float(np.mean(sq)), "auc": auc,
            "n_graded": len(sq)}, per


def all_scores(bench, net, res):
    roots = net["roots"]
    prior_m, prior_per = score(bench, lambda cid: net["prior"], roots)
    out = {"prior_only": prior_m, "per_case": {"prior_only": prior_per}}
    for s in SCEN:
        m, per = score(bench, lambda cid: res[cid][s]["posterior"], roots)
        m["brier_skill"] = 1 - m["brier"] / prior_m["brier"]
        out[s] = m
        out["per_case"][s] = per
    return out


def matrix_figure(bench, net, res, path, title):
    roots, short = net["roots"], net["short"]
    cases = list(bench["cases"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.48 * len(cases) + 2.6), sharey=True)
    cmap = matplotlib.colors.ListedColormap(BLUE)
    for ax, s in zip(axes, SCEN):
        M = np.array([[res[c][s]["posterior"][r] for r in roots] for c in cases])
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, c in enumerate(cases):
            truth = bench["cases"][c]["truth"]
            for j, r in enumerate(roots):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else "#1d2329")
                t = truth.get(r)
                if t in ("primary", "contributing"):
                    ax.add_patch(plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, fill=False, ec="#1d2329",
                                               lw=2.2 if t == "primary" else 1.0, ls="-" if t == "primary" else (0, (3, 2))))
                elif t == "ruled_out":
                    ax.text(j + 0.38, i - 0.25, "×", ha="center", va="center", fontsize=8, color="#8a3b12")
        ax.set_xticks(range(len(roots)))
        ax.set_xticklabels([short[r] for r in roots], rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels([f"{bench['cases'][c]['name'].split(',')[0]} ({bench['cases'][c]['date'][:4]})" for c in cases], fontsize=8.5)
        ax.set_title({"A": "A  Photos only", "B": "B  Photos + early public record"}[s], fontsize=10.5, loc="left")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, weight="bold")
    fig.text(0.01, 0.945, "Cell = posterior probability. Solid box = primary cause found by the inquiry; dashed box = contributing; "
             "× = ruled out; unmarked = not established.", fontsize=8.5, color="#4a545c")
    fig.text(0.005, 0.005, f"{SOURCES}.  Evidence encoded by an unblinded assessor.  {CREDIT}", fontsize=7.5, color="#4a545c")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def metrics_figure(scores, path):
    mets = [("top1", "Top-1: a primary cause ranked first"), ("top3", "Top-3: a primary cause in the first three"),
            ("auc", "AUC: found causes above ruled-out ones"), ("brier_skill", "Brier skill vs priors alone")]
    vers = list(scores)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    w = 0.36
    for ax, (k, lab) in zip(axes, mets):
        for i, s in enumerate(SCEN):
            vals = [scores[v][s][k] for v in vers]
            x = np.arange(len(vers)) + (i - 0.5) * w
            ax.bar(x, vals, w - 0.03, color=CAT[s], label={"A": "A  Photos only", "B": "B  Photos + record"}[s])
            for xx, vv in zip(x, vals):
                ax.text(xx, vv + (0.02 if vv >= 0 else -0.06), f"{vv:.2f}", ha="center", fontsize=8, color="#1d2329")
        if k != "brier_skill":
            base = scores[vers[0]]["prior_only"][k]
            ax.axhline(base, color="#4a545c", lw=1, ls=(0, (4, 3)))
            ax.text(-0.62, base + 0.02, "priors alone", ha="left", fontsize=7.5, color="#4a545c")
        else:
            ax.axhline(0, color="#4a545c", lw=0.8)
        ax.set_xticks(range(len(vers)))
        ax.set_xticklabels([{"v1": "v1 (elicited)", "cur": "v2 (library + calibration)"}[v] for v in vers], fontsize=9)
        ax.set_ylim(min(-0.1, min(scores[v][s][k] for v in vers for s in SCEN) - 0.1), 1.08)
        ax.set_title(lab, fontsize=9.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e6e3", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    n = scores[vers[0]]["prior_only"]["n_cases"]
    fig.suptitle(f"Validation on {n} past collapses with published findings (in-sample; small n)", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.text(0.005, 0.005, f"{SOURCES}.  {CREDIT}", fontsize=7.5, color="#4a545c")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="*", default=["v1", "cur"])
    a = ap.parse_args()
    bench = json.load(open(HERE / "benchmark.json"))
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    results, scores = {}, {}
    for v in a.versions:
        lp, npth = VERSIONS[v]
        lib, net = json.load(open(lp)), json.load(open(npth))
        res, unenc = run_version(lib, net, bench)
        sc = all_scores(bench, net, res)
        results[v] = {"library": str(lp.relative_to(HERE.parent)), "network": str(npth.relative_to(HERE.parent)),
                      "unencoded_items": unenc, "scores": sc,
                      "cases": {c: {s: {"items": r[s]["items"], "posterior": {k: round(p, 4) for k, p in r[s]["posterior"].items()}}
                                    for s in SCEN} for c, r in res.items()}}
        scores[v] = sc
        name = {"v1": "v1", "cur": "v2"}[v]
        matrix_figure(bench, net, res, out / f"fig_validation_matrix_{name}.png",
                      f"Cause ranking vs inquiry findings, {name}")
        print(f"[{name}] prior-only: top1 {sc['prior_only']['top1']:.2f} top3 {sc['prior_only']['top3']:.2f} "
              f"AUC {sc['prior_only']['auc']:.2f} Brier {sc['prior_only']['brier']:.3f}")
        for s in SCEN:
            m = sc[s]
            print(f"[{name}] {s}: top1 {m['top1']:.2f} top3 {m['top3']:.2f} AUC {m['auc']:.2f} "
                  f"Brier {m['brier']:.3f} skill {m['brier_skill']:+.2f}")
        for c in bench["cases"]:
            print(f"    {c:22s} " + "  ".join(
                f"{s}: #{sc['per_case'][s][c]['best_primary_rank']} {'/'.join(net['short'][r] for r in sc['per_case'][s][c]['ranking'])}"
                for s in SCEN))
        if unenc:
            print(f"    unencoded: {unenc}")
    json.dump(results, open(out / "validation_results.json", "w"), indent=1)
    metrics_figure(scores, out / "fig_validation_metrics.png")


if __name__ == "__main__":
    main()
