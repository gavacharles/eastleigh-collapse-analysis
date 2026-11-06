"""
Leave-one-out recalibration of the network's conditional probabilities against benchmark.json.

    python validation/calibrate.py            # report only
    python validation/calibrate.py --apply    # write the selected values to config/network.json

A small grid over the parameters that decide how design and material deficiencies reach a
ground-storey failure (cap_D, cap_M, init_CAP) is scored by the mean log score over inquiry-graded causes
(scenarios A and B). Each case is scored with parameters fitted on the other cases only, so the
held-out score estimates performance on a new case. Values are applied only if the held-out
score beats the expert-elicited values. init_A is not fitted: only one benchmark inquiry graded
ground-storey load-path loss, so the benchmark cannot inform it (a fit drives it down only to
make room for the graded causes). Priors are not fitted: the benchmark is a set of famous
inquiries, not a sample of collapses, so its cause frequencies are not base rates.
"""
import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bn  # noqa: E402

GRID = {"cap_D": [0.5, 0.7, 0.85], "cap_M": [0.45, 0.6, 0.75], "init_CAP": [0.25, 0.4, 0.55]}
SCEN = {"A": ["photos"], "B": ["photos", "record"]}


def case_items(lib, c, groups):
    keys = []
    for g in groups:
        for o in c[g]:
            if o["conf"] in ("high", "medium") and o["item"] in lib["items"] and o["item"] not in keys:
                keys.append(o["item"])
    return [lib["items"][k] for k in keys]


def case_eval(net, lib, c):
    """Log score, top-1 and top-3 hit for one case, averaged over scenarios."""
    ls, t1, t3 = [], [], []
    for groups in SCEN.values():
        p = bn.posterior(net, case_items(lib, c, groups))
        for r, v in c["truth"].items():
            q = float(np.clip(p[r], 1e-4, 1 - 1e-4))
            ls.append(np.log(1 - q) if v == "ruled_out" else np.log(q))
        order = sorted(net["roots"], key=lambda r: -p[r])
        best = min(order.index(r) for r, v in c["truth"].items() if v == "primary")
        t1.append(best == 0); t3.append(best < 3)
    return float(np.mean(ls)), float(np.mean(t1)), float(np.mean(t3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    bench = json.load(open(HERE / "benchmark.json"))["cases"]
    lib = json.load(open(HERE.parent / "config" / "evidence_library.json"))
    npath = HERE.parent / "config" / "network.json"
    net0 = json.load(open(npath))
    cases = list(bench)
    elicited = {k: net0["cpt"][k] for k in GRID}
    points = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    if elicited not in points:
        points.append(elicited)

    def net_with(pt):
        n = copy.deepcopy(net0)
        n["cpt"].update(pt)
        return n

    table = np.array([[case_eval(net_with(pt), lib, bench[c]) for c in cases] for pt in points])  # pts x cases x 3
    i_el = points.index(elicited)
    dist = [sum(abs(pt[k] - elicited[k]) for k in GRID) for pt in points]

    loo = []
    for j, c in enumerate(cases):
        train = np.delete(table[:, :, 0], j, axis=1).mean(axis=1)
        best = max(range(len(points)), key=lambda i: (round(train[i], 4), -dist[i]))  # ties -> closest to elicited
        loo.append({"case": c, "chosen": points[best], "held_out": table[best, j].tolist(), "elicited": table[i_el, j].tolist()})
    ho = np.array([r["held_out"] for r in loo]).mean(axis=0)
    el = np.array([r["elicited"] for r in loo]).mean(axis=0)
    full = max(range(len(points)), key=lambda i: (round(table[i, :, 0].mean(), 4), -dist[i]))
    adopt = ho[0] > el[0]

    print(f"elicited  {elicited}: log score {el[0]:.3f}  top1 {el[1]:.2f}  top3 {el[2]:.2f}")
    print(f"LOO (held-out)          : log score {ho[0]:.3f}  top1 {ho[1]:.2f}  top3 {ho[2]:.2f}")
    print(f"full-data best {points[full]}: in-sample log score {table[full, :, 0].mean():.3f}")
    stab = {k: sorted({r['chosen'][k] for r in loo}) for k in GRID}
    print("values chosen across folds:", stab)
    out = {"grid": GRID, "elicited": elicited, "elicited_score": dict(zip(["log", "top1", "top3"], el.tolist())),
           "loo_score": dict(zip(["log", "top1", "top3"], ho.tolist())), "folds": loo, "fold_values": stab,
           "full_data_best": points[full], "adopted": bool(adopt)}
    (HERE / "out").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "out" / "calibration.json", "w"), indent=1)
    if a.apply and adopt:
        net0["cpt"].update(points[full])
        net0["_calibration"] = (f"cpt {sorted(GRID)} set by leave-one-out calibration on validation/benchmark.json "
                                f"(2026-10-29); elicited values were {elicited}. See validation/README.md.")
        json.dump(net0, open(npath, "w"), indent=1)
        print("applied to", npath)
    elif a.apply:
        print("not applied: held-out score does not beat the elicited values")


if __name__ == "__main__":
    main()
