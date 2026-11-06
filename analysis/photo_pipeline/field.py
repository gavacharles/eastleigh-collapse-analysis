"""Fieldplan stage: which follow-up check would most change the cause ranking (value of information).

For each test in config/field_tests.json (one library item each), against the evidence of the case's
last scenario:
  P(positive)    predictive probability the check finds the item, Z(E + item) / Z(E)
  if positive / if negative   posterior of every cause after either outcome
  gain           expected reduction in uncertainty, sum over causes of binary entropy (bits)
Tests whose item is already in the evidence are skipped. Desk checks (no site visit) and field
checks are ranked together and labelled, so desk work can start now.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

import bn
from core import ROOT


def negate(item):
    terms = []
    for t in item["terms"]:
        if t["type"] not in ("flag", "any"):
            raise ValueError("field-test items must use flag/any terms")
        terms.append(dict(t, p=[1 - t["p"][0], 1 - t["p"][1]]))
    return {"terms": terms}


def entropy(post):
    p = np.clip(np.array(list(post.values())), 1e-9, 1 - 1e-9)
    return float(np.sum(-p * np.log2(p) - (1 - p) * np.log2(1 - p)))


def plan(case, items, used_keys):
    tests = json.load(open(ROOT / "config" / "field_tests.json"))["tests"]
    base, Z = bn.posterior(case.net, items, return_z=True)
    H0 = entropy(base)
    rows = []
    for tid, t in tests.items():
        if t["item"] in used_keys:
            continue
        it = case.lib["items"][t["item"]]
        pos, Zp = bn.posterior(case.net, items + [it], return_z=True)
        neg = bn.posterior(case.net, items + [negate(it)])
        pp = Zp / Z
        gain = H0 - (pp * entropy(pos) + (1 - pp) * entropy(neg))
        move = lambda post: max(case.net["roots"], key=lambda r: abs(post[r] - base[r]))
        rows.append(dict(id=tid, name=t["name"], access=t["access"], cost=t["cost"], how=t["how"], item=t["item"],
                         p_positive=pp, gain_bits=gain, base=base, if_positive=pos, if_negative=neg,
                         moves_pos=move(pos), moves_neg=move(neg)))
    return sorted(rows, key=lambda r: -r["gain_bits"]), base


def figure(case, rows, scen, path):
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(12, 0.42 * len(rows) + 1.8))
    col = {"desk": "#2a78d6", "field": "#eb6834"}
    y = np.arange(len(rows))
    ax.barh(y, [r["gain_bits"] for r in rows], color=[col[r["access"]] for r in rows], height=0.66)
    sh = case.net["short"]
    for i, r in enumerate(rows):
        rp, rn = r["moves_pos"], r["moves_neg"]
        ax.text(r["gain_bits"] + 0.004, i, f"P(found) {r['p_positive']:.2f};  if found: {sh[rp]} {r['base'][rp]:.2f}→{r['if_positive'][rp]:.2f};  "
                f"if not: {sh[rn]} {r['base'][rn]:.2f}→{r['if_negative'][rn]:.2f}", va="center", fontsize=7.8, color="#1d2329")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8.5)
    ax.set_xlabel("expected information gain about the causes (bits)", fontsize=9)
    ax.set_xlim(0, max(r["gain_bits"] for r in rows) * 2.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[matplotlib.patches.Patch(color=col["desk"], label="Desk check (no site visit)"),
                       matplotlib.patches.Patch(color=col["field"], label="Field check (site access)")],
              fontsize=8, frameon=False, loc="lower right")
    ax.set_title(f"{case.cfg['name']}: which check would change the ranking most\nagainst the evidence in scenario {scen.strip()}",
                 fontsize=11, loc="left", weight="bold")
    fig.text(0.005, 0.005, f"Data: case evidence; config/field_tests.json; Bayesian network config/network.json.  Preliminary.  {case.credit}",
             fontsize=7.5, color="#4a545c")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fieldplan(case, scenario_items):
    sc = scenario_items(case)
    scen = list(sc)[-1]
    groups = case.cfg["scenarios"][scen]
    keys = list(case.evidence_items()) if "photos" in groups else []
    keys += case.cfg.get("record_evidence", []) if "record" in groups else []
    keys += case.cfg.get("desk_evidence", []) if "desk" in groups else []
    rows, base = plan(case, sc[scen], set(keys))
    figure(case, rows, scen, case.out / "fig_field_priority.png")
    json.dump(dict(scenario=scen, rows=rows), open(case.out / "field_priority.json", "w"), indent=1)
    for r in rows[:5]:
        print(f"  {r['gain_bits']:.3f} bits  [{r['access']}] {r['name']}")
