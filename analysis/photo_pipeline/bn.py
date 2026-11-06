"""Generic Bayesian network for collapse-cause ranking (exact enumeration).

Network structure and probabilities: config/network.json. Evidence: lists of items,
each with likelihood terms (see config/evidence_library.json for the term format).
Conditioned on the building having collapsed.
"""
import itertools

import numpy as np


def noisy_or(active, weights, leak):
    q = 1 - leak
    for a, w in zip(active, weights):
        if a:
            q *= 1 - w
    return 1 - q


def term_value(t, fl):
    typ = t["type"]
    if typ == "flag":
        return t["p"][0] if fl[t["flag"]] else t["p"][1]
    if typ == "any":
        return t["p"][0] if any(fl[x] for x in t["of"]) else t["p"][1]
    if typ == "modes":
        v = 1.0
        for k, p in t["p"].items():
            if fl[k]:
                v *= p
        return v
    if typ == "noisy_or":
        return noisy_or([fl[x] for x in t["of"]], t["w"], t["leak"]) if fl[t["when"]] else 1.0
    raise ValueError(f"unknown term type {typ}")


def posterior(net, items, prior=None, return_z=False):
    """items: list of evidence dicts with 'terms'. Returns P(cause | collapse, evidence)
    (and, with return_z, the unnormalised evidence weight used for predictive probabilities)."""
    roots, cpt = net["roots"], net["cpt"]
    prior = prior or net["prior"]
    terms = [t for it in items for t in it.get("terms", [])]
    post = {r: 0.0 for r in roots}
    Z = 0.0
    for bits in itertools.product((0, 1), repeat=len(roots)):
        s = dict(zip(roots, bits))
        p_root = float(np.prod([prior[r] if s[r] else 1 - prior[r] for r in roots]))
        if p_root < 1e-14:
            continue
        p_cap = noisy_or([s["M"], s["X"], s["D"]], [cpt["cap_M"], cpt["cap_X"], cpt["cap_D"]], cpt["cap_leak"])
        for cap in (0, 1):
            pc = p_cap if cap else 1 - p_cap
            p_init = noisy_or([s["A"], s["F"], cap], [cpt["init_A"], cpt["init_F"], cpt["init_CAP"]], cpt["init_leak"])
            for init in (0, 1):
                pi = p_init if init else 1 - p_init
                pk1 = (cpt["casc_CAP"] if cap else cpt["casc_noCAP"]) if init else 0.0
                for casc in (0, 1):
                    pk = pk1 if casc else 1 - pk1
                    if pk == 0:
                        continue
                    base = bool(init and casc)
                    for bl, tp, lt in itertools.product((0, 1), repeat=3):
                        pb = (cpt["blast"] if bl else 1 - cpt["blast"]) if s["B"] else float(not bl)
                        pt = (cpt["top"] if tp else 1 - cpt["top"]) if s["T"] else float(not tp)
                        pl = (cpt["lateral"] if lt else 1 - cpt["lateral"]) if s["Q"] else float(not lt)
                        pm = pb * pt * pl
                        if pm == 0 or not (base or bl or tp or lt):
                            continue                      # condition on collapse
                        fl = dict(s, cap=bool(cap), base=base, blast=bool(bl), top=bool(tp), lat=bool(lt),
                                  base_only=base and not (bl or tp))
                        L = 1.0
                        for t in terms:
                            L *= term_value(t, fl)
                        w = p_root * pc * pi * pk * pm * L
                        Z += w
                        for r in roots:                    # B/T/Q reported as "that mode occurred"
                            if {"B": bl, "T": tp, "Q": lt}.get(r, s[r]):
                                post[r] += w
    post = {r: v / Z for r, v in post.items()}
    return (post, Z) if return_z else post


def credible_intervals(net, items, n=150, kappa=40, seed=3):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pr = {r: float(np.clip(rng.beta(p * kappa, (1 - p) * kappa), 1e-3, 1 - 1e-3)) for r, p in net["prior"].items()}
        draws.append(posterior(net, items, pr))
    return {r: (float(np.percentile([d[r] for d in draws], 5)), float(np.percentile([d[r] for d in draws], 95)))
            for r in net["roots"]}


def influence(net, items, labels):
    """Change in posterior when each evidence item is removed (one at a time)."""
    base = posterior(net, items)
    out = {}
    for i, lab in enumerate(labels):
        wo = posterior(net, items[:i] + items[i + 1:])
        out[lab] = {r: base[r] - wo[r] for r in net["roots"]}
    return out
