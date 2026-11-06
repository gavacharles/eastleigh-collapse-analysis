"""
Photo-based preliminary collapse assessment - command-line runner.

    python run.py cases/<case>                       # all stages
    python run.py cases/<case> --stages intake       # one or more stages
    python run.py cases/<case> --photos /path/to/dir # override photo folder

Stages: intake -> context -> earthengine -> evidence -> measure -> bayes -> fieldplan -> report   (see README.md for the procedure)
"""
import argparse
import json

import bn
import context
import core
import earthengine
import field
import figures

STAGES = ["intake", "context", "earthengine", "evidence", "measure", "bayes", "fieldplan", "report"]


def scenario_items(case):
    sup = case.evidence_items()
    groups = {"photos": [case.item(k) for k in sup],
              "record": [case.item(k) for k in case.cfg.get("record_evidence", [])],
              "desk": [case.item(k) for k in case.cfg.get("desk_evidence", [])],
              "external": [v for k, v in case.cfg.get("external_evidence", {}).items() if not k.startswith("_")]}
    return {name: [it for g in gs for it in groups[g]] for name, gs in case.cfg["scenarios"].items()}


def stage_bayes(case):
    sc = scenario_items(case)
    res = {k: bn.posterior(case.net, v) for k, v in sc.items()}
    ci = {k: bn.credible_intervals(case.net, v) for k, v in sc.items()}
    photo_keys = list(case.evidence_items())
    infl = bn.influence(case.net, [case.item(k) for k in photo_keys], photo_keys)
    json.dump(dict(prior=case.net["prior"], posterior=res, ci90=ci, influence_photos_only=infl), open(case.out / "bayes.json", "w"), indent=1)
    figures.bayes_figure(case, res, ci)
    figures.influence_figure(case, infl)
    for k, v in res.items():
        print(f"  {k}: " + ", ".join(f"{case.net['short'][r]} {p:.2f}" for r, p in sorted(v.items(), key=lambda x: -x[1])))


def mode_statement(used):
    if "PANCAKE_IN_FOOTPRINT" in used and "NO_TILT" in used:
        return ("Vertical, progressive (pancake) collapse within the footprint: slabs stacked flat, no tilt or overturning. "
                "Consistent with loss of vertical support; inconsistent with overturning, lateral sway or a blast.")
    if "TILTED_OR_OVERTURNED" in used:
        return "Tilted or overturning collapse: points to foundation failure or lateral action. Check ground, footings and adjacent works."
    if "DEBRIS_THROWN_FAR" in used:
        return "Debris projected well beyond the footprint: check for explosion (scorching, blast damage on neighbours)."
    return "Collapse mode not determined from the available photographs."


def stage_report(case):
    used = case.evidence_items()
    sup = json.load(open(case.out / "evidence_support.json"))
    inv = json.load(open(case.out / "inventory.json"))
    meas = json.load(open(case.out / "measurements.json"))
    bay = json.load(open(case.out / "bayes.json"))
    figs = figures.theme_figures(case)
    gaps = [f"{v['desc']} (needs: {v['look_for']})" for k, v in figures.gap_items(case, used)]
    gaps_short = [v["desc"] for k, v in figures.gap_items(case, used)]
    always = "Where failure started (ground storey vs upper floors): pancaked debris looks the same either way"
    figures.summary_card(case, bay["posterior"], mode_statement(used), [always] + gaps_short)
    net, sc = case.net, list(bay["posterior"])
    order = sorted(net["roots"], key=lambda r: -bay["posterior"][sc[0]][r])
    L = [f"# Photo-based preliminary assessment: {case.cfg['name']}\n",
         f"**Site:** {case.cfg['site']}  \n**Event:** {case.cfg['event_date']}  \n"
         f"**Input:** {len(inv)} photographs ({sum(i['view'] == 'wide' for i in inv)} wide, {sum(i['view'] == 'element' for i in inv)} element); "
         f"EXIF time/GPS: {'present' if any(i['gps'] or i['datetime'] for i in inv) else 'absent'}; {sum(i['redactions'] for i in inv)} faces/plates blurred.  \n"
         "**Status:** preliminary screening by an engineer, not a determination of cause.\n",
         "![Summary](fig_summary_card.jpg)\n",
         "## 1. Collapse mode\n", mode_statement(used) + "\n",
         "## 2. Evidence by theme\n"]
    for f in figs:
        L.append(f"![{f}]({f})\n")
    L.append("| Evidence item | Theme | Supporting photos |\n|---|---|---|")
    for k, v in used.items():
        L.append(f"| {case.item(k)['desc']} | {case.lib['themes'][case.item(k)['theme']]} | {', '.join(v)} |")
    if sup["excluded_low_confidence"]:
        L.append("\nShown but **not used in the ranking** (low confidence, needs verification): " +
                 "; ".join(case.item(k)["desc"] for k in sup["excluded_low_confidence"]) + ".")
    L.append("\n## 3. Measurements\n")
    for k, v in meas.items():
        L.append(f"- **{k.replace('_', ' ')}:** {v}")
    cpath = case.out / "context" / "context.json"
    if cpath.exists():
        ctx = json.load(open(cpath))
        acc = set(case.cfg.get("desk_evidence", []))
        r, o, so = ctx.get("rain") or {}, ctx.get("osm") or {}, ctx.get("soil") or {}
        L.append("\n## 4. Desk-study context (open data)\n\n![Context](fig_context.png)\n")
        L.append(f"- **Earthquakes (USGS, M≥4.5 within 1000 km):** " + ("; ".join(
            f"M{q['mag']} at {q['km']} km ({'within' if q['within_reach'] else 'beyond'} damaging reach)" for q in ctx["earthquakes"][:3]) or "none") + ".")
        if r:
            pc = "n/a" if r.get("r30_percentile") is None else f"{r['r30_percentile']:.0f}th percentile of the previous 10 years"
            L.append(f"- **Rain (Open-Meteo ERA5):** {r['r30']} mm in the 30 days before ({pc}); {r['r7']} mm in the last 7 days.")
        if o:
            L.append(f"- **Surroundings (OSM):** nearest mapped waterway {o['water_min_m'] if o['water_min_m'] is not None else 'none within 300'} m; "
                     f"{o['n_buildings']} buildings within 150 m; {len(o['construction'])} mapped construction sites.")
        L.append(f"- **Soil (SoilGrids):** " + ((so.get("error") or f"clay {so['clay_pct']} % ({', '.join(so['depths'])}), {so.get('wrb')}, {so['offset_km']} km from site")
                                               if so else f"unavailable ({ctx['errors'].get('soil', 'not fetched')})") + ".")
        L.append(f"- **Satellite archive (Esri Wayback):** {len(ctx.get('imagery', []))} distinct acquisitions in `context/sat/`; inspect for construction progress and adjacent works.")
        L.append("\nProposed from open data: " + ("; ".join(f"{p['item']} ({p['reason']}{'; ' + p['note'] if p.get('note') else ''}){' **accepted**' if p['item'] in acc else ''}" for p in ctx["proposals"]) or "nothing") +
                 ". Items enter the ranking only when listed in `case.json` → `desk_evidence`.")
    epath = case.out / "earthengine" / "ee.json"
    if epath.exists():
        ee_ = json.load(open(epath))
        acc = set(case.cfg.get("desk_evidence", []))
        hs = [h for h in ee_["heights"] if h["h_max_m"] is not None]
        if not cpath.exists():
            L.append("\n## 4. Desk-study context (open data)\n")
        L.append("\n![Earth Engine](fig_earthengine.png)\n")
        L.append("- **Building height at the site (Google Open Buildings 2.5D Temporal, max within 20 m):** " +
                 (", ".join(f"{h['year']} {h['h_max_m']} m (~{h['storeys_est']} storeys)" for h in hs) or "no building detected") + ".")
        L.append(f"- **Before/after change at the site vs a 300 m ring (Sentinel-2 brightness; Sentinel-1 VV dB):** {ee_['change'] or 'not available'}.")
        if ee_["proposals"]:
            L.append("- Proposed: " + "; ".join(f"{p['item']} ({p['reason']}){' **accepted**' if p['item'] in acc else ''}" for p in ee_["proposals"]) + ".")
    L.append("\n## 5. Bayesian ranking of causes\n\n![Bayes](fig_bayes_scenarios.png)\n\n| Cause | Prior | " + " | ".join(s.strip() for s in sc) +
             " |\n|---|---|" + "---|" * len(sc))
    for r in order:
        L.append(f"| {net['short'][r]} | {net['prior'][r]:.2f} | " + " | ".join(f"{bay['posterior'][s][r]:.2f}" for s in sc) + " |")
    L.append("\n![Influence](fig_evidence_influence.png)\n")
    L.append("## 6. What these photographs cannot establish (data gaps)\n")
    L.append(f"- {always}.")
    for g in gaps:
        L.append(f"- {g}.")
    fpath = case.out / "field_priority.json"
    if fpath.exists():
        fp = json.load(open(fpath))
        sh = net["short"]
        L.append(f"\n## 7. What to check next (value of information, against: {fp['scenario'].strip()})\n\n![Field priority](fig_field_priority.png)\n")
        L.append("| # | Check | Access | P(found) | If found | If not found | Gain (bits) |\n|---|---|---|---|---|---|---|")
        for i, rr in enumerate(fp["rows"], 1):
            a, b = rr["moves_pos"], rr["moves_neg"]
            L.append(f"| {i} | {rr['name']} ({rr['how']}) | {rr['access']} | {rr['p_positive']:.2f} | {sh[a]} {rr['base'][a]:.2f}→{rr['if_positive'][a]:.2f} | "
                     f"{sh[b]} {rr['base'][b]:.2f}→{rr['if_negative'][b]:.2f} | {rr['gain_bits']:.3f} |")
        L.append("\nDesk checks need no site visit and can start now.")
    L.append(f"\n## 8. Method and reproducibility\n\nIntake (inventory, SHA-256, perceptual-hash duplicate check, face blurring with YuNet plus reviewed manual boxes) "
             "→ engineer-reviewed tagging against a fixed evidence library (`config/evidence_library.json`) → evidence plates and log "
             "→ automatic measurements → exact Bayesian inference over a shared network (`config/network.json`), with 90 % intervals from "
             "Monte Carlo over priors and a one-at-a-time influence analysis → this report. Low-confidence observations are shown but excluded "
             "from inference. Every input, config and code file is hashed in `run_manifest.json`.\n")
    L.append(f"*{case.credit}. Data: {case.cfg['photo_source']}.*")
    (case.out / "report.md").write_text("\n".join(L))
    print("  report:", case.out / "report.md", "| figures:", figs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--stages", nargs="*", default=STAGES)
    ap.add_argument("--photos")
    a = ap.parse_args()
    case = core.Case(a.case, a.photos)
    fn = {"intake": core.intake, "context": context.context, "earthengine": earthengine.earthengine, "evidence": core.evidence, "measure": core.measure, "bayes": stage_bayes,
          "fieldplan": lambda c: field.fieldplan(c, scenario_items), "report": stage_report}
    for s in a.stages:
        print(f"[{s}]")
        fn[s](case)
    core.manifest(case, a.stages)


if __name__ == "__main__":
    main()
