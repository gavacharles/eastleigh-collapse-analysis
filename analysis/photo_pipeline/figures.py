"""Figures: one evidence figure per theme, Bayesian scenarios, evidence influence, summary card."""
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from core import COL, font

BG, INK, SOFT = (248, 248, 246), (24, 30, 36), (96, 106, 114)
CONF = {"high": (42, 157, 143), "medium": (230, 150, 30), "low": (170, 170, 170)}


def wrap(d, text, f, maxw):
    words, line, out = text.split(), "", []
    for w in words:
        t = (line + " " + w).strip()
        if d.textlength(t, font=f) > maxw:
            out.append(line); line = w
        else:
            line = t
    return out + [line]


def fill_crop(im, box, aspect, pad=0.10):
    """Crop around the tagged box, expanded to the tile aspect ratio (adds context, no empty bands)."""
    Wf, Hf = im.size
    x0, y0, x1, y1 = box[0] * Wf, box[1] * Hf, box[2] * Wf, box[3] * Hf
    w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    w, h = min(w, Wf), min(h, Hf)
    if w / h > aspect:
        w = h * aspect
    else:
        h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - w / 2, 0), Wf - w)
    top = min(max(cy - h / 2, 0), Hf - h)
    return im.crop((int(left), int(top), int(left + w), int(top + h)))


def gap_items(case, used, theme=None):
    """High-value library items not observed, excluding those contradicted by an observed item."""
    out = []
    for k, v in case.lib["items"].items():
        if v["value"] != "high" or v["theme"] == "record" or k in used:
            continue
        if any(c in used for c in v.get("contradicted_by", [])):
            continue
        if theme is None or v["theme"] == theme:
            out.append((k, v))
    return out


def theme_figures(case):
    obs_by_theme = {}
    for name, p in case.tags["photos"].items():
        for o in p["observations"]:
            obs_by_theme.setdefault(o["category"], []).append((name, o))
    used = case.evidence_items()
    order = {"high": 0, "medium": 1, "low": 2}
    out = []
    for i, (theme, title) in enumerate(case.lib["themes"].items()):
        if theme not in obs_by_theme:
            continue
        obs = sorted(obs_by_theme[theme], key=lambda x: order[x[1]["confidence"]])[:6]
        n = len(obs)
        cols = min(3, n); rows = (n + cols - 1) // cols
        tw, th = 540, 470
        W = 60 + cols * (tw + 30) + 520
        H = 170 + rows * (th + 150) + 90
        c = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(c)
        col = COL[theme]
        d.rectangle([0, 0, 14, H], fill=col)
        d.text((50, 34), f"{case.cfg['name'].upper()} · PHOTO EVIDENCE", font=font(20, 2), fill=col)
        d.text((50, 66), title, font=font(52, 2), fill=INK)
        for k, (name, o) in enumerate(obs):
            im = Image.open(case.out / "redacted" / name).convert("RGB")
            Wf, Hf = im.size
            crop = fill_crop(im, o["box"], tw / th).resize((tw, th), Image.LANCZOS)
            gx = 50 + (k % cols) * (tw + 30)
            gy = 170 + (k // cols) * (th + 150)
            d.rectangle([gx - 3, gy - 3, gx + tw + 3, gy + th + 3], fill=(255, 255, 255), outline=(220, 222, 224))
            c.paste(crop, (gx + (tw - crop.width) // 2, gy + (th - crop.height) // 2))
            cc = CONF[o["confidence"]]
            d.rectangle([gx, gy + th + 12, gx + 96, gy + th + 38], fill=cc)
            d.text((gx + 8, gy + th + 14), o["confidence"].upper(), font=font(16, 2), fill=(255, 255, 255))
            d.text((gx + 108, gy + th + 14), name, font=font(16), fill=SOFT)
            for j, ln in enumerate(wrap(d, o["label"], font(20), tw)[:3]):
                d.text((gx, gy + th + 46 + j * 27), ln, font=font(20), fill=INK)
        # right panel: what it tells us + gaps
        rx = 60 + cols * (tw + 30) + 10
        d.text((rx, 170), "WHAT IT TELLS US", font=font(18, 2), fill=col)
        y = 204
        items = [k for k in used if case.item(k)["theme"] == theme]
        excluded = [k for k, v in case.evidence_items(("low",)).items() if case.item(k)["theme"] == theme and k not in used]
        for k in items:
            for j, ln in enumerate(wrap(d, case.item(k)["desc"], font(20), 470)):
                if j == 0:
                    d.ellipse([rx, y + 9, rx + 9, y + 18], fill=col)
                d.text((rx + 20, y), ln, font=font(20), fill=INK); y += 27
            y += 10
        for k in excluded:
            for j, ln in enumerate(wrap(d, "Unverified (low confidence, not used in ranking): " + case.item(k)["desc"], font(18), 470)):
                d.text((rx + 20, y), ln, font=font(18), fill=SOFT); y += 25
            y += 10
        gaps = [v for k, v in gap_items(case, used, theme)]
        if gaps:
            y += 14
            d.text((rx, y), "NOT ESTABLISHED FROM THESE PHOTOS", font=font(18, 2), fill=(180, 60, 50)); y += 32
            for g in gaps[:5]:
                for j, ln in enumerate(wrap(d, f"{g['desc']}. Needs: {g['look_for']}", font(17), 470)[:4]):
                    d.text((rx + 20, y), ln, font=font(17), fill=SOFT); y += 23
                y += 8
        d.text((50, H - 50), f"Data: {case.cfg['photo_source']}; faces/plates blurred; engineer-reviewed tags. "
               f"Preliminary.   {case.credit}", font=font(17), fill=SOFT)
        fn = case.out / f"fig_evidence_{i + 1:02d}_{theme}.jpg"
        c.save(fn, quality=90)
        out.append(fn.name)
    return out


def bayes_figure(case, res, ci):
    net = case.net
    sc = list(res)
    order = sorted(net["roots"], key=lambda r: -res[sc[-1]][r])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    y = np.arange(len(order))
    colors = ["#9aa3a8", "#c77700", "#1f6aa5", "#2a9d8f"]
    k = len(sc) + 1
    hgt = 0.8 / k
    ax.barh(y - 0.4 + hgt / 2, [net["prior"][r] for r in order], hgt * 0.9, color="#e0e3e5", label="prior")
    for i, s in enumerate(sc):
        v = [res[s][r] for r in order]
        ax.barh(y - 0.4 + hgt * (i + 1.5), v, hgt * 0.9, color=colors[i % 4], label=s,
                xerr=[[v[j] - ci[s][r][0] for j, r in enumerate(order)], [ci[s][r][1] - v[j] for j, r in enumerate(order)]],
                error_kw=dict(lw=0.8, capsize=2))
    ax.set_yticks(y); ax.set_yticklabels([net["short"][r] for r in order]); ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="P(cause present | collapse, evidence)   (bars: 90% interval over priors)",
           title=f"{case.cfg['name']}: Bayesian ranking of causes")
    ax.legend(fontsize=9, loc="lower right")
    fig.text(0.005, 0.005, f"Data: {case.cfg['photo_source']} (engineer-reviewed); expert-elicited network (config/network.json).  "
             f"{case.credit}", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(case.out / "fig_bayes_scenarios.png", dpi=150)
    plt.close(fig)


def influence_figure(case, infl):
    net = case.net
    labs = list(infl)
    roots = [r for r in net["roots"] if max(abs(infl[l][r]) for l in labs) > 0.005]
    M = np.array([[infl[l][r] for r in roots] for l in labs])
    fig, ax = plt.subplots(figsize=(2.2 + 1.3 * len(roots), 1.4 + 0.55 * len(labs)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax.set_xticks(range(len(roots))); ax.set_xticklabels([net["short"][r].replace(" ", "\n", 1) for r in roots], fontsize=8)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([textwrap.shorten(case.item(l)["desc"], 70) for l in labs], fontsize=8)
    for i in range(len(labs)):
        for j in range(len(roots)):
            if abs(M[i, j]) > 0.01:
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_title("Photos-only: how much each observation moves each cause", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03)
    fig.text(0.005, 0.005, f"Posterior with item minus without.  {case.credit}", fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(case.out / "fig_evidence_influence.png", dpi=150)
    plt.close(fig)


def summary_card(case, res, mode_text, gaps):
    net = case.net
    s0 = list(res)[0]
    order = sorted(net["roots"], key=lambda r: -res[s0][r])
    W, H = 1600, 900
    c = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(c)
    d.text((56, 40), f"{case.cfg['name'].upper()} · PHOTO-ONLY PRELIMINARY ASSESSMENT", font=font(20, 2), fill=(214, 55, 46))
    d.text((56, 72), "What the photographs establish", font=font(50, 2), fill=INK)
    d.text((56, 160), "COLLAPSE MODE", font=font(18, 2), fill=SOFT)
    y = 190
    for ln in wrap(d, mode_text, font(25), 700):
        d.text((56, y), ln, font=font(25), fill=INK); y += 34
    d.text((56, y + 26), "NOT ESTABLISHED (photos needed)", font=font(18, 2), fill=(180, 60, 50))
    y += 58
    for g in gaps[:6]:
        for j, ln in enumerate(wrap(d, g, font(20), 690)[:2]):
            if j == 0:
                d.ellipse([56, y + 9, 65, y + 18], fill=(180, 60, 50))
            d.text((76, y), ln, font=font(20), fill=INK); y += 27
        y += 6
    x0 = 840
    d.text((x0, 160), f"CAUSE RANKING ({s0.strip()})", font=font(18, 2), fill=SOFT)
    for i, r in enumerate(order):
        yy = 196 + i * 72
        p = res[s0][r]; pr = net["prior"][r]
        d.text((x0, yy), net["short"][r], font=font(22, 2), fill=INK)
        d.rectangle([x0, yy + 34, x0 + 600, yy + 52], fill=(228, 230, 232))
        d.rectangle([x0, yy + 34, x0 + int(600 * p), yy + 52], fill=(31, 106, 165))
        d.line([x0 + int(600 * pr), yy + 30, x0 + int(600 * pr), yy + 56], fill=(214, 55, 46), width=3)
        d.text((x0 + 612, yy + 28), f"{p:.2f}", font=font(20, 2), fill=INK)
    d.text((x0, 196 + len(order) * 72), "red tick = prior (base rate before evidence)", font=font(16), fill=SOFT)
    d.text((56, H - 46), f"Data: {case.cfg['photo_source']}; engineer-reviewed tags; expert-elicited Bayesian network. "
           f"Preliminary screening, not a determination of cause.   {case.credit}", font=font(16), fill=SOFT)
    c.save(case.out / "fig_summary_card.jpg", quality=92)
