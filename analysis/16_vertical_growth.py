"""
Step 16 - Remote sensing: vertical-growth index of the block from archive imagery.

Shadow area cast by buildings scales with height / tan(sun elevation). For every
archive acquisition (step 11) the shadow fraction within 115 m of the site is
measured and multiplied by tan(sun elevation) at the typical very-high-resolution
overpass time (~10:30 local solar time), giving a height-proportional index.
Sensor radiometry and haze differ between images, so read it as a trend.
"""
import importlib.util
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from credit import SITE_LABEL, footer

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rs", HERE / "11_remote_sensing_timeline.py")
RS = importlib.util.module_from_spec(spec); spec.loader.exec_module(RS)
LAT, LON = RS.SITE


def sun_elevation(day, solar_hour=10.5, lat=LAT):
    n = day.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(math.radians(360 / 365 * (n - 81)))
    ha = math.radians(15 * (solar_hour - 12))
    phi = math.radians(lat)
    return math.degrees(math.asin(math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(ha)))


def main():
    tl = json.load(open(HERE / "gis" / "timeline" / "timeline.json"))
    rows = []
    for item in tl:
        im, (px, py) = RS.mosaic(item["release_id"])
        g = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
        res = 0.2986
        yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
        mask = np.hypot((xx - px) * res, (yy - py) * res) < 115
        L = g[mask]
        thr = 0.45 * np.median(L)
        frac = float((L < thr).mean())
        d = datetime.strptime(item["acquired"], "%Y%m%d")
        el = sun_elevation(d)
        rows.append(dict(date=item["acquired"], sensor=item["sensor"], shadow_frac=frac, sun_elev=el,
                         index=frac * math.tan(math.radians(el))))
    json.dump(rows, open(HERE / "gis" / "vertical_growth.json", "w"), indent=1)
    t = [datetime.strptime(r["date"], "%Y%m%d") for r in rows]
    idx = np.array([r["index"] for r in rows])
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(t, idx / idx[:3].mean(), "o-", color="#1f6aa5", lw=2)
    for ti, r, v in zip(t, rows, idx / idx[:3].mean()):
        ax.annotate(r["sensor"], (ti, v), textcoords="offset points", xytext=(0, 7), fontsize=7, ha="center", color="0.4")
    ax.set(ylabel="height index (2011-14 = 1)", title="Vertical growth of the block around the site (shadow-based, archive imagery)")
    ax.axhline(1, color="0.6", lw=0.8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    footer(fig, "Esri World Imagery Wayback (Maxar/Vantor/Airbus) - shadow fraction within 115 m x tan(sun elevation)",
           f"site: {SITE_LABEL} (position approx.)")
    fig.savefig(HERE / "gis" / "fig_vertical_growth.png", dpi=150)
    for r in rows:
        print(r["date"], r["sensor"], f"shadow {r['shadow_frac']:.2f}  sun {r['sun_elev']:.0f} deg  index {r['index']:.3f}")


if __name__ == "__main__":
    main()
