"""Earth Engine stage: building-height history and before/after satellite change at the site (no field work).

Needs the earthengine-api package, a Google Cloud project registered for Earth Engine
(case.json -> "ee_project") and credentials from `earthengine authenticate` (run once by the user).

  heights   Google Open Buildings 2.5D Temporal (2016-2023, yearly): max and mean building height
            within 20 m of the site point -> storey estimate and growth rate
  optical   Sentinel-2 SR, least-cloud composites 60 days before / up to 30 days after the event:
            RGB thumbnails and brightness change at the site against a 300 m ring
  radar     Sentinel-1 GRD VV, same windows: backscatter change (dB) at the site against the ring

Writes out/earthengine/ee.json, fig_earthengine.png and thumbnails. Proposes DESK_RAPID_VERTICAL_GROWTH
(engineer to accept into case.json -> desk_evidence) when the plot height rose by >= 2 storeys within 3 years.
"""
import json
import subprocess
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

STOREY_M = 3.2
SOURCES = "Google Open Buildings 2.5D Temporal v1; Copernicus Sentinel-2 and Sentinel-1 (ESA), via Google Earth Engine"


def site_stats(ee, img, site, ring, band, scale):
    a = img.select(band).reduceRegion(ee.Reducer.mean(), site, scale).get(band)
    b = img.select(band).reduceRegion(ee.Reducer.mean(), ring, scale).get(band)
    return ee.Dictionary({"site": a, "ring": b})


def heights(ee, pt):
    col = ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1").filterBounds(pt)
    years = sorted({int(y) for y in col.aggregate_array("system:time_start").map(lambda t: ee.Date(t).get("year")).getInfo()})
    zone = pt.buffer(20)
    out = []
    for y in years:
        im = col.filterDate(f"{y}-01-01", f"{y + 1}-01-01").mosaic()
        h = im.select("building_height").updateMask(im.select("building_presence").gt(0.5))
        r = h.reduceRegion(ee.Reducer.max().combine(ee.Reducer.mean(), sharedInputs=True), zone, 4).getInfo()
        hmax, hmean = r.get("building_height_max"), r.get("building_height_mean")
        out.append(dict(year=y, h_max_m=None if hmax is None else round(hmax, 1), h_mean_m=None if hmean is None else round(hmean, 1),
                        storeys_est=None if hmax is None else round(hmax / STOREY_M, 1)))
    return out


def s2_composite(ee, aoi, start, end):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate(str(start), str(end))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    n = col.size().getInfo()
    if not n:
        return None, 0, []

    def mask(i):
        scl = i.select("SCL")
        return i.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
    dates = col.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd")).getInfo()
    return col.map(mask).median(), n, sorted(set(dates))


def s1_composite(ee, aoi, start, end, orbit=None):
    """Median VV; with orbit (relative orbit number) only that track, so before/after share the viewing geometry."""
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(str(start), str(end))
           .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))
    if orbit is not None:
        col = col.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
    n = col.size().getInfo()
    return (col.select("VV").median(), n) if n else (None, 0)


def s1_orbit(ee, aoi, start, end):
    col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi).filterDate(str(start), str(end))
           .filter(ee.Filter.eq("instrumentMode", "IW")))
    orbits = col.aggregate_array("relativeOrbitNumber_start").getInfo()
    return orbits[0] if orbits else None


def thumb(img, aoi, path):
    url = img.visualize(bands=["B4", "B3", "B2"], min=0, max=2500).getThumbURL({"region": aoi, "dimensions": 512, "format": "png"})
    subprocess.run(["curl", "-sS", "-f", "-m", "120", "-o", str(path), url], check=True, capture_output=True)


def figure(case, res, out, path):
    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=0.15)
    ax = fig.add_subplot(gs[0])
    hs = [h for h in res["heights"] if h["h_max_m"] is not None]
    if hs:
        ax.plot([h["year"] for h in hs], [h["h_max_m"] for h in hs], "-o", color="#2a78d6", lw=2, ms=6, label="max within 20 m")
        ax.plot([h["year"] for h in hs], [h["h_mean_m"] for h in hs], "-o", color="#eb6834", lw=2, ms=6, label="mean within 20 m")
        sec = ax.secondary_yaxis("right", functions=(lambda m: m / STOREY_M, lambda s: s * STOREY_M))
        sec.set_ylabel(f"≈ storeys ({STOREY_M} m each)", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylabel("building height (m)", fontsize=9)
    ax.set_title("Building height at the site, Google Open Buildings 2.5D Temporal", fontsize=9.5, loc="left")
    ax.spines[["top"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    for i, k in enumerate(("before", "after")):
        a = fig.add_subplot(gs[1 + i])
        a.axis("off")
        f = out / f"s2_{k}.png"
        o = res["optical"].get(k) or {}
        if f.exists():
            a.imshow(Image.open(f))
            a.add_patch(plt.Circle((256, 256), 30 / (res["thumb_m"] / 512), fill=False, ec="#e34948", lw=1.5))
        state = "" if o.get("clear_at_site") else ": cloud-covered at the site, no comparison"
        a.set_title(f"Sentinel-2 {k} ({o.get('n', 0)} scenes, {', '.join(o.get('dates', [])[-2:]) or 'none'}){state}\nring 30 m",
                    fontsize=9, loc="left")
    ch = res.get("change", {})
    fig.suptitle(f"{case.cfg['name']}: Earth Engine context.  Optical brightness change (site minus ring) "
                 f"{ch.get('optical_brightness_change', 'n/a')}; radar VV change {ch.get('radar_vv_change_db', 'n/a')} dB",
                 x=0.01, ha="left", fontsize=11.5, weight="bold")
    fig.text(0.005, 0.005, f"Data: {SOURCES}.  10 m imagery cannot resolve a single plot reliably; read with the Wayback chips.  "
             f"Preliminary.  {case.credit}", fontsize=7.5, color="#4a545c")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def earthengine(case):
    loc, proj = case.cfg.get("location"), case.cfg.get("ee_project")
    if not loc or not proj:
        print("  skipped: needs case.json location and ee_project")
        return
    try:
        import ee
        ee.Initialize(project=proj)
        ee.Number(1).getInfo()
    except Exception as e:
        print(f"  skipped: Earth Engine unavailable ({str(e)[:160]}). Run `earthengine authenticate` and check that "
              "earthengine.googleapis.com is reachable.")
        return
    out = case.out / "earthengine"
    out.mkdir(parents=True, exist_ok=True)
    ev = date.fromisoformat(case.cfg["event_date"])
    pt = ee.Geometry.Point(loc["lon"], loc["lat"])
    site, ring = pt.buffer(30), pt.buffer(300).difference(pt.buffer(60))
    aoi = pt.buffer(400).bounds()
    res = {"project": proj, "heights": heights(ee, pt), "optical": {}, "radar": {}, "change": {}, "thumb_m": 800}
    after_end = min(ev + timedelta(days=30), date.today())
    win = {"before": (ev - timedelta(days=60), ev), "after": (ev + timedelta(days=1), after_end)}
    comps = {}
    for k, (a, b) in win.items():
        if b <= a:
            continue
        img, n, dates = s2_composite(ee, aoi, a, b)
        valid = img is not None and (img.select("B4").reduceRegion(ee.Reducer.count(), site, 10).get("B4").getInfo() or 0) > 0
        res["optical"][k] = {"n": n, "dates": dates, "clear_at_site": valid}
        if valid:
            thumb(img, aoi, out / f"s2_{k}.png")
            comps[k] = img.select(["B4", "B3", "B2"]).reduce(ee.Reducer.mean()).rename("bright")
    orbit = s1_orbit(ee, aoi, *win["after"]) if "after" in win and win["after"][1] > win["after"][0] else None
    for k, (a, b) in win.items():
        if b <= a or orbit is None:
            continue
        s1, n1 = s1_composite(ee, aoi, a, b, orbit)
        res["radar"][k] = {"n": n1, "relative_orbit": orbit}
        if s1 is not None:
            comps[k + "_vv"] = s1
    if "before" in comps and "after" in comps:
        d = comps["after"].subtract(comps["before"]).rename("d")
        st = site_stats(ee, d, site, ring, "d", 10).getInfo()
        if st["site"] is not None and st["ring"] is not None:
            res["change"]["optical_brightness_change"] = round(st["site"] - st["ring"], 1)
    if "before_vv" in comps and "after_vv" in comps:
        d = comps["after_vv"].subtract(comps["before_vv"]).rename("d")
        st = site_stats(ee, d, site, ring, "d", 10).getInfo()
        if st["site"] is not None and st["ring"] is not None:
            res["change"]["radar_vv_change_db"] = round(st["site"] - st["ring"], 2)
    hs = [h for h in res["heights"] if h["h_max_m"] is not None]
    res["proposals"] = []
    for i, h in enumerate(hs):
        for g in hs[i + 1:]:
            if g["year"] - h["year"] <= 3 and g["h_max_m"] - h["h_max_m"] >= 2 * STOREY_M:
                res["proposals"].append(dict(item="DESK_RAPID_VERTICAL_GROWTH",
                                             reason=f"plot height {h['h_max_m']} m ({h['year']}) to {g['h_max_m']} m ({g['year']})"))
                break
        if res["proposals"]:
            break
    res["sources"] = SOURCES
    json.dump(res, open(out / "ee.json", "w"), indent=1)
    figure(case, res, out, case.out / "fig_earthengine.png")
    for h in hs:
        print(f"  {h['year']}: max {h['h_max_m']} m (~{h['storeys_est']} storeys), mean {h['h_mean_m']} m")
    print(f"  change: {res['change'] or 'n/a'}")
    for p in res["proposals"]:
        print(f"  proposed {p['item']}: {p['reason']}")
