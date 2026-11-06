"""Context stage: open data for any site from case.json location + event_date (no field work).

Fetches (cached under out/context/raw/):
  earthquakes   USGS ComCat, M >= 4.5 within 1000 km, event date -2 to +1 day
  rainfall      Open-Meteo ERA5 reanalysis: 60 days before the event, plus the same 30-day window in the 10 previous years
  surroundings  OpenStreetMap (Overpass): waterways within 300 m, buildings and construction sites within 150 m
  soils         ISRIC SoilGrids 250 m: clay content and WRB class (nearest unmasked pixel within ~2 km)
  imagery       Esri World Imagery Wayback: one chip per distinct acquisition (construction timeline, adjacent works)

Writes out/context/context.json (facts + proposed evidence items with reasons), fig_context.png and
satellite chips. Proposed items do NOT enter the ranking until the engineer copies them into
case.json -> desk_evidence after checking them.
"""
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UA = "building-collapse-preliminary/1.0 (research)"
SOURCES = {"quake": "USGS ComCat", "rain": "Open-Meteo ERA5 reanalysis", "osm": "OpenStreetMap contributors (ODbL)",
           "soil": "ISRIC SoilGrids 250 m", "img": "Esri World Imagery Wayback (Maxar, Airbus and others)"}


def curl(url, out=None, timeout=90):
    cmd = ["curl", "-sS", "-f", "-L", "-m", str(timeout), "-A", UA, url]
    if out:
        subprocess.run(cmd[:1] + ["-o", str(out)] + cmd[1:], check=True, capture_output=True)
        return None
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def cached_json(raw, name, url, data=None):
    f = raw / name
    if not f.exists():
        if data is None:
            body = curl(url)
        else:
            body = subprocess.run(["curl", "-sS", "-f", "-m", "120", "-A", UA, "--data-urlencode", f"data={data}", url],
                                  check=True, capture_output=True).stdout
        f.write_bytes(body)
    return json.loads(f.read_text())


def dist_m(lat0, lon0, lat, lon):
    kx = 111320 * math.cos(math.radians(lat0))
    return math.hypot((lon - lon0) * kx, (lat - lat0) * 110540)


def to_xy(lat0, lon0, lat, lon):
    return (lon - lon0) * 111320 * math.cos(math.radians(lat0)), (lat - lat0) * 110540


def seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    L2 = (bx - ax) ** 2 + (by - ay) ** 2
    t = 0 if L2 == 0 else max(0, min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / L2))
    return math.hypot(px - ax - t * (bx - ax), py - ay - t * (by - ay))


# ------------------------------------------------------------------ fetchers
def earthquakes(raw, lat, lon, ev):
    url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
           f"&starttime={ev - timedelta(days=2)}&endtime={ev + timedelta(days=1)}"
           f"&latitude={lat}&longitude={lon}&maxradiuskm=1000&minmagnitude=4.5")
    d = cached_json(raw, "usgs.json", url)
    out = []
    for f in d["features"]:
        lo, la = f["geometry"]["coordinates"][:2]
        m = f["properties"]["mag"]
        r = dist_m(lat, lon, la, lo) / 1000
        reach = 10 ** (0.5 * m - 1.2)          # screening radius for damaging shaking (~MMI VI): M5 20 km, M6 63 km, M7 200 km
        out.append(dict(mag=m, km=round(r), place=f["properties"]["place"], time=f["properties"]["time"],
                        reach_km=round(reach), within_reach=r <= reach))
    return sorted(out, key=lambda q: q["km"])


def rainfall(raw, lat, lon, ev):
    base = "https://archive-api.open-meteo.com/v1/archive?latitude={}&longitude={}&start_date={}&end_date={}&daily=precipitation_sum&timezone=auto"
    d = cached_json(raw, "rain_event.json", base.format(lat, lon, ev - timedelta(days=60), ev))["daily"]
    days, p = d["time"], d["precipitation_sum"]
    if any(v is None for v in p[-7:]) and (date.today() - ev).days < 90:      # reanalysis lags ~5 days: fill from forecast archive
        fc = cached_json(raw, "rain_recent.json", f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                         f"&past_days=92&forecast_days=1&daily=precipitation_sum&timezone=auto")["daily"]
        fmap = dict(zip(fc["time"], fc["precipitation_sum"]))
        p = [v if v is not None else fmap.get(t) for t, v in zip(days, p)]
    p = np.array([v if v is not None else np.nan for v in p], float)
    clim = cached_json(raw, "rain_clim.json", base.format(lat, lon, date(ev.year - 10, 1, 1), date(ev.year - 1, 12, 31)))["daily"]
    cmap = dict(zip(clim["time"], clim["precipitation_sum"]))
    past = []
    for y in range(ev.year - 10, ev.year):
        try:
            e = ev.replace(year=y)
        except ValueError:                                  # 29 Feb
            e = ev.replace(year=y, day=28)
        vals = [cmap.get(str(e - timedelta(days=k))) for k in range(30)]
        if all(v is not None for v in vals):
            past.append(sum(vals))
    r30, r7 = float(np.nansum(p[-30:])), float(np.nansum(p[-7:]))
    pct = float(np.mean([r30 > x for x in past]) * 100) if past else None
    return dict(days=days, daily=[None if np.isnan(v) else round(float(v), 1) for v in p], r7=round(r7, 1),
                r30=round(r30, 1), clim30=[round(x, 1) for x in past], r30_percentile=pct,
                missing_days=int(np.isnan(p).sum()))


def surroundings(raw, lat, lon):
    q = (f"[out:json][timeout:60];("
         f'way["waterway"~"river|stream|canal|drain|ditch"](around:300,{lat},{lon});'
         f'way["natural"="water"](around:300,{lat},{lon});'
         f'way["building"](around:150,{lat},{lon});'
         f'way["landuse"="construction"](around:150,{lat},{lon});'
         f");out geom;")
    d = None
    for url in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            d = cached_json(raw, "osm.json", url, data=q)
            break
        except subprocess.CalledProcessError:
            continue
    if d is None:
        return None
    water, bld, cons = [], [], []
    for e in d.get("elements", []):
        t, g = e.get("tags", {}), e.get("geometry", [])
        xy = [to_xy(lat, lon, n["lat"], n["lon"]) for n in g]
        if not xy:
            continue
        dmin = min(seg_dist((0, 0), xy[i], xy[i + 1]) for i in range(len(xy) - 1)) if len(xy) > 1 else math.hypot(*xy[0])
        rec = dict(xy=xy, d=round(dmin, 1), tags={k: t[k] for k in ("name", "waterway", "building", "building:levels", "height", "landuse") if k in t})
        if "waterway" in t or t.get("natural") == "water":
            water.append(rec)
        elif t.get("landuse") == "construction" or t.get("building") == "construction":
            cons.append(rec)
        else:
            bld.append(rec)
    levels = [float(b["tags"]["building:levels"]) for b in bld if b["tags"].get("building:levels", "").replace(".", "").isdigit()]
    return dict(water=water, buildings=bld, construction=cons,
                water_min_m=min((w["d"] for w in water), default=None),
                n_buildings=len(bld), n_with_levels=len(levels), max_levels=max(levels, default=None))


def soils(raw, lat, lon):
    f = raw / "soilgrids.json"
    if f.exists():
        return json.loads(f.read_text())
    res, answered = None, 0
    for r_km in (0, 0.5, 1.0, 2.0):
        pts = [(lat, lon)] if r_km == 0 else [(lat + r_km / 110.5 * math.sin(a), lon + r_km / (111.3 * math.cos(math.radians(lat))) * math.cos(a))
                                               for a in np.radians([0, 90, 180, 270])]
        for la, lo in pts:
            try:
                d = json.loads(curl(f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lo:.5f}&lat={la:.5f}"
                                    "&property=clay&depth=0-5cm&depth=30-60cm&value=mean", timeout=30))
            except subprocess.CalledProcessError as e:
                if not answered:                            # service down or rate-limited: do not mistake it for a masked pixel
                    raise RuntimeError(f"SoilGrids service unavailable (curl exit {e.returncode}); re-run the context stage later") from None
                continue
            answered += 1
            try:
                vals = [x["values"]["mean"] for x in d["properties"]["layers"][0]["depths"]]
                if all(v is not None for v in vals):
                    c = json.loads(curl(f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lo:.5f}&lat={la:.5f}&number_classes=3", timeout=60))
                    res = dict(clay_pct=[v / 10 for v in vals], depths=["0-5 cm", "30-60 cm"], wrb=c.get("wrb_class_name"),
                               wrb_top3=c.get("wrb_class_probability"), offset_km=r_km)
                    break
            except Exception:
                continue
        if res:
            break
    res = res or {"error": f"no unmasked SoilGrids pixel within 2 km ({answered} points answered)"}
    f.write_text(json.dumps(res))
    return res


def imagery(raw, out, lat, lon, z=18, keep=12):
    cfg = cached_json(raw, "waybackconfig.json", "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json")
    items = sorted(cfg.items(), key=lambda kv: kv[1]["itemTitle"])
    cache = raw / "wayback_meta.json"
    meta = json.loads(cache.read_text()) if cache.exists() else {}

    def md(item):
        url = (f"{item['metadataLayerUrl']}/identify?geometry={lon},{lat}&geometryType=esriGeometryPoint&sr=4326&layers=all"
               f"&tolerance=2&mapExtent={lon-.01},{lat-.01},{lon+.01},{lat+.01}&imageDisplay=800,800,96&returnGeometry=false&f=json")
        try:
            r = json.loads(curl(url, timeout=40)).get("results", [])
            return r[0]["attributes"] if r else {}
        except Exception:
            return {}
    todo = [(k, v) for k, v in items if k not in meta]
    with ThreadPoolExecutor(8) as ex:
        for (k, v), a in zip(todo, ex.map(lambda kv: md(kv[1]), todo)):
            meta[k] = dict(date=a.get("SRC_DATE"), res=a.get("SRC_RES"), sensor=a.get("SRC_DESC"))
    cache.write_text(json.dumps(meta))
    seen, uniq = set(), []
    for k, _ in items:
        m = meta.get(k, {})
        if m.get("date") and (m["date"], m.get("sensor")) not in seen:
            seen.add((m["date"], m.get("sensor"))); uniq.append((k, m))
    uniq = sorted(uniq, key=lambda km: str(km[1]["date"]))[-keep:]
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    cx, cy = int(x), int(y)
    mpp = 156543.03 * math.cos(math.radians(lat)) / n
    sat = out / "sat"; sat.mkdir(exist_ok=True)
    chips, log, hashes = [], [], set()
    for k, m in uniq:
        rows = []
        for ty in (cy - 1, cy, cy + 1):
            row = []
            for tx in (cx - 1, cx, cx + 1):
                f = raw / f"wb_{k}_{z}_{ty}_{tx}.jpg"
                if not f.exists():
                    try:
                        curl(f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{k}/{z}/{ty}/{tx}", f)
                    except subprocess.CalledProcessError:
                        pass
                im = cv2.imread(str(f)) if f.exists() else None
                row.append(im if im is not None else np.full((256, 256, 3), 200, np.uint8))
            rows.append(np.hstack(row))
        img = np.vstack(rows)
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in hashes:
            continue
        hashes.add(h)
        px, py = int((x - cx + 1) * 256), int((y - cy + 1) * 256)
        d = str(m["date"])
        cv2.circle(img, (px, py), int(50 / mpp), (60, 60, 255), 2)
        cv2.drawMarker(img, (px, py), (60, 60, 255), cv2.MARKER_CROSS, 20, 2)
        img = cv2.copyMakeBorder(img, 256, 256, 256, 256, cv2.BORDER_CONSTANT, value=(200, 200, 200))[py:py + 512, px:px + 512].copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(img, f"{d[:4]}-{d[4:6]}-{d[6:8]}  {m.get('sensor') or ''}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(sat / f"chip_{d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        chips.append(img)
        log.append(dict(acquired=d, sensor=m.get("sensor"), res_m=m.get("res"), release=k))
    return log, chips, mpp


# ------------------------------------------------------------------ stage
def propose(ctx):
    P = []
    hits = [q for q in ctx["earthquakes"] if q["within_reach"]]
    if hits:
        q = max(hits, key=lambda q: q["mag"])
        P.append(dict(item="RECORD_EARTHQUAKE", reason=f"M{q['mag']} at {q['km']} km ({q['place']}); screening reach {q['reach_km']} km"))
    r = ctx.get("rain") or {}
    if r.get("r30_percentile") is not None and r["r30_percentile"] >= 90:
        P.append(dict(item="DESK_ANTECEDENT_HEAVY_RAIN", reason=f"{r['r30']} mm in 30 days, above {r['r30_percentile']:.0f} % of the previous 10 years"))
    s = ctx.get("osm") or {}
    if s.get("water_min_m") is not None and s["water_min_m"] <= 30:
        P.append(dict(item="DESK_NEAR_WATERCOURSE", reason=f"mapped waterway {s['water_min_m']} m from the site point"))
    if s.get("construction"):
        P.append(dict(item="ADJACENT_EXCAVATION", reason=f"{len(s['construction'])} construction site(s) mapped within 150 m: check satellite chips for excavation",
                      note="photo-theme item: accept only if the chips or photos show an excavation"))
    so = ctx.get("soil") or {}
    if so.get("clay_pct"):
        why = f"clay {so['clay_pct']} %, WRB {so.get('wrb')} ({so['offset_km']} km from site)"
        if "Vertisol" in str(so.get("wrb")):
            P.append(dict(item="DESK_EXPANSIVE_CLAY", reason=why))
        elif max(so["clay_pct"]) >= 40:                     # clay-rich but not a swelling class (e.g. kaolinitic Ferralsols)
            P.append(dict(item="DESK_EXPANSIVE_CLAY", reason=why,
                          note=f"high clay but {so.get('wrb')} is not usually expansive: accept only if local geology confirms swelling (black cotton) soil"))
    return P


def figure(case, ctx, chips, mpp, path):
    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.38, wspace=0.28)
    # rainfall
    ax = fig.add_subplot(gs[0, :2])
    r = ctx.get("rain")
    if r:
        t = np.arange(-len(r["daily"]) + 1, 1)
        ax.bar(t, [v or 0 for v in r["daily"]], width=0.8, color="#2a78d6")
        ax.axvspan(-29.5, 0.5, color="#cde2fb", alpha=0.45, lw=0)
        ax.set_xlabel("days before the collapse", fontsize=9); ax.set_ylabel("daily rain (mm)", fontsize=9)
        pc = "n/a" if r["r30_percentile"] is None else f"{r['r30_percentile']:.0f}th percentile"
        ax.set_title(f"Rainfall: {r['r30']} mm in the last 30 days (shaded) = {pc} of the same window, previous 10 years; "
                     f"last 7 days {r['r7']} mm", fontsize=9.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    # earthquakes + soil + proposals (text)
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    lines = ["Earthquakes M≥4.5 within 1000 km (event −2 d to +1 d)"]
    lines += [f"  M{q['mag']}  {q['km']} km  {'WITHIN' if q['within_reach'] else 'beyond'} reach" for q in ctx["earthquakes"][:4]] or ["  none recorded"]
    so = ctx.get("soil") or {}
    lines += ["", "Soil (SoilGrids 250 m)", f"  {so.get('error') or ('clay ' + str(so['clay_pct']) + ' %, ' + str(so.get('wrb')) + ', ' + str(so['offset_km']) + ' km off') if so else 'unavailable: ' + ctx['errors'].get('soil', '')[:40]}"]
    lines += ["", "Proposed evidence (engineer to accept):"] + ([f"  • {p['item']}" for p in ctx["proposals"]] or ["  none"])
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=9, family="monospace", color="#1d2329")
    # surroundings
    ax = fig.add_subplot(gs[1, 0])
    s = ctx.get("osm")
    if s:
        for b in s["buildings"]:
            ax.fill(*zip(*b["xy"]), color="#d9d9d4", ec="#9a9a94", lw=0.4)
        for c in s["construction"]:
            ax.fill(*zip(*c["xy"]), color="#eda100", alpha=0.5, ec="#c98500", lw=0.8)
        for w in s["water"]:
            ax.plot(*zip(*w["xy"]), color="#2a78d6", lw=2)
        ax.add_patch(plt.Circle((0, 0), 30, fill=False, ec="#e34948", ls="--", lw=1))
        ax.plot(0, 0, "x", color="#e34948", ms=9, mew=2)
        ax.set_xlim(-160, 160); ax.set_ylim(-160, 160); ax.set_aspect("equal")
        ax.set_title(f"OSM within 150 m: {s['n_buildings']} buildings, nearest waterway "
                     f"{'none ≤300 m' if s['water_min_m'] is None else str(s['water_min_m']) + ' m'}\n"
                     "red ring 30 m; amber = construction; blue = water", fontsize=9, loc="left")
        ax.tick_params(labelsize=7); ax.set_xlabel("m", fontsize=8)
    # satellite strip
    for i, im in enumerate(chips[-2:]):
        ax = fig.add_subplot(gs[1, 1 + i])
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ax.set_title(f"Satellite {'before' if i == 0 and len(chips) > 1 else 'latest'} (ring 50 m)", fontsize=9, loc="left")
    fig.suptitle(f"{case.cfg['name']}: desk-study context (open data, no field work)", x=0.01, ha="left", fontsize=13, weight="bold")
    fig.text(0.005, 0.005, "Data: " + "; ".join(SOURCES.values()) + f".  Preliminary.  {case.credit}", fontsize=7.5, color="#4a545c")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def context(case):
    loc = case.cfg.get("location")
    if not loc:
        print("  skipped: add \"location\": {\"lat\": .., \"lon\": ..} to case.json")
        return
    lat, lon = loc["lat"], loc["lon"]
    ev = date.fromisoformat(case.cfg["event_date"])
    out = case.out / "context"
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ctx, errors = {"location": loc, "event_date": str(ev)}, {}
    for key, fn in (("earthquakes", lambda: earthquakes(raw, lat, lon, ev)), ("rain", lambda: rainfall(raw, lat, lon, ev)),
                    ("osm", lambda: surroundings(raw, lat, lon)), ("soil", lambda: soils(raw, lat, lon))):
        try:
            ctx[key] = fn()
        except Exception as e:                              # offline or service down: report, carry on
            ctx[key] = [] if key == "earthquakes" else None
            errors[key] = str(e)[:200]
    try:
        ctx["imagery"], chips, mpp = imagery(raw, out, lat, lon)
    except Exception as e:
        ctx["imagery"], chips, mpp = [], [], None
        errors["imagery"] = str(e)[:200]
    ctx["errors"] = errors
    ctx["proposals"] = propose(ctx)
    ctx["sources"] = SOURCES
    slim = json.loads(json.dumps(ctx, default=str))
    if slim.get("osm"):
        for k in ("water", "buildings", "construction"):
            slim["osm"][k] = [{kk: vv for kk, vv in e.items() if kk != "xy"} for e in slim["osm"][k]]
    json.dump(slim, open(out / "context.json", "w"), indent=1)
    figure(case, ctx, chips, mpp, case.out / "fig_context.png")
    acc = set(case.cfg.get("desk_evidence", []))
    for p in ctx["proposals"]:
        print(f"  proposed {p['item']}{' (accepted)' if p['item'] in acc else ''}: {p['reason']}" + (f"  NOTE: {p['note']}" if p.get("note") else ""))
    if not ctx["proposals"]:
        print("  no evidence items proposed from open data")
    for k, e in errors.items():
        print(f"  {k} unavailable: {e}")
