"""
Step 7 - GIS analysis and maps (UTM 37S, EPSG:32737).

  A  Site context on satellite imagery
  B  Terrain and hydrology: hillshade, contours, watercourses, distance and fall
     from the site to the nearest river, elevation profile
  C  Exposure: neighbours within 1x and 2x the collapsed height, dust corridors
  D  Emergency access: network routes from fire stations and nearest hospitals
  E  High-rise screening: tall buildings in the Eastleigh area (OSM-tagged only)

Also writes gis/eastleigh_layers.geojson-style JSON for the 3-D page and an
interactive Leaflet map (gis/eastleigh_map.html) with switchable basemaps.
"""
import json
import math
from pathlib import Path

import contextily as cx
import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pyproj import Transformer
from shapely.geometry import LineString, Point, Polygon

from credit import CREDIT, SRC_DEM, SRC_ESRI, SRC_ESRI_GREY, SRC_OSM, SRC_PRESS, footer

HERE = Path(__file__).resolve().parent
RAW = HERE / "gis" / "raw"
OUT = HERE / "gis"
meta = json.load(open(RAW / "meta.json"))
SITE_LL = tuple(meta["site"])                    # lat, lon
H_BLDG = 55.0                                    # m, ~18 storeys
CRS = "EPSG:32737"
to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
SITE = Point(*to_utm.transform(SITE_LL[1], SITE_LL[0]))
SITE_UNCERT = 100.0                              # m, plot not yet confirmed
ATTR_IMG = "Imagery: Esri, Maxar, Earthstar Geographics | Data: (c) OpenStreetMap contributors"
ATTR_OSM = "(c) OpenStreetMap contributors"


# ------------------------------------------------------------------ load layers
def ways(name, closed=False):
    els = json.load(open(RAW / f"{name}.json"))["elements"]
    rows = []
    for e in els:
        g = e.get("geometry")
        if not g or len(g) < 2:
            continue
        coords = [(p["lon"], p["lat"]) for p in g]
        geom = Polygon(coords) if closed and len(coords) >= 4 else LineString(coords)
        rows.append(dict(id=e["id"], geometry=geom, **{k.replace(":", "_"): v for k, v in e["tags"].items()}))
    return gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(CRS)


def levels(row):
    for key in ("building_levels",):
        v = row.get(key)
        try:
            return float(str(v).split(";")[0])
        except (TypeError, ValueError):
            pass
    h = row.get("height")
    try:
        return float(str(h).replace("m", "")) / 3.0
    except (TypeError, ValueError):
        return np.nan


bld = ways("buildings", closed=True)
bld = bld[bld.geometry.is_valid]
bld["levels"] = bld.apply(levels, axis=1)
roads = ways("roads")
water = ways("water")
rivers = water[water["waterway"].isin(["river", "stream"])]
drains = water[water["waterway"].isin(["drain", "ditch", "canal"])]
ems_els = json.load(open(RAW / "ems.json"))["elements"]
ems = gpd.GeoDataFrame(
    [dict(name=e["tags"].get("name", "(unnamed)"), amenity=e["tags"]["amenity"],
          geometry=Point((e.get("center") or e)["lon"], (e.get("center") or e)["lat"]))
     for e in ems_els], crs="EPSG:4326").to_crs(CRS)
site_gdf = gpd.GeoDataFrame(geometry=[SITE], crs=CRS)
zone = SITE.buffer(SITE_UNCERT)

dem = np.load(RAW / "dem.npz")
elev, (lw, ln, le, ls_) = dem["elev"], dem["bounds"]
xw, yn = to_utm.transform(lw, ln)
xe, ys = to_utm.transform(le, ls_)

stats = {}


def scalebar(ax, length=200, loc=(0.05, 0.05)):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * loc[0]
    y = y0 + (y1 - y0) * loc[1]
    ax.plot([x, x + length], [y, y], color="k", lw=4, solid_capstyle="butt", zorder=20)
    ax.plot([x, x + length], [y, y], color="w", lw=2, solid_capstyle="butt", zorder=21,
            dashes=(length / 8, length / 8))
    ax.text(x + length / 2, y + (y1 - y0) * 0.012, f"{length:.0f} m", ha="center", va="bottom",
            fontsize=8, zorder=22, bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
    ax.annotate("N", xy=(x1 - (x1 - x0) * 0.05, y1 - (y1 - y0) * 0.05),
                xytext=(x1 - (x1 - x0) * 0.05, y1 - (y1 - y0) * 0.12),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5), ha="center", fontsize=10,
                fontweight="bold", zorder=22)


def frame(ax, half):
    ax.set_xlim(SITE.x - half, SITE.x + half)
    ax.set_ylim(SITE.y - half, SITE.y + half)
    ax.set_xticks([])
    ax.set_yticks([])


def site_marker(ax, label=True):
    gpd.GeoSeries([zone], crs=CRS).boundary.plot(ax=ax, color="#ff2d2d", lw=1.2, ls="--", zorder=15)
    ax.plot(SITE.x, SITE.y, marker="*", ms=16, color="#ff2d2d", mec="w", mew=1.2, zorder=16)
    if label:
        ax.text(SITE.x + 18, SITE.y + 18, "Reported site\n(Ninth St x 2nd Ave)\n±100 m until plot confirmed",
                fontsize=7.5, color="w", zorder=16, bbox=dict(fc="#b31b1b", ec="none", alpha=0.85, pad=2))


# ------------------------------------------------------------------ A. context
def map_context():
    fig, ax = plt.subplots(figsize=(10, 10))
    half = 600
    frame(ax, half)
    cx.add_basemap(ax, crs=CRS, source=cx.providers.Esri.WorldImagery, zoom=17, attribution=ATTR_IMG)
    view = bld.cx[SITE.x - half:SITE.x + half, SITE.y - half:SITE.y + half]
    view[view.levels.isna()].boundary.plot(ax=ax, color="w", lw=0.3, alpha=0.6)
    tagged = view[view.levels.notna()]
    tagged.plot(ax=ax, column="levels", cmap="plasma", vmin=1, vmax=20, alpha=0.75, edgecolor="w",
                lw=0.4, legend=True, legend_kwds=dict(label="OSM storeys (tagged)", shrink=0.6))
    major = roads[roads.highway.isin(["primary", "secondary", "tertiary", "trunk"])]
    major.cx[SITE.x - half:SITE.x + half, SITE.y - half:SITE.y + half].plot(ax=ax, color="#ffd23f", lw=1.6)
    for nm in ("Ninth Street", "2nd Avenue Eastleigh", "First Avenue Eastleigh"):
        r = roads[roads.get("name") == nm]
        if len(r):
            r.plot(ax=ax, color="#36c9ff", lw=2.2)
            c = r.unary_union.interpolate(0.5, normalized=True)
            if abs(c.x - SITE.x) < half and abs(c.y - SITE.y) < half:
                ax.text(c.x, c.y, nm, color="w", fontsize=8, fontweight="bold",
                        bbox=dict(fc="#0b4f6c", ec="none", alpha=0.8, pad=1.5))
    site_marker(ax)
    scalebar(ax, 200)
    ax.set_title("A. Site context - Eastleigh, Kamukunji Sub-County, Nairobi", fontsize=12, loc="left")
    footer(fig, SRC_ESRI, SRC_OSM, "site location: " + SRC_PRESS)
    fig.savefig(OUT / "map_A_context.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ B. terrain & hydrology
def map_terrain():
    fig = plt.figure(figsize=(14, 8.8))
    ax = fig.add_axes([0.02, 0.05, 0.58, 0.9])
    axp = fig.add_axes([0.66, 0.55, 0.32, 0.36])
    axt = fig.add_axes([0.66, 0.08, 0.32, 0.36])
    ext = (xw, xe, ys, yn)
    lsrc = LightSource(azdeg=315, altdeg=40)
    dx = (xe - xw) / elev.shape[1]
    rgb = lsrc.shade(elev, cmap=plt.cm.gist_earth, vert_exag=4, dx=dx, dy=dx, blend_mode="soft",
                     vmin=1580, vmax=1700)
    ax.imshow(rgb, extent=ext, origin="upper")
    X = np.linspace(xw, xe, elev.shape[1])
    Y = np.linspace(yn, ys, elev.shape[0])
    cs = ax.contour(X, Y, elev, levels=np.arange(1580, 1700, 5), colors="k", linewidths=0.35, alpha=0.6)
    ax.clabel(cs, levels=cs.levels[::4], fontsize=6, fmt="%d")
    rivers.plot(ax=ax, color="#0077ff", lw=2.2)
    drains.plot(ax=ax, color="#4fb3ff", lw=0.8)
    for nm, grp in rivers.groupby(rivers.get("name").fillna("")):
        if nm:
            c = grp.unary_union.centroid
            p = grp.unary_union.interpolate(grp.unary_union.project(c))
            ax.text(p.x, p.y, nm, color="#002b80", fontsize=8, fontweight="bold",
                    bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
    bld.cx[SITE.x - 1800:SITE.x + 1800, SITE.y - 1800:SITE.y + 1800].plot(ax=ax, color="0.2", alpha=0.18, lw=0)
    # nearest river / drain and profile
    riv_u = rivers.unary_union
    near = riv_u.interpolate(riv_u.project(SITE))
    d_riv = SITE.distance(riv_u)
    d_drain = SITE.distance(drains.unary_union) if len(drains) else np.nan
    ax.plot([SITE.x, near.x], [SITE.y, near.y], color="#ff2d2d", lw=1.2, ls=":")

    def z_at(x, y):
        c = int((x - xw) / (xe - xw) * elev.shape[1])
        r = int((yn - y) / (yn - ys) * elev.shape[0])
        return float(elev[np.clip(r, 0, elev.shape[0] - 1), np.clip(c, 0, elev.shape[1] - 1)])

    z_site, z_riv = z_at(SITE.x, SITE.y), z_at(near.x, near.y)
    # SRTM is a radar surface model (buildings, trees, ~5 m noise): take the terrain
    # trend over ~100 m before computing slope, otherwise noise reads as steep slope
    from scipy.ndimage import gaussian_filter
    gy, gx = np.gradient(gaussian_filter(elev, sigma=50 / dx), dx, dx)
    c = int((SITE.x - xw) / (xe - xw) * elev.shape[1])
    r = int((yn - SITE.y) / (yn - ys) * elev.shape[0])
    slope = math.degrees(math.atan(math.hypot(gx[r, c], gy[r, c])))
    stats.update(site_elev_m=z_site, nearest_river_m=d_riv, nearest_river_elev_m=z_riv,
                 nearest_drain_m=d_drain, site_slope_deg=slope,
                 nearest_river_name=str(rivers.iloc[rivers.distance(SITE).argmin()].get("name")))
    site_marker(ax, label=False)
    frame(ax, 1800)
    scalebar(ax, 500)
    ax.set_title("B. Terrain and hydrology (SRTM-based DEM, 5 m contours)", fontsize=12, loc="left")
    ax.legend(handles=[Line2D([], [], color="#0077ff", lw=2.2, label="river / stream"),
                       Line2D([], [], color="#4fb3ff", lw=0.8, label="mapped drain"),
                       Line2D([], [], color="#ff2d2d", ls=":", label="site to nearest river")],
              loc="lower right", fontsize=8)
    # profile along site -> nearest river, extended 300 m behind the site
    vec = np.array([near.x - SITE.x, near.y - SITE.y]); vec /= np.linalg.norm(vec)
    s = np.linspace(-300, d_riv + 150, 200)
    prof = [z_at(SITE.x + v * vec[0], SITE.y + v * vec[1]) for v in s]
    axp.fill_between(s, 1580, prof, color="#c9b48a")
    axp.plot(s, prof, color="#6b5530")
    axp.axvline(0, color="#ff2d2d", lw=1.2)
    axp.text(5, max(prof) - 1, "site", color="#ff2d2d", fontsize=8)
    axp.axvline(d_riv, color="#0077ff", lw=1.2)
    axp.text(d_riv + 5, max(prof) - 1, "river", color="#0077ff", fontsize=8)
    axp.set(xlabel="distance from site towards nearest river [m]", ylabel="elevation [m]",
            ylim=(min(prof) - 3, max(prof) + 3), title="Elevation profile")
    axt.axis("off")
    txt = (f"Site elevation            {z_site:7.0f} m\n"
           f"Terrain slope (~100 m)    {slope:7.1f} deg\n"
           f"Nearest river/stream      {d_riv:7.0f} m  ({stats['nearest_river_name']})\n"
           f"  fall to river           {z_site - z_riv:7.0f} m\n"
           f"Nearest mapped drain      {d_drain:7.0f} m\n\n"
           "Reading: the site sits on the interfluve between the\n"
           "Mathare and Nairobi river valleys, well above and away\n"
           "from both floodplains. Terrain alone gives no slope-\n"
           "stability or flooding mechanism. Foundation risk turns\n"
           "on soil type (black-cotton clay / weathered tuff),\n"
           "groundwater and any adjacent excavation, which need\n"
           "a site investigation, not a DEM.\n\n"
           "DEM: SRTM-based surface model, ~30 m, +/-5 m noise.")
    axt.text(0, 1, txt, va="top", family="monospace", fontsize=9)
    footer(fig, SRC_DEM, SRC_OSM)
    fig.savefig(OUT / "map_B_terrain_hydrology.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ C. exposure
def map_exposure():
    fig, ax = plt.subplots(figsize=(10, 10))
    half = 260
    frame(ax, half)
    cx.add_basemap(ax, crs=CRS, source=cx.providers.Esri.WorldImagery, zoom=18, attribution=ATTR_IMG)
    r1, r2 = SITE.buffer(H_BLDG), SITE.buffer(2 * H_BLDG)
    d = bld.distance(SITE)
    b1 = bld[d <= H_BLDG]
    b2 = bld[(d > H_BLDG) & (d <= 2 * H_BLDG)]
    b2.plot(ax=ax, color="#ffb703", alpha=0.55, edgecolor="w", lw=0.4)
    b1.plot(ax=ax, color="#e63946", alpha=0.65, edgecolor="w", lw=0.5)
    gpd.GeoSeries([r1, r2], crs=CRS).boundary.plot(ax=ax, color=["#e63946", "#ffb703"], lw=1.6)
    for g, lab in ((r1, "1 x H = 55 m: debris reach (toppling bound)"), (r2, "2 x H = 110 m")):
        ax.text(SITE.x, SITE.y + (g.bounds[3] - SITE.y) + 4, lab, ha="center", color="w", fontsize=8,
                bbox=dict(fc="k", ec="none", alpha=0.6, pad=1.5))
    # dust corridors: street canyons along Ninth Street and 2nd Avenue (observed dust ran down the street)
    for nm, col in (("Ninth Street", "#90e0ef"), ("2nd Avenue Eastleigh", "#90e0ef")):
        r = roads[roads.get("name") == nm]
        if len(r):
            corr = r.unary_union.intersection(SITE.buffer(300)).buffer(10)
            gpd.GeoSeries([corr], crs=CRS).plot(ax=ax, color=col, alpha=0.30, edgecolor=col, lw=0.8)
    for _, b in b1.iterrows():
        if not np.isnan(b.levels):
            c = b.geometry.representative_point()
            ax.text(c.x, c.y, f"{b.levels:.0f}F", fontsize=7, color="w", ha="center", fontweight="bold")
    site_marker(ax, label=False)
    scalebar(ax, 50)
    n1, n2 = len(b1), len(b2)
    tag1 = b1.levels.notna().sum()
    stats.update(exposed_1H=n1, exposed_2H=n2, exposed_1H_tagged=int(tag1),
                 exposed_1H_max_levels=float(np.nanmax(b1.levels)) if tag1 else None)
    ax.legend(handles=[Patch(color="#e63946", alpha=0.65, label=f"within 55 m: {n1} footprints"),
                       Patch(color="#ffb703", alpha=0.55, label=f"55-110 m: {n2} footprints"),
                       Patch(color="#90e0ef", alpha=0.5, label="street-canyon dust corridors (300 m)"),
                       Line2D([], [], color="#ff2d2d", ls="--", label="site location uncertainty (100 m)")],
              loc="lower right", fontsize=8)
    ax.set_title("C. Exposure - neighbours to inspect (labels: OSM storeys)", fontsize=12, loc="left")
    footer(fig, SRC_ESRI, SRC_OSM, "height 55 m (~18 storeys): " + SRC_PRESS)
    fig.savefig(OUT / "map_C_exposure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return b1


# ------------------------------------------------------------------ D. emergency access
SPEED = dict(motorway=60, trunk=50, primary=40, secondary=35, tertiary=30, unclassified=20,
             residential=15, service=10, living_street=10)


def map_access():
    G = nx.Graph()
    for _, r in roads.iterrows():
        hw = str(r.highway).replace("_link", "")
        v = SPEED.get(hw, 15) / 3.6
        cs = list(r.geometry.coords)
        for a, b in zip(cs[:-1], cs[1:]):
            L = math.dist(a, b)
            ka, kb = (round(a[0], 1), round(a[1], 1)), (round(b[0], 1), round(b[1], 1))
            G.add_edge(ka, kb, length=L, time=L / v, hw=hw)
    nodes = np.array(list(G.nodes))

    def nearest(p):
        return tuple(nodes[np.argmin((nodes[:, 0] - p.x) ** 2 + (nodes[:, 1] - p.y) ** 2)])

    s_node = nearest(SITE)
    lengths = nx.single_source_dijkstra_path_length(G, s_node, weight="time")
    paths = nx.single_source_dijkstra_path(G, s_node, weight="time")
    fac = ems[ems.amenity.isin(["fire_station", "hospital"])].copy()
    fac["node"] = fac.geometry.apply(nearest)
    fac["t_min"] = fac.node.map(lambda n: lengths.get(n, np.inf) / 60)
    fac["km"] = fac.node.map(lambda n: sum(G[a][b]["length"] for a, b in zip(paths[n][:-1], paths[n][1:]))
                             / 1000 if n in paths else np.nan)
    fire = fac[fac.amenity == "fire_station"].sort_values("t_min")
    hosp = fac[fac.amenity == "hospital"].sort_values("t_min").head(4)
    fig, ax = plt.subplots(figsize=(11, 10))
    half = 4200
    frame(ax, half)
    cx.add_basemap(ax, crs=CRS, source=cx.providers.Esri.WorldGrayCanvas, zoom=14, attribution="Basemap: Esri | " + ATTR_OSM)
    roads.plot(ax=ax, color="0.6", lw=0.3)
    rivers.plot(ax=ax, color="#0077ff", lw=1)
    sel = []
    for rows, col in ((fire, "#d62828"), (hosp, "#2a9d8f")):
        for _, f in rows.iterrows():
            p = paths.get(f.node)
            if p:
                ax.plot(*zip(*p), color=col, lw=2.4, alpha=0.85)
            ax.plot(f.geometry.x, f.geometry.y, "s" if col == "#d62828" else "P", ms=10, color=col, mec="w")
            if f.amenity == "fire_station":
                ax.text(f.geometry.x + 60, f.geometry.y + 60, f"{f['name'][:30]}\n{f.km:.1f} km, ~{f.t_min:.0f} min*",
                        fontsize=7.5, bbox=dict(fc="w", ec=col, alpha=0.9, pad=1.5))
            sel.append(dict(name=f["name"], type=f.amenity, km=round(f.km, 2), minutes=round(f.t_min, 1)))
    hl = "\n".join(f"{h['name'][:28]:28s} {h.km:4.1f} km  ~{h.t_min:3.0f} min*" for _, h in hosp.iterrows())
    ax.text(0.99, 0.99, "Nearest mapped hospitals\n" + hl, transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=7.5, bbox=dict(fc="w", ec="#2a9d8f", alpha=0.9))
    site_marker(ax, label=False)
    scalebar(ax, 1000)
    ax.set_title("D. Emergency access - network routes to the site", fontsize=12, loc="left")
    ax.text(0.01, 0.01, "*free-flow time from assumed road-class speeds; real response in Eastleigh traffic and crowds\n"
            "is much slower. Only facilities mapped in OSM are included (2 fire stations).",
            transform=ax.transAxes, fontsize=7.5, bbox=dict(fc="w", ec="none", alpha=0.8))
    ax.legend(handles=[Line2D([], [], color="#d62828", lw=2.4, marker="s", label="fire station route"),
                       Line2D([], [], color="#2a9d8f", lw=2.4, marker="P", label="hospital route (4 nearest)")],
              loc="upper left", fontsize=8)
    footer(fig, SRC_ESRI_GREY, SRC_OSM, "routing: networkx on OSM roads (this analysis)")
    fig.savefig(OUT / "map_D_emergency_access.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # last-mile street widths: classes within 300 m of site
    near = roads[roads.distance(SITE) < 300]
    stats.update(access=sel, last_mile_classes=near.highway.value_counts().to_dict())


# ------------------------------------------------------------------ E. high-rise screening
def map_screening():
    fig, ax = plt.subplots(figsize=(11, 10))
    half = 2000
    frame(ax, half)
    cx.add_basemap(ax, crs=CRS, source=cx.providers.Esri.WorldGrayCanvas, zoom=15, attribution="Basemap: Esri | " + ATTR_OSM)
    view = bld.cx[SITE.x - half:SITE.x + half, SITE.y - half:SITE.y + half]
    view.plot(ax=ax, color="0.55", lw=0, alpha=0.5)
    tall = view[view.levels >= 8]
    cent = tall.geometry.centroid
    hb = ax.hexbin(cent.x, cent.y, gridsize=18, extent=(SITE.x - half, SITE.x + half, SITE.y - half, SITE.y + half),
                   cmap="inferno", mincnt=1, alpha=0.55)
    tall.plot(ax=ax, column="levels", cmap="plasma", vmin=8, vmax=20, lw=0, legend=True,
              legend_kwds=dict(label="storeys (OSM-tagged, >= 8)", shrink=0.6))
    plt.colorbar(hb, ax=ax, shrink=0.6, label="tall buildings per hex (~250 m)")
    site_marker(ax, label=False)
    scalebar(ax, 500)
    n_tag = view.levels.notna().sum()
    stats.update(screen_tall_ge8=len(tall), screen_total=len(view), screen_tagged=int(n_tag))
    ax.set_title(f"E. High-rise screening - {len(tall)} mapped buildings >= 8 storeys within 2 km",
                 fontsize=12, loc="left", color="k")
    ax.text(0.01, 0.01, f"Only {n_tag} of {len(view)} footprints carry storey tags in OSM: this is a lower bound.\n"
            "A full screen would use satellite-derived heights (e.g. Open Buildings 2.5D) and county permits.",
            transform=ax.transAxes, fontsize=7.5, color="w", bbox=dict(fc="k", ec="none", alpha=0.6))
    footer(fig, SRC_ESRI_GREY, SRC_OSM)
    fig.savefig(OUT / "map_E_highrise_screening.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ exports
def export_for_web(b1):
    """Buildings within 300 m (lat/lon + levels) for the 3-D page; roads and rivers nearby."""
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    near = bld[bld.distance(SITE) < 300]
    feats = []
    for _, b in near.iterrows():
        xs, ys = b.geometry.exterior.xy
        feats.append(dict(xy=[[round(x - SITE.x, 1), round(y - SITE.y, 1)] for x, y in zip(xs, ys)],
                          lv=None if np.isnan(b.levels) else b.levels))
    rd = []
    for _, r in roads[roads.distance(SITE) < 320].iterrows():
        rd.append(dict(hw=r.highway, name=r.get("name") if isinstance(r.get("name"), str) else None,
                       xy=[[round(x - SITE.x, 1), round(y - SITE.y, 1)] for x, y in r.geometry.coords]))
    json.dump(dict(site_ll=SITE_LL, buildings=feats, roads=rd), open(OUT / "web_layers.json", "w"),
              separators=(",", ":"))
    # interactive map (local; needs internet for basemap tiles)
    m = folium.Map(location=SITE_LL, zoom_start=17, tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, Maxar", name="Satellite").add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    bl = bld[bld.distance(SITE) < 700].to_crs("EPSG:4326")[["geometry", "levels"]]
    bl["levels"] = bl.levels.fillna(-1)
    folium.GeoJson(bl, name="Buildings (OSM)",
                   style_function=lambda f: dict(color="#fff", weight=0.5, fillOpacity=0.5,
                                                 fillColor="#ff7b00" if f["properties"]["levels"] >= 8 else "#888"),
                   tooltip=folium.GeoJsonTooltip(["levels"], aliases=["storeys (-1 = untagged)"])).add_to(m)
    folium.GeoJson(rivers.to_crs("EPSG:4326")[["geometry"]], name="Rivers",
                   style_function=lambda f: dict(color="#0077ff", weight=3)).add_to(m)
    for r, c in ((H_BLDG, "#e63946"), (2 * H_BLDG, "#ffb703"), (SITE_UNCERT, "#ffffff")):
        folium.Circle(SITE_LL, radius=r, color=c, fill=False, weight=2,
                      tooltip=f"{r:.0f} m").add_to(m)
    folium.Marker(SITE_LL, tooltip="Reported site (Ninth St x 2nd Ave Eastleigh) - confirm plot",
                  icon=folium.Icon(color="red", icon="warning-sign")).add_to(m)
    m.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:8px;left:8px;z-index:9999;background:rgba(255,255,255,.88);'
        'padding:4px 8px;font:12px sans-serif;border-radius:3px">Eastleigh collapse site (preliminary) - '
        'Data: Esri World Imagery; OpenStreetMap contributors | ' + CREDIT + '</div>'))
    folium.LayerControl().add_to(m)
    m.save(str(OUT / "eastleigh_map.html"))


if __name__ == "__main__":
    map_context(); print("A done")
    map_terrain(); print("B done")
    b1 = map_exposure(); print("C done")
    map_access(); print("D done")
    map_screening(); print("E done")
    export_for_web(b1)
    json.dump(stats, open(OUT / "gis_stats.json", "w"), indent=1, default=str)
    print(json.dumps(stats, indent=1, default=str))
