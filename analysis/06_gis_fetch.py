"""
Step 6 - GIS data acquisition (open data, cached under gis/raw/).

  * OpenStreetMap via Overpass: buildings (+levels/height tags), roads, waterways,
    land use, emergency facilities (fire, hospital, police).
  * Terrain: AWS Open Data "Terrain Tiles" (Terrarium PNG, SRTM-based in Kenya),
    decoded to elevation in metres.

Site: OSM has no feature for the reported supermarket, so the site is taken as
the reported zone around the Ninth Street / 2nd Avenue Eastleigh junction.
CONFIRM THE PLOT (LR number or GPS) and update SITE below.
"""
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RAW = HERE / "gis" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = "building-collapse-forensics/1.0 (research)"


def curl(url, out, data=None):
    """curl rather than urllib: the system Python's LibreSSL fails TLS with some servers."""
    cmd = ["curl", "-sS", "-f", "-m", "200", "-A", UA, "-o", str(out), url]
    if data is not None:
        cmd[1:1] = ["--data-urlencode", f"data={data}"]
    subprocess.run(cmd, check=True)

SITE = (-1.27585, 36.85125)          # lat, lon (Ninth St x 2nd Ave Eastleigh, approx.)
CTX = 0.020                          # deg half-width, ~2.2 km
EMS = 0.045                          # deg half-width for emergency facilities, ~5 km


def bbox(lat, lon, d):
    return f"{lat - d},{lon - d},{lat + d},{lon + d}"


def overpass(name, q):
    out = RAW / f"{name}.json"
    if out.exists():
        return json.loads(out.read_text())
    for url in ("https://overpass-api.de/api/interpreter",
                "https://overpass.kumi.systems/api/interpreter"):
        try:
            try:
                curl(url, out, data=q)
            except Exception:
                out.unlink(missing_ok=True)
                raise
            res = json.loads(out.read_text())
            print(f"  {name}: {len(res['elements'])} elements")
            return res
        except Exception as e:          # try mirror
            print(f"  {url} failed: {e}")
            time.sleep(3)
    raise RuntimeError(name)


def terrain(lat, lon, d, z=15):
    """Mosaic Terrarium tiles covering bbox -> (elev array, west, north, east, south)."""
    def tile(lat, lon):
        n = 2 ** z
        x = (lon + 180) / 360 * n
        y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
        return x, y
    x0, y0 = tile(lat + d, lon - d)
    x1, y1 = tile(lat - d, lon + d)
    xs, ys = range(int(x0), int(x1) + 1), range(int(y0), int(y1) + 1)
    rows = []
    for ty in ys:
        row = []
        for tx in xs:
            f = RAW / f"terrarium_{z}_{tx}_{ty}.png"
            if not f.exists():
                url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{tx}/{ty}.png"
                curl(url, f)
            a = np.asarray(Image.open(f).convert("RGB")).astype(float)
            row.append(a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768)
        rows.append(np.hstack(row))
    elev = np.vstack(rows)
    n = 2 ** z
    lon_w, lon_e = xs[0] / n * 360 - 180, (xs[-1] + 1) / n * 360 - 180
    lat_of = lambda ty: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    lat_n, lat_s = lat_of(ys[0]), lat_of(ys[-1] + 1)
    np.savez_compressed(RAW / "dem.npz", elev=elev, bounds=[lon_w, lat_n, lon_e, lat_s])
    print(f"  DEM {elev.shape}, {elev.min():.0f}-{elev.max():.0f} m")


def main():
    lat, lon = SITE
    b = bbox(lat, lon, CTX)
    print("OSM ...")
    overpass("buildings", f'[out:json][timeout:180];way["building"]({b});out tags geom;')
    overpass("roads", f'[out:json][timeout:180];way["highway"~"^(motorway|trunk|primary|secondary|tertiary|'
                      f'unclassified|residential|service|living_street)(_link)?$"]({bbox(lat, lon, EMS)});out tags geom;')
    overpass("water", f'[out:json][timeout:180];(way["waterway"]({b});relation["waterway"="river"]({b}););out tags geom;')
    try:   # optional context layer; the public Overpass servers sometimes time out
        overpass("landuse", f'[out:json][timeout:120];way["landuse"]({b});out tags geom;')
    except RuntimeError:
        print("  landuse skipped")
    overpass("ems", f'[out:json][timeout:180];nwr["amenity"~"^(fire_station|hospital|police)$"]'
                    f'({bbox(lat, lon, EMS)});out tags center;')
    print("Terrain ...")
    terrain(lat, lon, CTX)
    json.dump({"site": SITE, "ctx_deg": CTX, "ems_deg": EMS}, open(RAW / "meta.json", "w"))


if __name__ == "__main__":
    main()
