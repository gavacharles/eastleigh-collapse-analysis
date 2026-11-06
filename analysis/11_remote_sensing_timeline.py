"""
Step 11 - Remote sensing: construction timeline from archived very-high-resolution imagery.

Esri World Imagery Wayback keeps every release of the global basemap since 2014.
For each release, the metadata service gives the acquisition date, sensor and
resolution of the image covering the site. Distinct acquisitions are mosaicked
(zoom 19, ~0.3 m) around the site and written as a dated chip series, which is
then read for construction progress (and shadow-based height where visible).
"""
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from credit import CREDIT, stamp

HERE = Path(__file__).resolve().parent
RAW = HERE / "gis" / "raw" / "wayback"
RAW.mkdir(parents=True, exist_ok=True)
OUT = HERE / "gis" / "timeline"
OUT.mkdir(parents=True, exist_ok=True)
SITE = tuple(json.load(open(HERE / "gis" / "raw" / "meta.json"))["site"])
Z = 19
UA = "building-collapse-forensics/1.0 (research)"


def curl(url, out=None):
    cmd = ["curl", "-sS", "-f", "-m", "60", "-A", UA, url]
    if out:
        cmd[1:1] = ["-o", str(out)]
        subprocess.run(cmd, check=True)
        return None
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def tile_xy(lat, lon, z=Z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def metadata(item):
    url = (f"{item['metadataLayerUrl']}/identify?geometry={SITE[1]},{SITE[0]}&geometryType=esriGeometryPoint"
           f"&sr=4326&layers=all&tolerance=2&mapExtent={SITE[1]-.01},{SITE[0]-.01},{SITE[1]+.01},{SITE[0]+.01}"
           f"&imageDisplay=800,800,96&returnGeometry=false&f=json")
    try:
        res = json.loads(curl(url)).get("results", [])
    except Exception:
        return None
    if not res:
        return None
    a = res[0]["attributes"]
    return dict(date=a.get("SRC_DATE"), res=a.get("SRC_RES"), sensor=a.get("SRC_DESC"),
                vendor=a.get("NICE_DESC"), product=a.get("NICE_NAME"))


def get_tile(release, z, ty, tx):
    """Tile at zoom z; if the release has none at that zoom, upsample the parent tile."""
    for dz in range(0, 3):
        zz, py, px = z - dz, ty >> dz, tx >> dz
        f = RAW / f"{release}_{zz}_{py}_{px}.jpg"
        if not f.exists():
            try:
                curl(f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/"
                     f"default028mm/MapServer/tile/{release}/{zz}/{py}/{px}", f)
            except subprocess.CalledProcessError:
                continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        if dz:
            k = 2 ** dz
            sub = 256 // k
            oy, ox = (ty - (py << dz)) * sub, (tx - (px << dz)) * sub
            img = cv2.resize(img[oy:oy + sub, ox:ox + sub], (256, 256), interpolation=cv2.INTER_CUBIC)
        return img
    return np.full((256, 256, 3), 200, np.uint8)


def mosaic(release, n=3):
    x, y = tile_xy(*SITE)
    cx, cy = int(x), int(y)
    rows = []
    for ty in range(cy - n // 2, cy + n // 2 + 1):
        row = []
        for tx in range(cx - n // 2, cx + n // 2 + 1):
            row.append(get_tile(release, Z, ty, tx))
        rows.append(np.hstack(row))
    img = np.vstack(rows)
    # site pixel position inside the mosaic
    px = int((x - (cx - n // 2)) * 256)
    py = int((y - (cy - n // 2)) * 256)
    return img, (px, py)


def main():
    cfg = json.load(open(HERE / "gis" / "raw" / "waybackconfig.json"))
    items = sorted(cfg.items(), key=lambda kv: kv[1]["itemTitle"])
    cache = RAW / "metadata.json"
    meta = json.load(open(cache)) if cache.exists() else {}
    todo = [(k, v) for k, v in items if k not in meta]
    with ThreadPoolExecutor(8) as ex:
        for (k, v), m in zip(todo, ex.map(lambda kv: metadata(kv[1]), todo)):
            meta[k] = dict(m or {}, release=v["itemTitle"][-11:-1])
    json.dump(meta, open(cache, "w"), indent=1)
    # first release carrying each distinct acquisition
    seen, uniq = set(), []
    for k, v in items:
        m = meta.get(k, {})
        key = (m.get("date"), m.get("sensor"))
        if m.get("date") and key not in seen:
            seen.add(key)
            uniq.append((k, m))
    uniq.sort(key=lambda km: km[1]["date"])
    print(f"{len(items)} releases -> {len(uniq)} distinct acquisitions over the site")
    chips, log, hashes = [], [], set()
    for k, m in uniq:
        img, (px, py) = mosaic(k)
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in hashes:                 # same pixels re-published under another date
            continue
        hashes.add(h)
        d = m["date"]
        label = f"{d[:4]}-{d[4:6]}-{d[6:]}  {m.get('sensor', '')} {m.get('res', '')} m"
        chip = img.copy()
        cv2.circle(chip, (px, py), int(100 / 0.3 * 0.5), (60, 60, 255), 2)        # ~50 m radius ring
        cv2.drawMarker(chip, (px, py), (60, 60, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.rectangle(chip, (0, 0), (chip.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(chip, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(OUT / f"chip_{d}.jpg"), chip, [cv2.IMWRITE_JPEG_QUALITY, 88])
        chips.append((d, chip))
        log.append(dict(release_id=k, acquired=d, sensor=m.get("sensor"), res_m=m.get("res"),
                        vendor=m.get("vendor"), product=m.get("product"), first_release=m["release"]))
        print(" ", label)
    json.dump(log, open(OUT / "timeline.json", "w"), indent=1)
    # contact sheet
    th = [cv2.resize(c, (384, 384)) for _, c in chips]
    cols = 4
    while len(th) % cols:
        th.append(np.full_like(th[0], 255))
    sheet = np.vstack([np.hstack(th[i:i + cols]) for i in range(0, len(th), cols)])
    sheet = stamp(sheet, "Esri World Imagery Wayback archive (Maxar / Vantor / Airbus imagery; dates from Esri metadata)",
                  "site marker: Ninth St x 2nd Ave Eastleigh, ring ~50 m", scale=0.5)
    cv2.imwrite(str(OUT / "timeline_contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", OUT / "timeline_contact_sheet.jpg")


if __name__ == "__main__":
    main()
