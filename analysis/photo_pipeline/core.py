"""Core stages: intake (inventory + privacy), evidence plates/log, measurements, run manifest."""
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
AV = "/System/Library/Fonts/Avenir Next.ttc"
COL = {"collapse_mode": (214, 55, 46), "material": (123, 44, 191), "element": (31, 106, 165),
       "construction_stage": (42, 157, 143), "neighbour": (230, 120, 0), "foundation": (120, 85, 50),
       "rescue": (0, 160, 230)}


def font(size, idx=0):
    try:
        return ImageFont.truetype(AV, size, index=idx)
    except OSError:                                    # non-macOS fallback
        return ImageFont.load_default()


class Case:
    def __init__(self, case_dir, photo_dir=None):
        self.dir = Path(case_dir).resolve()
        self.cfg = json.load(open(self.dir / "case.json"))
        self.tags = json.load(open(self.dir / "tags.json"))
        self.lib = json.load(open(ROOT / "config" / "evidence_library.json"))
        self.net = json.load(open(ROOT / "config" / "network.json"))
        self.photo_dir = Path(photo_dir) if photo_dir else (self.dir / self.cfg["photo_dir"]).resolve()
        self.out = self.dir / "out"
        self.out.mkdir(exist_ok=True)
        self.credit = self.cfg.get("credit", "")

    def photos(self):
        return sorted(self.photo_dir.glob(self.cfg.get("photo_glob", "*.jp*g")))

    def evidence_items(self, min_conf=("high", "medium")):
        """Library items supported by at least one reviewed observation of sufficient confidence."""
        sup = {}
        for name, p in self.tags["photos"].items():
            for o in p["observations"]:
                e = o.get("evidence")
                if e and o["confidence"] in min_conf:
                    sup.setdefault(e, set()).add(name)
        return {k: sorted(v) for k, v in sorted(sup.items())}

    def item(self, key):
        return self.lib["items"].get(key) or self.cfg.get("external_evidence", {}).get(key)


# ------------------------------------------------------------------ intake
def intake(case):
    import imagehash
    red = case.out / "redacted"
    red.mkdir(exist_ok=True)
    model = str(ROOT / "models" / "face_yunet.onnx")
    inv, hashes = [], {}
    for f in case.photos():
        im = Image.open(f)
        ex = im.getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        h = imagehash.phash(im)
        dup = next((n for n, hh in hashes.items() if h - hh <= 6), None)
        hashes[f.name] = h
        bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        boxes = []
        for scale in (1.0, 0.5):
            sm = cv2.resize(bgr, (int(W * scale), int(H * scale)))
            det = cv2.FaceDetectorYN.create(model, "", (sm.shape[1], sm.shape[0]), 0.6, 0.3, 5000)
            _, faces = det.detect(sm)
            for fc in (faces if faces is not None else []):
                x, y, w, hh = (fc[:4] / scale).astype(int)
                boxes.append((x, y, x + w, y + hh))
        for (x0, y0, x1, y1) in case.tags.get("redact_manual", {}).get(f.name, []):
            boxes.append((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
        for (x0, y0, x1, y1) in boxes:
            pad = int(0.25 * (x1 - x0))
            x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
            if x1 > x0 and y1 > y0:
                k = max(31, (x1 - x0) // 2 * 2 + 1)
                bgr[y0:y1, x0:x1] = cv2.GaussianBlur(bgr[y0:y1, x0:x1], (k, k), 0)
        cv2.imwrite(str(red / f.name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        inv.append(dict(file=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest(), w=W, h=H,
                        datetime=tags.get("DateTime"), camera=tags.get("Model"),
                        gps=bool(ex.get_ifd(0x8825)) if ex else False, phash=str(h), duplicate_of=dup,
                        view=case.tags["photos"].get(f.name, {}).get("view", "UNTAGGED"), redactions=len(boxes)))
    json.dump(inv, open(case.out / "inventory.json", "w"), indent=1)
    untagged = [i["file"] for i in inv if i["view"] == "UNTAGGED"]
    print(f"  {len(inv)} photos, {sum(i['redactions'] for i in inv)} redactions, duplicates: "
          f"{[i['file'] for i in inv if i['duplicate_of']]}, untagged: {untagged}")
    if untagged:
        print("  -> add these to tags.json (see templates/tags_template.json) before the evidence stage")


# ------------------------------------------------------------------ evidence plates + log
def evidence(case):
    (case.out / "plates").mkdir(exist_ok=True)
    rows = []
    for name, p in case.tags["photos"].items():
        im = Image.open(case.out / "redacted" / name).convert("RGB")
        im.thumbnail((1600, 1600))
        W, H = im.size
        d = ImageDraw.Draw(im, "RGBA")
        lw = max(3, W // 300)
        for k, o in enumerate(p["observations"], 1):
            x0, y0, x1, y1 = o["box"]
            box = [x0 * W, y0 * H, x1 * W, y1 * H]
            c = COL[o["category"]]
            d.rectangle(box, outline=c + (255,), width=lw)
            fs = max(18, W // 45)
            tw = d.textlength(str(k), font=font(fs, 2))
            d.rectangle([box[0], box[1], box[0] + tw + 14, box[1] + fs + 10], fill=c + (235,))
            d.text((box[0] + 7, box[1] + 3), str(k), font=font(fs, 2), fill=(255, 255, 255))
            rows.append(dict(photo=name, obs=k, theme=o["category"], label=o["label"], confidence=o["confidence"],
                             evidence=o.get("evidence", ""), box=json.dumps(o["box"])))
        fs = max(16, W // 70)
        lines = [f"{k}  {o['label']}  [{o['confidence']}]" for k, o in enumerate(p["observations"], 1)]
        strip = Image.new("RGB", (W, (fs + 10) * len(lines) + fs + 40), (16, 20, 24))
        sd = ImageDraw.Draw(strip)
        for i, ln in enumerate(lines):
            sd.rectangle([14, 18 + i * (fs + 10), 26, 14 + i * (fs + 10) + fs], fill=COL[p["observations"][i]["category"]])
            sd.text((36, 12 + i * (fs + 10)), ln, font=font(fs), fill=(235, 235, 235))
        sd.text((14, strip.height - fs - 14), f"Photo {name} (faces/plates blurred). Engineer-reviewed tags.  {case.credit}",
                font=font(int(fs * 0.8)), fill=(170, 170, 170))
        plate = Image.new("RGB", (W, H + strip.height))
        plate.paste(im, (0, 0)); plate.paste(strip, (0, H))
        plate.save(case.out / "plates" / f"plate_{name}", quality=90)
    with open(case.out / "evidence_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    used = case.evidence_items()
    allref = case.evidence_items(min_conf=("high", "medium", "low"))
    json.dump(dict(used=used, excluded_low_confidence=sorted(set(allref) - set(used))),
              open(case.out / "evidence_support.json", "w"), indent=1)
    for k, v in used.items():
        print(f"  {k} ({len(v)}): {case.item(k)['desc']}")
    for k in sorted(set(allref) - set(used)):
        print(f"  [excluded, low confidence] {k}")


# ------------------------------------------------------------------ measurements
def measure(case):
    from scipy.signal import find_peaks
    cfg = case.cfg.get("measurements", {})
    res = {}
    lc = cfg.get("layer_count")
    if lc:
        im = cv2.imread(str(case.out / "redacted" / lc["photo"]))
        h, w = im.shape[:2]
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        white = ((hsv[..., 1] < 45) & (hsv[..., 2] > 185)).astype(float)
        counts = []
        for x0 in np.arange(lc["x_range"][0], lc["x_range"][1] + 1e-9, 0.05):
            prof = np.convolve(white[:, int(x0 * w):int((x0 + 0.08) * w)].mean(1), np.ones(9) / 9, "same")
            seg = prof[int(lc["y_range"][0] * h):int(lc["y_range"][1] * h)]
            pk, _ = find_peaks(seg, height=0.06, distance=int(0.02 * h))
            counts.append(len(pk))
        res["slab_layers_auto_count"] = int(np.median(counts))
        res["slab_layers_auto_count_range"] = [int(min(counts)), int(max(counts))]
        res["layer_count_photo"] = lc["photo"]
    if "neighbour_storeys_min" in cfg:
        res["building_storeys_min"] = cfg["neighbour_storeys_min"]
    res["not_measurable"] = ("Pile height/volume: needs 20+ overlapping photos or a drone survey (structure-from-motion). "
                             "Bar diameters, tie spacing, cover: need a scale reference in element close-ups.")
    json.dump(res, open(case.out / "measurements.json", "w"), indent=1)
    print("  ", res)


# ------------------------------------------------------------------ manifest
def manifest(case, stages):
    import importlib.metadata as md
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    vers = {}
    for pkg in ("opencv-python-headless", "numpy", "scipy", "matplotlib", "Pillow", "ImageHash"):
        try:
            vers[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            vers[pkg] = None
    m = dict(run_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), stages=stages,
             python=platform.python_version(), packages=vers,
             config_sha256={f: sha(ROOT / "config" / f) for f in ("network.json", "evidence_library.json")},
             case_sha256={f: sha(case.dir / f) for f in ("case.json", "tags.json")},
             code_sha256={f.name: sha(f) for f in sorted(ROOT.glob("*.py"))},
             face_model="OpenCV Zoo YuNet 2023mar", inputs=[(i["file"], i["sha256"][:16]) for i in
                                                            json.load(open(case.out / "inventory.json"))])
    json.dump(m, open(case.out / "run_manifest.json", "w"), indent=1)
