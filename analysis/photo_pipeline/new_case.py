"""
Create a new case folder from a set of photos.

    python new_case.py cases/<case_name> /path/to/photos --name "Building X collapse" --site "..." --date YYYY-MM-DD --lat -1.28 --lon 36.85

Writes case.json (metadata, scenarios) and tags.json (one empty entry per photo, ready for
tagging against config/evidence_library.json). Photos are referenced, never copied or modified.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("case_dir"); ap.add_argument("photos")
ap.add_argument("--name", required=True); ap.add_argument("--site", default="")
ap.add_argument("--date", default=""); ap.add_argument("--credit", default="Credit: Gava Charles")
ap.add_argument("--glob", default="*.jp*g")
ap.add_argument("--lat", type=float); ap.add_argument("--lon", type=float)
ap.add_argument("--ee-project", help="Google Cloud project registered for Earth Engine (optional)")
a = ap.parse_args()

d = Path(a.case_dir); d.mkdir(parents=True, exist_ok=False)
photos = sorted(Path(a.photos).glob(a.glob))
case = {"name": a.name, "site": a.site, "event_date": a.date,
        "photo_dir": str(Path(a.photos).resolve()), "photo_glob": a.glob, "credit": a.credit,
        "photo_source": f"{len(photos)} site photographs",
        "location": {"lat": a.lat, "lon": a.lon} if a.lat is not None else None,
        "ee_project": a.ee_project,
        "record_evidence": [],
        "desk_evidence": [],
        "external_evidence": {"_doc": "Optional case-specific non-photo evidence, same term format as the library."},
        "scenarios": {"A  Photos only": ["photos"], "B  Photos + public record": ["photos", "record"],
                      "C  Photos + record + desk data": ["photos", "record", "desk"]},
        "measurements": {}}
tags = {"_doc": "For each photo: view (wide|element|aerial); observations with label, category (theme), confidence "
                "(high|medium|low), box [x0,y0,x1,y1] normalised, evidence (library item id). Low confidence = shown, not used.",
        "photos": {p.name: {"view": "", "observations": []} for p in photos},
        "redact_manual": {p.name: [] for p in photos}}
json.dump(case, open(d / "case.json", "w"), indent=1)
json.dump(tags, open(d / "tags.json", "w"), indent=1)
print(f"created {d} with {len(photos)} photos. Next: python run.py {d} --stages intake context; review out/redacted and "
      f"out/context/context.json; tag photos in tags.json; then python run.py {d}")
