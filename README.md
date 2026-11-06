# Eastleigh building collapse (Nairobi, 23 October 2026): forensic analysis

This is a preliminary, evidence-limited engineering analysis of the collapse of a multi-storey building under construction on 9th Street, Eastleigh, Nairobi. It uses only public material: one eyewitness phone video, site photographs, press reports and open data. It is **not a determination of cause and does not assign fault.** Drawings, material tests and a site investigation are needed to confirm or overturn it.

The full technical write-up is [analysis/REPORT.md](analysis/REPORT.md).

## Main findings

- **Base-first failure.** The first suspended slab dropped into the ground storey while every floor above stayed square.
- **The street corner led.** It fell about 0.6 m before the centre of the building moved. The failure crossed the ground storey in under a second.
- **Low resistance.** Each storey resisted only about 0.4–0.5 times the weight falling on it. Crushing took 4.5–5 s, against about 3 s for free fall.
- **Height.** The measured descent rules out 12 storeys and is consistent with 16–18.
- **Low reserve plus a local trigger.** In a parametric digital twin, a frame of design quality survives losing two corner columns. Only low-reserve frames (about 1.6–2.3 times the building's weight) both stand intact and then fail across the whole ground storey, as the video shows.
- **Cause ranking** (Bayesian network, 90% ranges). The failure started in the ground storey: 0.83 (0.72–0.89). Underlying causes: extra storeys or overload 0.72, deficient materials 0.54, design deficiency 0.43, foundation 0.13. Explosion, earthquake or wind, and top-down construction failure are all below 0.01. These are not shares of blame, and a collapse usually has more than one cause.

## Repository layout

| Path | Contents |
|---|---|
| `analysis/01_…21_*.py` | Numbered analysis steps: video kinematics, physics and PINNs, digital twin, GIS, rainfall, seismic and soils, remote sensing, Bayesian ranking, figures and animations |
| `analysis/REPORT.md` | Full method, findings, limitations and recommended investigations |
| `analysis/out/` | Numerical results (kinematics, physics fits, FE twin, Bayesian network) |
| `analysis/gis/` | Processed GIS layers and timelines |
| `analysis/visualisations/` | Figures, maps, interactive pages and animations ([index](analysis/visualisations/README.md)) |
| `analysis/photo_pipeline/` | A reusable pipeline for photo-based preliminary collapse assessment, with a Bayesian network, field-check prioritisation and validation against 12 past collapses ([README](analysis/photo_pipeline/README.md)) |

## What is deliberately not included

- **The eyewitness video and anything derived from its pixels or audio** (frames, annotated stills, side-by-side animations, audio). The footage shows identifiable people. Measurements taken from it are included as CSV and JSON.
- **The site photographs,** including face-blurred versions. They are third-party material.
- **Raw satellite and map tiles.** The fetch scripts re-download them under their providers' terms.
- Social-media material and narrated videos.

## Reproducing the analysis

Python 3.9+. The main packages are numpy, scipy, matplotlib, opencv-python, imageio (with imageio-ffmpeg), torch (for the PINNs), geopandas, shapely, pyproj, contextily, folium, networkx, obspy and pyvista. The photo pipeline pins its own versions in `analysis/photo_pipeline/requirements.txt`.

- **Steps that need the video** (01, 04, 05, 08, 14 and part of 21) expect `VID-20261023-WA0009.mp4` in the repository root. Without it, start from the measurements in `analysis/out/kinematics.csv`.
- **The photo pipeline** expects the photographs in the repository root (`84*.jpg`). See `analysis/photo_pipeline/cases/eastleigh/case.json`. The Earth Engine stage needs your own Google Earth Engine project ID in `ee_project`.
- **Everything else** (physics fits, digital twin, Bayesian network, GIS, rainfall, seismic) runs from open data or from the files in `analysis/out/`.

## Data sources

Esri World Imagery and Wayback (Maxar, Airbus and others), OpenStreetMap contributors (ODbL), SRTM-based terrain tiles, Open-Meteo ERA5, USGS ComCat, ISRIC SoilGrids, Raspberry Shake, Google Open Buildings 2.5D Temporal, Sentinel-1 and Sentinel-2 (Copernicus), and press reports from Kenyans.co.ke, Africanews/AP and PBS NewsHour. Each figure states its own sources.

## Status and limitations

All sections, reinforcement and loads in the digital twin are assumed, because no drawings were available. The plot location is known only to within about 100 m. Metric scale assumes 3.0 m storeys. Probabilities come from an expert-elicited, partly calibrated Bayesian network. They should be challenged, and the scripts make them easy to change.

Credit: Gava Charles
