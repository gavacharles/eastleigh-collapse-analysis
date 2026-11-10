# eastleigh-collapse-analysis

Python scripts, results and visualisations for a preliminary engineering analysis of the 23 October 2026 building collapse on 9th Street, Eastleigh, Nairobi, built only from public material (one eyewitness video, site photographs, press reports and open data). It is not a determination of cause and does not assign fault.

The full technical write-up is [analysis/REPORT.md](analysis/REPORT.md).

## Layout

| Path | Contents |
|---|---|
| `analysis/01_…21_*.py` | Numbered analysis steps: video kinematics, physics and PINNs, digital twin, GIS, rainfall, seismic and soils, remote sensing, Bayesian ranking, figures and animations |
| `analysis/REPORT.md` | Method, results, limitations and recommended investigations |
| `analysis/out/` | Numerical results (kinematics, physics fits, FE twin, Bayesian network) |
| `analysis/gis/` | Processed GIS layers and timelines |
| `analysis/visualisations/` | Figures, maps, interactive pages and animations ([index](analysis/visualisations/README.md)) |
| `analysis/photo_pipeline/` | Reusable pipeline for photo-based preliminary collapse assessment ([README](analysis/photo_pipeline/README.md)) |

## Not included

- The eyewitness video and anything derived from its pixels or audio, because it shows identifiable people. Measurements taken from it are included as CSV and JSON.
- The site photographs, including blurred versions (third-party material).
- Raw satellite and map tiles; the fetch scripts re-download them under their providers' terms.
- Social-media material and narrated videos.

## Running

Python 3.9+. Main packages: numpy, scipy, matplotlib, opencv-python, imageio (with imageio-ffmpeg), torch (for the PINNs), geopandas, shapely, pyproj, contextily, folium, networkx, obspy and pyvista. The photo pipeline pins its own versions in `analysis/photo_pipeline/requirements.txt`.

- Steps that need the video (01, 04, 05, 08, 14 and part of 21) expect `VID-20261023-WA0009.mp4` in the repository root. Without it, start from `analysis/out/kinematics.csv`.
- The photo pipeline expects the photographs in the repository root (`84*.jpg`); see `analysis/photo_pipeline/cases/eastleigh/case.json`. Its Earth Engine stage needs your own project ID in `ee_project`.
- All other steps run from open data or from the files in `analysis/out/`.

## Data sources

Esri World Imagery and Wayback (Maxar, Airbus and others), OpenStreetMap contributors (ODbL), SRTM-based terrain tiles, Open-Meteo ERA5, USGS ComCat, ISRIC SoilGrids, Raspberry Shake, Google Open Buildings 2.5D Temporal, Sentinel-1 and Sentinel-2 (Copernicus), and published press reports.

Credit: Gava Charles
