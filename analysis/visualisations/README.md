# Eastleigh collapse: visualisation index

**Site:** LR No. 36/II/40, 9th Street, Eastleigh, Kamukunji Sub-County, Nairobi. **Event:** 23 October 2026, about 13:40 EAT.

**Status:** preliminary, not a determination of cause.

Every item shows its data sources and "Credit: Gava Charles".

The eyewitness footage shows identifiable bystanders, so this repository contains no video frames, footage or audio. Items built from the footage are listed under "Not in this repository".

## 01 GIS maps (`01_GIS_maps/`)

| File | What it shows | Data |
|---|---|---|
| `map_A_context.png` | Site, streets and OSM buildings on satellite imagery; storey tags where mapped | Esri World Imagery, OSM |
| `map_B_terrain_hydrology.png` | Hillshade, 5 m contours, rivers and drains; profile from the site to the nearest river | SRTM-based terrain tiles, OSM |
| `map_C_exposure.png` | Buildings within 55 m (1 × height) and 110 m; street-canyon dust corridors | Esri, OSM |
| `map_D_emergency_access.png` | Network routes from fire stations and the nearest hospitals | Esri, OSM, networkx |
| `map_E_highrise_screening.png` | Mapped buildings of 8+ storeys within 2 km, with density | Esri, OSM |
| `fig_rainfall.png` | Daily rainfall July–October 2026 | Open-Meteo |
| `eastleigh_map.html` | Interactive map with switchable basemaps and layers (needs internet) | Esri, OSM |
| `eastleigh_gis_atlas.html` | Atlas of plates A–E plus a remote-sensing programme table | as above |

## 02 Remote sensing, satellite and sensors (`02_remote_sensing_satellite_sensors/`)

| File | What it shows | Data |
|---|---|---|
| `timeline_contact_sheet.jpg` | 12 dated satellite acquisitions of the site, 2011–2026 | Esri Wayback (Maxar, Vantor, Airbus) |
| `archive_imagery_chips/chip_YYYYMMDD.jpg` | Each acquisition at full resolution (~0.3 m) | Esri Wayback |
| `fig_vertical_growth.png` | Shadow-based vertical-growth index of the block | Esri Wayback |
| `fig_seismic_candidate.png` | Citizen seismometer record, 13:20–14:00 EAT; weak candidate at 13:36 | Raspberry Shake AM.R8A6C |

## 03 3-D reconstructions (`03_3D_reconstruction_animations/`)

| File | What it shows |
|---|---|
| `eastleigh_3d.html` | Interactive 3-D scene: OSM street block, collapse replay, camera presets |
| `eastleigh_reconstruction.html` | Interactive 2-D elevation replay with live kinematics |

## 04 Physics and advanced structural mechanics (`04_physics_and_structural_mechanics/`)

| File | What it shows |
|---|---|
| `fig_kinematics_pinn.png` | Measured descent vs crush-up ODE and PINNs; velocity; identified resistance; initiation at the corner |
| `fig_physics_energy.png` | Energy budget (potential, kinetic, dissipated), resistance vs weight, moving mass, momentum (16 and 18 storeys) |
| `fig_fe_mechanics.png` | Corner-loss pushdown with energy-based dynamic capacity; column crushing (peak vs mean, measured band); 3-D cascade by scenario; failure envelope |
| `fig_twin_cascade.png` | 2-D alternate-path cascade screening (early model) |
| `fig_reserve_map.png` | Axial load ratio P/(f′c Ag) of every column, 30 vs 15 MPa; ground-storey reserve P0/P (corner 3.1×, interior 1.1×) |
| `fig_sudden_loss.png` | Single-mass overshoot under sudden column loss (2.0×/1.85×), and frame vs catenary bridging |
| `fig_resistance_vs_weight.png` | Storey crushing force against the weight of the falling block: why it could not stop |
| `fig_height_crosscheck.png` | Measured descent vs crush-up model for 12, 14, 16 and 18 storeys (12 rejected) |
| `fig_failure_envelope_map.png` | Twin failure envelope: holds / zips like the video / could not have stood |
| `fig_debris_footprint.png` | Location (±100 m) plus a to-scale plan and section of the expected debris pile (12–19 m) |
| `anim_pushdown_hinges.mp4/.gif`, `fig_pushdown_hinges_final.png` | Plastic hinges forming over the lost corner, synced with the capacity curve |
| `anim_column_crushing.mp4/.gif` | A ground-storey column crushing (cover spall → bar buckling → core crush), synced with force–shortening |

## 05 Frame-by-frame analysis (`05_frame_by_frame_analysis/`)

| File | What it shows |
|---|---|
| `frame_metrics.csv`, `kinematics.csv` | Per-frame measurements and model values |
| `fig_audio_timeline.png` | Spectrogram and audio level against the kinematics |

## 06 Frame-by-frame animation (`06_frame_by_frame_animation/`)

| File | What it shows |
|---|---|
| `collapse_reconstruction.gif` | Physics-driven elevation and live plots (no footage) |

## 07 Digital twin (`07_digital_twin/`)

| File | What it shows |
|---|---|
| `fig_twin_3d_stages.jpg` | 3-D FE twin in four stages: intact demand/capacity, corner pair lost (deformed ×40), cascade step 1, full ground-storey loss |
| `R2_structural_twin.mp4/.gif` | Animated twin: column-by-column failure, bridging deformation, storey-by-storey crush-up, live plots |
| `R2_structural_twin_filmstrip.jpg` | Key frames |

## 08 Bayesian cause ranking (`08_bayesian_cause_ranking/`)

| File | What it shows |
|---|---|
| `fig_bayes_causes.png` | Prior vs posterior with 90 % intervals; most probable combinations; influence of each evidence item |
| `fig_bayes_sensitivity.png` | How each assumed probability moves the three leading causes |

## Not in this repository

- The eyewitness video, its frames, audio and anything that shows footage (annotated frames, frame viewer, optical-flow figure, side-by-side animations).
- The site photographs, including the face-blurred versions and photo plates.
- Raw satellite and map tiles (re-download them with `06_gis_fetch.py` and the photo pipeline's context stage).
- Narrated story, social-media cuts, covers and cards.

## Retired

- R1 (eyewitness-view 3-D) and R3 (aerial 3-D): removed because they did not render well.
- The earlier six-scene story was replaced by the narrated story.

Scripts that produce each item are listed in `../REPORT.md`; they write to `analysis/out/` and `analysis/gis/` by default.
