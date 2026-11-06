# Eastleigh (Nairobi) building collapse, 23 Oct 2026: preliminary video-based failure analysis

**Status:** preliminary, evidence-limited. Based on one 9.9 s eyewitness clip (`VID-20261023-WA0009.mp4`, 480×864, 30 fps, WhatsApp-compressed, hand-held with zoom and pan) and early press reports. This is not a determination of cause.

**Site:** LR No. 36/II/40, 9th Street, Eastleigh (off Yusuf Haji Ave), Kamukunji Sub-County, Nairobi City County. No parcel geometry is publicly available, so maps use the Ninth Street / 2nd Avenue junction ±100 m.

**Context from reports (cross-checked):** under construction. Reported height varies: 12 storeys (Reuters, Capital FM), 16 (The Standard, Dawan Africa), 18 (Kenyans.co.ke, AP). A storey count on the video gives about 16 (±2), and the collapse kinematics reject 12 (section 17). Governor Sakaja said the building had construction approval but no occupation certificate, and that the **ground-floor shops were already occupied** without one. There is no report that a shop was being opened or fitted out at the time. Residents reportedly saw cracks beforehand. Time was about 13:40. About 18 workers and some food sellers were reported unaccounted for.

---

## 1. Findings

| # | Finding | Basis | Confidence |
|---|---|---|---|
| F1 | **Failure began in the lowest storey (L0, ground/shop level), not at height.** The first suspended slab (L1) dropped into L0 while everything above stayed geometrically intact. | Visual (frames 0–56); CV kinematics | High |
| F2 | **It started at the street-corner (left-in-frame) end.** The L1 slab at that end dropped about 0.45 m relative to the centre by 0.4 s and about 0.6 m by 0.8 s, before the centre of the building moved measurably. | Same-façade differential optical flow (zoom- and pan-invariant) | High for the sequence; ±0.1–0.15 m on magnitude |
| F3 | **The failure then spread along L0 within about 1 s** (a "zipper"). The whole superstructure began descending at t0 ≈ 0.7–0.85 s, as one near-rigid block, almost vertically. | CV tracking; ODE fit | High |
| F4 | **Air and dust jetted horizontally out of L0 at 1.3–2.0 s.** That means the L0 volume was being crushed. Ladders standing on the ground stayed in place while the slab came down onto them. | Visual | High |
| F5 | **Rubble, ladders and partly missing blockwork were visible at L0 at t = 0.** Their purpose is unknown: it could be construction debris, repairs, or removal of walls. The ground-floor shops were occupied (Governor). | Visual; press | High that the material was there; its purpose is unverified |
| F6 | **The collapse was resisted, not free-falling.** Resistance ≈ 0.38 × moving weight early (1.5–2.2 s), rising to ≈ 0.9–1.1 at 3.8–4.9 s. Peak velocity ≈ 11–12 m/s at about 4 s. Total crush ≈ 4.5–4.8 s, against 2.9 s for free fall over the same drop. | PINN inversion plus classical ODE | Medium (scale assumes 3.0 m storeys) |
| F7 | **The descent fits crush-up (base-first progressive crushing) physics.** A constant-resistance model is rejected: χ²/dof 3.4 against 0.37 for crush-up, and 0.22 for crush-up with column capacity tapering up the height. The fitted mean dynamic crushing force per storey is ≈ 8 storey-weights (sensitivity range 6.8–10.6). | Model comparison | Medium |
| F8 | A tall slender remnant was briefly visible in the dust at about 5.3 s. It could be a lift or stair core standing while the floors stripped off around it. | Visual, single frame | Low |

## 2. Failure sequence (video time)

1. **Before t = 0 (pre-event):** L0 is being altered or worked on (rubble, ladders, removed blockwork). Cracks were reportedly observed earlier.
2. **0–0.8 s (local failure):** vertical load-bearing elements at the street-corner end of L0 lose axial capacity. The L1 slab at that end drops about 0.6 m. Load redistributes to the neighbouring columns.
3. **0.7–1.4 s (zipper across L0):** neighbouring L0 columns are overloaded in turn and fail in sequence across the frontage. The whole building above L1 loses support.
4. **1–2.3 s (global onset):** the superstructure drops as a rigid block, accelerating at about 0.6 g net (resistance ≈ 0.4). The L0 storey is squashed, and air and dust are expelled sideways.
5. **2.3–5+ s (crush-up):** the block's lower storeys are progressively crushed against the debris. Velocity peaks at about 11–12 m/s, then resistance approaches weight as the rubble pile builds. The building ends as a compact debris pile, mostly within its footprint, with a dust surge down the street.

## 3. Hypotheses tested against the evidence

| Hypothesis | Fits the evidence? | Notes |
|---|---|---|
| **H1. Loss of L0 load-bearing elements at the corner end** (removed, cut, chased or damaged; or local overload) | **Strongly consistent** | Explains F1–F5. The digital twin shows this mechanism zipping across L0 (S2, S4, S5). |
| **H2. L0 columns globally under-strength** (low as-built concrete strength, under-design, more storeys than designed, inadequate detailing) | **Consistent; likely an enabling condition** | Does not by itself explain why the corner went first. In the twin, design-quality columns with one corner column removed are borderline (adjacent DCR = 1.00, holds); halving f′c turns the same defect into a full cascade. **H1 + H2 fits best.** |
| **H3. Foundation failure or differential settlement** (corner footing, soil, adjacent excavation) | **Partly consistent; cannot be excluded** | Would also drop the corner. But visible crushing of L0 and the lateral jets point to failure in the columns above ground. In the twin, 50 mm of settlement alone holds (DCR 0.72). Larger settlement or footing punching remain possible. |
| H4. Top-down failure (construction overload or formwork collapse at upper floors) | **Inconsistent** | The upper floors stay intact until their support is removed from below. |
| H5. Explosion (gas or deliberate) | **Inconsistent** | No flash, no fireball, no ejection from a pressure front before motion. Initiation is slow (about 1 s), and debris was already at the base. |
| H6. Lateral action (earthquake or wind) | **Inconsistent with the video** | No sway or toppling, and the descent is near-vertical. Check seismic records to close this out. |
| H7. Vehicle or plant impact | No evidence | Nothing visible at the corner. |

**Most consistent explanation:** local loss of axial capacity in ground-storey vertical elements at the street-corner end, for reasons the video cannot show (removal or damage of elements, or local overload of weak columns). It acted on a ground storey with limited reserve capacity, so the failure cascaded across L0 in about 1 s and set off a base-first crush-up collapse of the whole building. A foundation contribution at the corner cannot be ruled out from the video.

## 4. Methods

| Layer | What was done | Script |
|---|---|---|
| **Computer vision** | Lucas–Kanade optical flow with forward–backward checks. All measurements are relative: façade vs static ground wall, normalised by a 6-storey ruler on the façade (zoom-invariant); left vs centre at the same level (tilt). Roof phase uses a similarity transform estimated from static street and background points. Pixels are converted to metres using the storey pitch measured on the façade (autocorrelation), assuming 3.0 m storeys. | `01_extract_kinematics.py` |
| **Physics + PINNs** | Equation of motion z″ = g(1 − ρ), with ρ = resistance / moving weight. (1) Classical least-squares fit of a Bažant-type crush-up model, with 27-case sensitivity on storey height, compaction ratio and storey count. (2) PINN-A: free-form inverse PINN learning ρ(t) as a network. (3) PINN-B: crush-up-constrained PINN with a single learnable crushing coefficient k. The PINNs also bridge the 2.3–3.8 s gap where the camera pans away. Each method uses an ensemble of 5 seeds. | `02_physics_pinn.py` |
| **Digital-twin seed** | Parametric 18-storey 2-D RC frame (street elevation) analysed by the direct stiffness method, with cracked stiffness and a P–M envelope. GSA-style alternate-path cascade (DAF 2) across seven scenarios: design vs poor concrete, column removal, section loss, footing settlement. **All sections and loads are assumed.** | `03_digital_twin_cascade.py` |
| Evidence panel | Annotated key frames | `04_evidence_figure.py` |
| Animation | Video synced with physics-driven elevation and live plots (`out/collapse_reconstruction.mp4`, `.gif`) | `05_animate.py` |

Outputs are in `out/`: `fig_evidence_frames.jpg`, `fig_kinematics_pinn.png`, `fig_twin_cascade.png`, `kinematics.csv`, `physics_results.json`, `twin_results.json`.

## 5. Limitations

- **One camera.** Hand-held, compressed and dust-obscured. Absolute metres depend on the assumed 3.0 m storey height (±10 % scales F2 and F6 proportionally). The left-end measurement may include some movement of netting or canopy.
- **The 2.3–3.8 s window is not observed.** The physics models bridge it; it is not measured.
- **The digital twin is a seed, not a model of this building.** Geometry, sections, reinforcement and loads are generic assumptions. It demonstrates a mechanism and its sensitivity. It does not estimate this building's actual capacity.
- **The video cannot show the cause of the capacity loss.** It cannot distinguish removed walls or columns, poor concrete, corroded or inadequate reinforcement, or a footing defect. Physical evidence is needed.

## 6. Recommended next steps (to turn this into a full forensic digital twin)

1. **Preserve the evidence:** L0 column and wall remnants at the street-corner end, especially fracture surfaces, cut or chased bars, and any jacking or propping. Debris should be recovered in mapped zones.
2. **Collect every other video, CCTV feed and drone image.** Multi-view footage allows 3-D reconstruction (structure-from-motion) and proper metric kinematics, removing the storey-height assumption.
3. **Obtain documents:** approved drawings and structural calculations, the approved vs built storey count, NCA and county inspection records, and contractor logs of the works at L0 (what was removed or modified, and when).
4. **Test materials:** cores for f′c, rebar size, spacing and cover from the debris, carbonation and chloride, and mill certificates.
5. **Investigate foundations and ground:** footing exposure at the corner, soil investigation, adjacent excavations or drainage, and any pre-event crack or settlement reports.
6. **Upgrade the twin:** a 3-D nonlinear model (OpenSees / ETABS / Abaqus) calibrated to the surveyed geometry and tested materials, run through the same cascade analysis. Then fit the PINN to the multi-view kinematics to identify the real crushing resistance.
7. **ML on the debris field:** segmentation of drone orthophotos (column stubs, slab stacking, debris-throw direction) to confirm the collapse direction and initiation zone.

## 7. Structural mechanics of the sequence

**Load path and reserve.** Gravity load runs from slab to beams (or flat slab) to columns to footings, so the L0 columns carry almost the whole weight of the building, W. Their safety margin can be read as the axial-load ratio n = P / (f′c·Ag). Typical design keeps n at about 0.3–0.4 under service load. If the as-built f′c is half the design value, n roughly doubles and little reserve is left. The twin shows exactly this sensitivity: S3 holds, S4 cascades.

**Initiation: axial crushing at L0.** A gravity-loaded RC column that loses capacity does so by cover spalling, then buckling of the longitudinal bars between widely spaced or poorly anchored ties, then crushing of the core. The failure is brittle when confinement is poor. Capacity can be lost directly if columns are cut or chased (smaller Ag, possibly cut bars), or indirectly. In frames like this, blockwork infill often ends up carrying part of the gravity load as the concrete creeps, and it braces the columns; if that infill is removed or damaged, the columns pick up the load suddenly. The video shows blockwork partly missing at L0, but not why.

**Redistribution and the zipper.** When a column is lost, the column line above it hangs. Its load has to cross a double span through Vierendeel (frame) action of the beams and slabs above. Catenary action could add capacity only if the bottom bars run continuously through the columns, which is often not the case. Sudden loss of a member applies the redistributed load dynamically: for an undamped single-degree-of-freedom system, the peak response is up to 2× the static increment. Neighbouring columns can therefore see 1.5–2× their previous load. With n ≈ 0.6–0.8, that exceeds capacity, and they fail in turn. Each stage takes roughly a quarter period of the local vertical mode (about 0.1–0.2 s), which is consistent with the whole L0 zipper finishing in about 0.5–0.8 s. If the floors are flat slabs, punching shear at the neighbouring columns is an alternative propagation path.

**Global collapse: an energy balance.** Once L0 is gone, the upper block of mass M descends. Each storey crushed releases potential energy of about M·g·(1−λ)h and dissipates about F_c·(1−λ)h. The collapse accelerates while F_c < M·g, which is ρ < 1. The measured ρ ≈ 0.4 early is therefore what makes the collapse unstoppable. Peak static column capacity is 2–3 W, but the mean post-peak crushing resistance of RC (buckled bars, rubblised concrete) is only about 15–40 % of peak, which is below the weight of the falling block. This is why loss of a whole storey in a tall RC building is rarely arrested, and why robustness design concentrates on preventing it: ties, continuity of bottom bars, key-element design (EN 1991-1-7, BS 8110 tie provisions).

**Why crush-up, not crush-down.** When failure starts at the base, the moving block meets the ground and the rubble, so its own lowest storeys are crushed one by one (Bažant–Verdure crush-up). The moving mass falls while the per-storey resistance stays similar, so ρ rises over time: about 0.4 at first, about 1 near the end. The measurements show exactly this trend, and the constant-ρ model fails. The sideways dust jets are air expelled as each storey's volume collapses in a few tenths of a second.

## 8. Using GIS

| GIS layer or technique | What it tests | Hypotheses |
|---|---|---|
| **Historical very-high-resolution imagery** (Google Earth history, Maxar/Planet), with change detection and building height from shadow length | Construction timeline, storeys added per month, and therefore concrete age when loaded. Built height vs approved height. | H2 |
| **Parcel, zoning and permit layers** (Nairobi County e-permit records, Ardhisasa parcels, zoning policy) | Approved storeys and plot ratio vs built. Same developer or contractor elsewhere in the city. | H2, regulatory |
| **InSAR time series** (Sentinel-1, free, about 12-day revisit, PS/SBAS) | Mm-scale settlement of the site or neighbourhood in the months before the collapse. Borehole-driven subsidence. | H3 |
| **Geology, soils, hydrology** (geological map, black-cotton soil extent, riparian lines of the Mathare and Nairobi rivers, borehole logs) | Expansive or soft soils and groundwater at the corner footing. | H3 |
| **Post-event drone orthophoto + DSM** (photogrammetry) | Debris-pile height and volume, which calibrates the compaction ratio λ in the physics model. Spread direction confirms a vertical, in-footprint collapse. Voids for search and rescue. | Model validation; rescue |
| **3-D city model** (CityGML / 3D Tiles; OSM, Google Open Buildings 2.5D) | Places the digital twin in context: neighbours at risk from debris and dust, access routes for responders in narrow streets. | Response; risk |
| **City-wide screening** (ML on building footprints, heights, imagery, permits) | Ranks similar high-rise buildings in Eastleigh and Nairobi for priority inspection. | Prevention |

## 9. GIS results (site taken as Ninth St × 2nd Ave Eastleigh, ±100 m; plot to be confirmed)

| Finding | Value | Implication |
|---|---|---|
| Site elevation and setting | ≈1,647 m, on the interfluve between the Mathare and Nairobi river valleys; terrain slope ≈3° | No floodplain or steep slope. Terrain gives no foundation trigger. |
| Nearest watercourse | Mathare River, ≈1.3 km away and ≈27 m lower; no mapped drain at the site | Same as above. |
| Antecedent rain | 4 mm in 7 days, 45 mm in 30 days (start of the short rains) | No wet-ground trigger. H3, if real, needs a soil or excavation cause. |
| Neighbours exposed | 34 footprints within 55 m (1 × height), 56 more within 110 m; tagged neighbours of 9–13 storeys | Inspection list for the adjoining buildings. |
| Emergency access | 2 mapped fire stations, ≈4.3 and ≈7.0 km by road; hospitals within ≈1 km; last mile on residential and service streets | Crowds and narrow streets slow the response, as reported. |
| District screening | ≥314 mapped buildings of 8 or more storeys within 2 km; only 318 of 17,596 footprints have storey tags | Lower bound. Needs satellite heights and permits to complete. |

Outputs: `gis/map_A…E*.png`, `gis/fig_rainfall.png`, `gis/eastleigh_map.html` (interactive, switchable basemaps) and `gis/eastleigh_gis_atlas.html`. Scripts: `06_gis_fetch.py`, `07_gis_maps.py`, `09_rainfall.py`.

## 10. Frame-by-frame record, audio and 3-D

- **Frame by frame:** `08_frame_by_frame.py` annotates all 298 frames with the tracked patches, measured and model values, phase and audio level. Outputs are `frames/`, `out/frame_by_frame.mp4`, `out/frame_metrics.csv` and `frame_viewer.html`. These contain footage of identifiable people, so keep them local.
- **Audio:** the clip's audio is dominated by voices, screams and the phone's automatic gain. Its level is high and broadband up to about 3.8 s, then drops about 10 dB. There is no distinct sound event before the visible motion, and phone microphones do not capture low-frequency structural rumble, so the audio is not diagnostic of initiation. Recordings from nearby seismometers or Raspberry Shake stations would be.
- **3-D scene:** the real OSM street block within 300 m is extruded (untagged buildings at an assumed 4 storeys) and the collapse is animated from the fitted physics, with exposure rings and dust biased along the street canyons. It is published as a private web page; a copy is at `out/eastleigh_3d.html`.

## 11. Further work

1. **Confirm the plot** (LR number or GPS), then re-run the GIS layers on the true footprint.
2. **Gather multi-view footage** (other videos, CCTV) and run structure-from-motion to get metric 4-D kinematics. This removes the storey-height assumption and shows the far elevation.
3. **Pull seismic records** from the Kenyan national network and any citizen seismometers in Nairobi. They give an independent timestamp and a measure of energy release.
4. **Build a construction timeline** from historical very-high-resolution imagery and shadow heights: storeys per month, concrete age when loaded, and built vs approved height.
5. **Run InSAR:** Sentinel-1 persistent-scatterer analysis for neighbourhood settlement, plus tasked high-resolution SAR on the adjoining buildings.
6. **Survey the debris by drone:** photogrammetry or LiDAR for volume and height (calibrates the compaction ratio λ), throw direction and the core remnant.
7. **Run high-fidelity collapse simulation:** Applied Element Method (e.g. ELS) or explicit FE (LS-DYNA) on the surveyed geometry and tested materials, compared against the measured kinematics.
8. **Rank the hypotheses with Bayesian updating:** a Bayesian network combining video, physics, GIS, materials and records, so the ranking comes with stated probabilities.
9. **Screen the city:** combine a building-height inventory, vertical-growth detection and permit data into a risk index, and inspect the highest-ranked buildings first.

## 12. Finite-element digital twin and advanced mechanics (`10_fe_twin.py`)

The twin is a parametric 3-D RC frame: 5 × 4 column grid on 5.6 × 6.7 m spans, 18 storeys, cracked stiffness. All sections, reinforcement and loads are assumed.

- **Modal:** fundamental periods 4.4/4.0/3.4 s. After corner loss, the local vertical mode is 0.75 s, so each cascade stage takes about 0.19 s. That is consistent with the measured ~0.6–0.8 s zipper.
- **3-D cascade (DAF 2):** design quality (sum P0/W ≈ 2.75) holds after losing one or two corner-end columns. As-built f′c 15 with two corner-end columns lost (S6) zips across all 20 L0 columns in 3 steps, outward from the corner. Losing one column holds unless the reserve is at its limit.
- **Failure envelope:** 12 of 48 combinations both stand intact and reproduce the full L0 zipper. All have reserve ≈ 1.6–2.3 plus initiating damage of one or two columns. **It needs both reduced capacity and a local trigger.**
- **Pushdown with energy-based dynamic capacity (Izzuddin):** the corner bay can just bridge with design reinforcement (λd ≈ 1.01) but not with poor reinforcement (λd ≈ 0.48 < 1). Poor bridging is consistent with the measured, continuing corner sag.
- **Column crushing (Mander concrete + bar buckling):** mean crushing force per storey ≈ 0.25 W (f′c 15) to 0.37 W (f′c 30), against 0.38–0.46 W measured from the video. Peak capacity is 1.9–3.2 W, so post-peak resistance is only about 13 % of peak: that is why the collapse could not arrest. *Correction:* an earlier version used a 30-column count and wrongly concluded that the measured resistance pointed to 15 MPa concrete. With assumed sections, this comparison cannot tell concrete strengths apart, and the evidence item was withdrawn from the Bayesian network.

## 13. Bayesian ranking of causes (`13_bayes_causes.py`)

Exact Bayesian network, conditioned on the collapse having happened, with 12 evidence items (video CV, physics, FE twin, GIS/remote sensing, soils, rain, press). The 90 % intervals come from Monte Carlo over all conditional probabilities.

| Cause | Prior | Posterior | 90 % CI |
|---|---|---|---|
| Loss or alteration of ground-storey load-bearing elements | 0.25 | **0.80** | 0.62–0.92 |
| Extra storeys / vertical extension | 0.30 | **0.71** | 0.50–0.86 |
| Deficient materials | 0.50 | **0.62** | 0.47–0.74 |
| Design / detailing deficiency | 0.30 | **0.37** | 0.25–0.51 |
| Foundation failure | 0.15 | **0.24** | 0.09–0.44 |
| Top-down construction failure | 0.08 | **0.00** | 0.00–0.00 |
| Explosion | 0.01 | **0.00** | 0.00–0.00 |
| Lateral event | 0.01 | **0.00** | 0.00–0.00 |

**Most probable combinations:** L0 load-path loss + Deficient materials + Extra storeys (0.18); L0 load-path loss + Extra storeys (0.13); L0 load-path loss + Deficient materials (0.09). Sensitivity (`fig_bayes_sensitivity.png`): the order among the top three depends on the priors, but all three stay well above the rest.

The strongest single evidence items are the rubble, ladders and missing blockwork at L0 and the corner-first motion (for L0 load-path loss), and the occupied ground-floor shops while construction continued above (for extra storeys / vertical extension). Foundation failure stays possible (≈ 0.24): high-clay soils raise it, the dry spell and terrain lower it. The probabilities are expert-elicited and should be challenged; the script makes them easy to change.

## 14. Remote sensing, seismic and soils

- **Archive imagery (`11_remote_sensing_timeline.py`):** 197 Esri Wayback releases give 12 distinct acquisitions over the site, 2011–2026 (WV02/03, Pléiades, GE01, Legion, 0.3–0.5 m). The latest is 16 Jan 2026. Low-rise blocks become a dense high-rise cluster.
- **Vertical-growth index (`16_vertical_growth.py`):** shadow area × tan(sun elevation) within 115 m rises about 4–5× from 2017 to 2025–26, with the steepest rise in 2021–2023. Hazy 2019–20 images are outliers.
- **Seismic (`12_seismic_soils.py`):** Raspberry Shake AM.R8A6C, 11.1 km away. The strongest 10-s energy in 13:20–14:00 EAT falls at 13:36:31 EAT, close to the reported ~13:40. It barely exceeds urban noise, so it is **inconclusive**. Check Kenyan national network stations.
- **Soils (SoilGrids 250 m):** the urban pixel is masked. The nearest valid pixel (~0.9 km E) has 43–48 % clay at 0–2 m; classification Cambisols / Ferralsols, with Vertisols at 9 % probability. Clay-rich ground keeps the foundation explanation open until a site investigation.

## 15. Animations

All of these are in `out/reconstructions/` and `out/story/`, each with sources and credit on every frame.

| File | Content |
|---|---|
| `R2_structural_twin.mp4/.gif` | FE twin cutaway: L0 DCR colours, failure order, deformation, crush-up, live plots |
| `eastleigh_story.mp4` / `_share.mp4` / `_social.gif` | Narrated (voice + subtitles): location, then the 3-D reconstruction of the failure, then a short verdict |
| `*_filmstrip.jpg` | Key frames of each reconstruction |

## 16. Was it a soft-storey failure?

**Partly. It was a ground-storey (weak-storey) failure, but under gravity, not the classic seismic soft-storey sway mechanism.**

- **What "soft storey" normally means.** A storey with much lower lateral stiffness or strength than the storeys above. The usual case is a tall, open ground floor of shops under infilled floors. Under earthquake or wind, the lateral drift concentrates there, and the columns fail in shear and P-Δ as the building sways and topples. Eurocode 8 and ASCE 7 treat it as a vertical irregularity.
- **What the evidence shows.** There was no lateral action: no earthquake, no wind event and no sway in the video. The descent was near-vertical, with crush-up kinematics. So the classic sway mechanism did not happen.
- **What did happen is a ground-storey axial (weak-storey) failure, and the soft-storey configuration helped cause it:**
  1. L0 was taller (≈4 m) and more open than the floors above: occupied shops, with blockwork partly missing at the time of the video.
  2. In buildings like this, the blockwork infill above carries part of the gravity load once the frame creeps, and braces the columns. At L0 parts of that blockwork were missing (rubble and broken blocks visible at t = 0); why is not known. Missing infill means the L0 columns had less support and bracing exactly where the axial demand is highest.
  3. If the L0 columns were also under-strength, the capacity reserve at L0 was small. The twin shows a reserve of ≈1.6–2.0 against ≈2.75 for design quality. A local loss at the corner then zipped across the storey (FE envelope, scenario S6).
- **In engineering terms:** a *gravity-load weak storey at L0 in an already low-reserve frame*. The "soft/open ground storey" geometry made it worse but was not the trigger. The trigger was loss of vertical capacity at the corner, not drift.
- **Practical implication for similar Nairobi buildings:** treat any removal or cutting of ground-floor walls or columns in a multi-storey frame as structural work that needs an engineer's assessment, and do not occupy buildings without an occupation certificate. Assess L0 axial reserve and robustness, not only seismic soft-storey checks.

## 17. Height cross-check (`17_height_sensitivity.py`) and energy budget (`18_physics_energy.py`)

| Storeys N | Resistance at onset (× weight) | Fit χ²/dof |
|---|---|---|
| 12 | 0.40 | 13–18 (rejected) |
| 14 | 0.40 | 3.6–6.6 |
| 16 | 0.42–0.44 | 0.7–1.6 |
| 18 | 0.46–0.49 | 0.2–0.4 |

The video kinematics favour **16–18 storeys** and reject 12. The resistance at onset stays at about 0.4–0.5 × weight for any N, so the conclusions do not depend on the disputed height.

**Energy:** 1.3 GJ (16 storeys) to 1.65 GJ (18 storeys) of potential energy is released. Kinetic energy peaks at about 0.3 GJ around t ≈ 3.3 s, and essentially all of it ends up dissipated in crushing.

## 18. Engineering explainers (`20_engineering_figures.py`, `21_mechanics_animations.py`)

- **Reserve map:** with 15 MPa concrete, interior L0 columns sit at P/(f′c Ag) ≈ 1.15 (reserve 1.1×) while the corner column sits at 0.42 (3.1×). A corner-first failure is therefore not explained by load alone.
- **Sudden loss:** the single-mass response peaks at 2.0× (undamped) or 1.85× (5 % damping), at T/2 ≈ 0.37 s for the damaged bay's local mode.
- **Resistance against weight:** peak storey capacity is 1.9–3.2 W, but the mean over the crush is 0.25–0.37 W, below the weight of the falling block throughout.
- **Height cross-check, failure-envelope map, Bayesian sensitivity and debris footprint** (pile 12–19 m, mostly in footprint).
- **Animations:** pushdown hinge sequence (poor reinforcement stays below the load), column crushing synced with force–shortening, timeline graphic, and the optical-flow measurement figure (local only).

## 19. Visualisation index

All outputs are in `analysis/visualisations/` (see its README). Every item carries its data sources and "Credit: Gava Charles".

*Credit: Gava Charles. Sources: [Kenyans.co.ke](https://www.kenyans.co.ke/news/128803-several-feared-trapped-building-collapses-nairobis-eastleigh-area), [Africanews / AP](https://www.africanews.com/2026/10/23/several-feared-trapped-as-multi-storey-building-collapses-in-nairobi/), [PBS NewsHour](https://www.pbs.org/newshour/amp/world/a-tall-building-under-construction-collapses-in-nairobi-with-4-believed-to-be-trapped).*
