# Photo-based preliminary collapse assessment

A reproducible pipeline for a structural engineer to produce a **preliminary screening assessment** of a building collapse from photographs alone: no video, drawings or test results are required. It produces a reasoned cause ranking with stated uncertainty and an explicit list of what the photographs cannot establish.

It is a screening tool. It does not determine cause. Material testing, drawings and site investigation are still needed.

Credit: Gava Charles

## What it can and cannot do

| Usually established from photos | Usually not established from photos |
|---|---|
| Collapse mode (pancake, tilt/overturn, partial, blast-like) | Where failure started (ground storey vs upper floors) |
| Whether foundation or lateral action is plausible | Concrete strength, cover, as-built bar diameters (without scale or testing) |
| Construction stage, height, neighbour exposure | Tie spacing and laps (unless close-ups with a tape exist) |
| Visible defects (cut columns, sparse ties, smooth or corroded bars, honeycombing) | Timing and dynamics (needs video or seismic records) |

In the Eastleigh case, photos alone established a vertical pancake collapse and ruled out explosion, earthquake and wind, and overturning. They left ground-storey load-path loss at about 0.5. The video raised it to 0.83.

## Folder layout

```
photo_pipeline/
  run.py                    stages: intake -> context -> earthengine -> evidence -> measure -> bayes -> fieldplan -> report
  context.py                open-data context stage;  earthengine.py: building-height history and Sentinel before/after (Google Earth Engine);  field.py: what-to-check-next (value of information)
  new_case.py               scaffold a new case from a folder of photos
  core.py, bn.py, figures.py
  config/network.json       Bayesian network: causes, priors, conditional probabilities (shared by all cases)
  config/evidence_library.json   what can be observed, what to look for, how it bears on each cause
  config/field_tests.json   follow-up checks (desk or field), each observing one library item
  validation/               benchmark of past collapses with published findings (see validation/README.md)
  models/face_yunet.onnx    face detector for privacy blurring (OpenCV Zoo)
  cases/<case>/case.json    case metadata, scenarios, optional non-photo evidence, measurement settings
  cases/<case>/tags.json    engineer-reviewed observations per photo
  cases/<case>/out/         all outputs, plus run_manifest.json (hashes of inputs, config, code; package versions)
```

## Procedure (SOP): desk study, no field work

1. **Set up once.** Install with `python -m venv venv && venv/bin/pip install -r requirements.txt` (Python 3.9+, `curl` on the path).
2. **Create the case.** Run `python new_case.py cases/<name> /path/to/photos --name "..." --site "..." --date YYYY-MM-DD --lat <lat> --lon <lon>`.
   - Photos can come from press, social media or rescue teams. Record each source in `photo_source`.
3. **Intake and context.** Run `python run.py cases/<name> --stages intake context`.
   - **Intake** inventories and hashes every photo, flags duplicates, reads EXIF, and blurs faces. **Check every image in `out/redacted/`**, add boxes in `tags.json → redact_manual` for anything missed, and re-run. Do not publish anything until this is done.
   - **Context** pulls open data for the location and date (cached, re-runnable): USGS earthquakes, Open-Meteo rainfall against the previous 10 years, OpenStreetMap waterways, neighbours and construction sites, SoilGrids clay and soil class, and Esri Wayback satellite chips (`out/context/sat/`). It writes `fig_context.png` and **proposes** evidence items in `out/context/context.json`, each with its reason.
   - **Accept a proposal only after checking it.** For example, a "construction site nearby" proposal needs an excavation visible on the chips, and a high-clay soil is only expansive if it is a swelling clay. List accepted items in `case.json → desk_evidence`.
   - If a service is down, the stage says so and carries on. Re-run it later.
   - **Earth Engine (optional):** set `ee_project` in `case.json` (or `--ee-project` in `new_case.py`) to a Google Cloud project registered for Earth Engine, and run `earthengine authenticate` once. Then `python run.py cases/<name> --stages earthengine` writes `fig_earthengine.png` with three things:
     - the building height at the point each year, 2016–2023 (Google Open Buildings 2.5D Temporal; Africa and South/Southeast Asia);
     - Sentinel-2 before/after imagery (cloud-covered scenes are flagged);
     - Sentinel-1 radar change, with before and after matched to the same orbit.
     
     It proposes `DESK_RAPID_VERTICAL_GROWTH` when the plot rose by two or more storeys within three years. The height is the maximum within 20 m of the point, so **with an unconfirmed plot location it may belong to a neighbour.**
4. **Tag.** Edit `tags.json` for each photo:
   - `view`: wide, element or aerial (satellite chips can be tagged as aerial).
   - `observations`: each with `label`, `category` (a theme in the library), `confidence`, `box` (normalised x0, y0, x1, y1) and `evidence` (a library item ID).
   - **The engineer reviews and owns every tag.** Use `low` confidence for anything unverified: low-confidence items appear in the figures but are excluded from the ranking. Tag only what is visible.
5. **Public record.** Add record items (permits vs storeys built, occupancy, prior distress, works under way) to `record_evidence`, keeping the source URLs in your case notes. Put case-specific findings, such as video analysis, in `external_evidence`. Each scenario in `scenarios` combines `photos`, `record`, `desk` and `external`.
6. **Run everything.** Run `python run.py cases/<name>` (all stages).
7. **Review `out/report.md`.**
   - Check the influence chart: no single observation should drive the result unless it is genuinely decisive.
   - **Section 7, "What to check next",** ranks follow-up checks by how much each would change the ranking. Each is marked desk (records, drawings, interviews, archive imagery: start now) or field (needs site access: hand to whoever can get there). The checks are defined in `config/field_tests.json`.
8. **Archive** the case folder together with `run_manifest.json`.

## Using it in another country (e.g. Uganda)

Nothing in the code is tied to Kenya: the open-data sources are global and Open Buildings covers East Africa. When starting a case elsewhere:
- **Check the priors** in `config/network.json` against local collapse records. The East African stock is similar, but document any change.
- **Field-test sources** in `config/field_tests.json` name NCA and NBRB/KCCA. Add the local regulator if needed. EAS 412 rebar is harmonised across the East African Community.
- **No Ugandan case is in the validation benchmark yet.** Adding Ugandan collapses with published findings (for example from NBRB, KCCA or court records) to `validation/benchmark.json` would test the network on local stock.

## Outputs

- `fig_context.png` and `context/`: open-data context (rain, earthquakes, surroundings, soil, satellite chips) and the proposed desk evidence.
- `fig_field_priority.png` and `field_priority.json`: follow-up checks ranked by expected information gain, marked desk or field.
- `fig_summary_card.jpg`: collapse mode, photos-only ranking, data gaps (one slide).
- `fig_evidence_NN_<theme>.jpg`: one figure per theme, with zoomed crops of each observation, what it tells us, and what is not established.
- `fig_bayes_scenarios.png`: the ranking under each evidence scenario, with 90 % intervals.
- `fig_evidence_influence.png`: how much each observation moves each cause.
- `plates/`: every photo annotated, with numbered observations.
- `evidence_log.csv`: every observation, with its photo, box, confidence and evidence item.
- `report.md`: the preliminary assessment.

## Extending and calibrating

- **New observation types:** add an item to `evidence_library.json` with its theme, description, `look_for` text, likelihood `terms` and `contradicted_by`. It is then available to every case.
- **Priors:** `network.json` holds base rates for collapses in the target building stock. Revise them per region, and document why.
- **Validation:** `validation/` scores the ranking against 12 past collapses with published findings (`python validation/validate.py`). In v2, a primary cause ranks in the top three in 92 % of cases from photos alone, and in 100 % with the early public record. The set is small and unblinded; read `validation/README.md` before quoting these figures. Add new cases to `validation/benchmark.json` and score them before changing the library or network.
- **3-D (future stage):** with 20 or more overlapping photos or drone imagery, add structure-from-motion (e.g. COLMAP) to measure pile height and volume.

## Known limits

- Probabilities are expert-elicited. Three network weights were recalibrated by leave-one-out on 12 benchmark cases (`validation/`). That is a small, unblinded set, so the rankings are still screening-grade.
- Photos of a pancake collapse cannot separate deficient materials, overload and design deficiency. Those three typically sit within about 0.15 of each other; use the public record and testing to separate them.
- Face detection can miss small or turned faces, and can occasionally blur a non-face. Manual review is mandatory.
- The Bayesian network treats evidence items as conditionally independent given the causes and modes. Avoid tagging the same fact twice under different items.
