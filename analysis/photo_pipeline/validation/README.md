# Validation against past collapses

Does the photo pipeline's cause ranking agree with what inquiries found? This folder tests that on 12 past collapses with published findings, records what failed, and documents the changes made in response.

Credit: Gava Charles

## Run

```
python validation/validate.py      # v1 (frozen snapshot) vs current config -> out/
python validation/calibrate.py     # leave-one-out calibration report (--apply writes config/network.json)
```

| File | Content |
|---|---|
| `benchmark.json` | 12 cases: photo observations (scenario A), early public record (scenario B), graded inquiry findings, sources |
| `evidence_library_v1.json`, `network_v1.json` | frozen pre-validation configs |
| `out/fig_validation_matrix_v1.png`, `_v2.png` | posterior per case and cause, with the inquiry findings marked |
| `out/fig_validation_metrics.png` | top-1, top-3, AUC, Brier skill: v1 vs v2, A vs B |
| `out/validation_results.json`, `out/validation_log.txt`, `out/calibration.json` | full numbers |

## Benchmark

Rana Plaza 2013, Sampoong 1995, Harbour Cay 1981, Lotus Riverside 2009, Ronan Point 1968, Ikoyi Gerrard Road 2021, SCOAN guesthouse 2014, Ita-Faji 2019, Rönesans Rezidans 2023, Hotel New World 1986, Melcom Achimota 2012, Champlain Towers South 2021.

Findings are graded per cause as primary, contributing or ruled out; anything else is "not established" and not scored. Tiers: official inquiry/coroner (7), professional body or federal agency (3), interim (Surfside, NIST final pending), expert opinion only (Rönesans). Kenyan cases (Huruma 2016, Mirema 2023, others) were excluded because no published cause finding could be found.

## Results

Prior alone (no photos): top-1 0.33, top-3 0.75.

| Version | Scenario | Top-1 | Top-3 | AUC | Brier skill |
|---|---|---|---|---|---|
| v1 (elicited) | A photos only | 0.50 | 0.58 | 0.90 | +0.26 |
| v1 | B photos + record | 0.50 | 0.58 | 0.90 | +0.26 |
| v2 | A photos only | 0.58 | 0.92 | 0.92 | +0.36 |
| v2 | B photos + record | 0.67 | 1.00 | 0.94 | +0.47 |
| v2, leave-one-out held-out (A and B averaged) | | 0.62 | 0.96 | | |

### What v1 got wrong

1. **Every pancake collapse got the same ranking** (materials > L0 load-path loss > extra storeys). Photos of a pancake debris pile do not distinguish materials, overload and design. That is expected and is now stated in every report.
2. **Design deficiency was never in the top three**, although inquiries found it primary in 5 of 12 cases. v1's top-3 rate (0.58) was *worse than the priors alone* (0.75). Cause: in the network, design deficiency reached collapse only weakly (through low capacity → initiation).
3. **No discriminating evidence for explosion, earthquake or construction-stage failure.** Ronan Point (gas explosion) ranked 7th and Rönesans (earthquake) 7th; Harbour Cay (failure during a pour) had top-down construction at 0.07.

### Changes (v2)

- **Library:** added `WIDESPREAD_DAMAGE_NEARBY`, `PILES_OR_FOOTINGS_EXPOSED_FAILED`, `WALL_PANELS_BLOWN_OUT` (photo items) and `RECORD_FLOORS_EXCEED_PERMIT`, `RECORD_USE_CHANGE_OVERLOAD`, `RECORD_POUR_IN_PROGRESS`, `RECORD_EXPLOSION_REPORTED`, `RECORD_EARTHQUAKE` (record items). Their likelihoods were set by judgement before the v2 run and were not tuned afterwards.
- **Cause X** relabelled "Extra storeys or loading beyond design" (vertical extension, change of use, heavy plant). Three inquiries (Rana Plaza, Sampoong, Ita-Faji) blamed overload from use, not only extra floors.
- **Network calibration** (`calibrate.py`): leave-one-out over `cap_D`, `cap_M`, `init_CAP` selected `cap_D` 0.5 → 0.85, `cap_M` 0.6 → 0.45, `init_CAP` unchanged. The same values were chosen in all 12 folds. The held-out top-3 rate rose from 0.71 to 0.96, and the held-out log score from −0.432 to −0.419. `init_A` (L0 load-path loss) was deliberately **not** fitted: only one inquiry graded it, so a fit lowers it only to make room for the graded causes.

### Effect on Eastleigh

Photos only: deficient materials 0.60 → 0.55, design deficiency 0.38 → 0.45, L0 load-path loss 0.49 (unchanged). With the video: L0 load-path loss 0.83 (unchanged). The conclusions do not change. v1 numbers are kept in `out/eastleigh_bayes_v1.json`.

## Limits: read before quoting these numbers

- **Small and unblinded.** 12 cases, and the photo observations were encoded by an assessor who knew the findings. The v2 library items were added in response to these cases, so the v2 figures for Ronan Point, Rönesans and Harbour Cay are in-sample by construction. Only the network calibration has a held-out estimate.
- **Observations come from published descriptions of photographs**, not a re-tagging of the images in this pipeline.
- **Selection bias.** Famous inquiries over-represent exotic causes (explosion, earthquake, design errors) and under-grade L0 alterations. For that reason the priors were not fitted to the benchmark.
- **Calibrated values sit at the edge of the grid** (`cap_D` at its maximum, `cap_M` at its minimum). The data would push further. They were not extended past engineering plausibility.
- **Top-down construction stays weak from photos alone.** Props are present at most construction-stage collapses, whatever the cause. It needs the record ("pour in progress") to rise (Harbour Cay: 0.07 → 0.35).
- **Explosion with a robustness-deficient design** (Ronan Point): the network credits the explosion but has no path for "lack of robustness amplified the trigger". Design deficiency is under-scored in that case.

## Next steps

1. **Blind re-tagging:** an independent engineer tags actual photographs of these 12 cases, without the findings, using the pipeline's `tags.json`. Compare the results with this run.
2. **Held-out set:** add 10+ further cases, ideally East African with NCA, NBRRI or court findings, and score them **without** further changes to library or network.
3. **Base rates:** set priors from a regional collapse register (e.g. the NCA audit of 87 collapses in Kenya), not from inquiries.
