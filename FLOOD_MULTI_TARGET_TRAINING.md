# Joint training with every available target type

Trained 22 October 2026. This run combines the available source types in **one shared CNN** with separate observation heads. It is a training artifact, not a validated citywide flood-probability map.

## Labels used

| Head | Training evidence | Supervision |
| --- | ---: | --- |
| UNOSAT map | 59 patch-dates from 2021 and one Chiang Mai 2024 scene | Mapped wet area; low-weight likely dry only inside analyzed extent. |
| EOS-RS map | One Hat Yai 2025 and two Bangkok 2026 patches | Mapped positive area only; transparent pixels unknown. |
| GISTDA map | 142 external-region 2026 patch-dates | Mapped positive area only; absent polygons unknown. |
| Traffy urban report | **29,535 positive 250 m cell-days** on **1,253 days** in Bang Kapi and Nong Chok patches, January 2022–September 2025 | Conditional location ranking of complaint-buffer cells inside Bangkok; no complaint is not dry. |
| BMA road incident | **51 matched depth cell-days** on **22 days** in the Bang Kapi patch, before October 2025 | Separate official-site ranking. |
| BMA reported depth | The same 51 matched cell-days | Low-weight conditional depth loss only at matched incidents; every other cell's depth unknown. |

The BMA archive has many more incidents, but most cannot be placed in a precise grid cell. These 51 cell-days came from **52 report rows** uniquely matched by place name to the 2026 BMA risk-point inventory, then clipped to the Bang Kapi patch and the training period. Their locations are historical proxies because the inventory postdates the events. No depth or dry target was fabricated for the unmatched rows. The [data audit](artifacts/flood_open_weak_train/multi_target_data_audit.json) lists source counts and matched records.

## Fitting

The model has one 22-channel spatial/rain encoder and six output heads: UNOSAT, EOS-RS, GISTDA, Traffy, BMA site, and BMA depth. The urban heads additionally see built cover, road density, and a rolling **prior 365-day report count**, computed without the current or future day. Source-specific heads let the shared representation learn from all evidence without calling an unreported site dry or forcing an urban complaint to look like satellite water.

Training ran **64 epochs and 3,072 updates**. The sampled task counts were 297 UNOSAT 2021, 78 UNOSAT 2024, 87 EOS-RS 2025, 147 EOS-RS Bangkok 2026, 765 GISTDA, 1,399 Traffy, and 299 BMA. The two single-scene external city events account for **165/3,072 updates (5.4%)**, down from 586/1,536 (38.2%) in the earlier satellite-only Bangkok-holdout CNN. Urban days and satellite patch-dates were rotated/flipped during fitting. Task weights and the 64-epoch endpoint were fixed before looking at any new held-out result.

The [checkpoint](artifacts/flood_open_weak_train/multi_target/multi_target_cnn_all_open_sources.pt) and [training record](artifacts/flood_open_weak_train/multi_target/training_run.json) are saved. All 205 satellite patch-date scenes entered this run, including the two 2026 Bangkok EOS-RS scenes. Therefore the earlier Bangkok EOS-RS scores are **not** holdout scores for this checkpoint. Urban report and BMA depth training stops on **30 September 2025**, preserving October–December report/depth rows outside fitting, but 2026 satellite labels entered the model; applying it backward to Q4 2025 would be a cross-time diagnostic, not a forward-time forecast.

## Checks and interpretation

The data builder confirmed finite rain and spatial inputs, report labels only inside the BMA candidate mask, and depth labels only at matched sites. The checkpoint reloads and gives finite 80 × 80 outputs for every head. Weather changes its outputs, and prior-report history changes the urban outputs. These are software checks, **not evidence of predictive improvement**. The source-ranking losses have a `log(number of cells)` offset and may be negative; comparing their raw values across tasks does not measure accuracy.

The satellite, Traffy, and BMA heads output **relative rankings of their respective observation processes**. The depth output is conditional on a matched BMA report and is not a citywide flood-depth estimate. Full-calendar-day rain can include hours after a satellite pass or complaint, and OSM/WorldCover snapshots can postdate older events. No probability calibration or rainfall-counterfactual validation was performed.

The subsequent [Q4 2025 urban diagnostic](FLOOD_MULTI_TARGET_Q4_DIAGNOSTIC.md) finds a small, uncertain report-capture gain over an urban-only CNN and no pooled gain over rolling report history. It is retrospective because this checkpoint was trained with 2026 satellite labels.

Reproduce from the cached public data:

```bash
.venv/bin/python flood_multi_target_data.py
.venv/bin/python flood_multi_target_train.py
```
