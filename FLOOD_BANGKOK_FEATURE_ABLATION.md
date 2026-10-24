# Which inputs hurt cross-district flood-report transfer?

We audited the 17 feature channels in the 250 m CNN, then retrained the same ten-source-district, 2022–July 2025 model with one suspect group set to its **source-training mean** after normalization. Bang Khae remained outside source training. Checkpoint selection used Bang Kapi August–September 2025; Bang Khae August–September provided a pretest transfer check and an intercept adjustment. October–December 2025 is a **reused exploratory test**. The label is a 100 m Traffy flood-report buffer touching a cell-day, not verified water.

## Pretest feature audit

The [feature audit](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_feature_shift.json) compared source training with Bang Khae training and August–September calibration. These are marginal associations, so they cannot identify causal importance.

| Input | Source-to-Bang Khae observation | Possible issue |
|---|---|---|
| Absolute elevation | Positive report cells were lower than average in pooled source training, but higher than average in Bang Khae training and calibration. | An absolute height rule may transfer poorly. |
| Pump/gate locations | Bang Khae's mean nearest gate distance was **1.11 source standard deviations** higher; its mean nearest pump distance was **0.54** higher. | Location inventories may encode district identity and do not measure capacity or operation. |
| Prior report count | Positive/negative cell separation fell from **1.33 source standard deviations** in source training to **0.15** in Bang Khae August–September. | Report history may partly encode local reporting behavior. |
| Rain features | Rain amount, peaks, wet hours, and antecedent rain retained positive associations in Bang Khae. | No comparable directional reversal appeared in this audit. |

## Matched ablations

All rows below use the same architecture, chronological split, optimizer, and target-district intercept fitting. The source checkpoint is selected on Bang Kapi, without using Bang Khae test outcomes. Each ablation retrains from scratch. The test has **248** report-positive cell-days across **840** Bang Khae cells.

| Source CNN input | Bang Khae Aug–Sep AP | Oct–Dec cell Spearman | Oct–Dec daily AP | Brier | Predicted / observed cell-days |
|---|---:|---:|---:|---:|---:|
| All 17 features | 0.01087 | **0.324** | 0.02846 | 0.003186 | 58 / 248 |
| No absolute elevation | 0.01337 | 0.318 | 0.02913 | 0.003190 | 50 / 248 |
| No pump/gate inventory | 0.01072 | 0.316 | **0.03066** | **0.003184** | 61 / 248 |
| No prior report count | **0.01361** | **0.176** | 0.02159 | 0.003195 | 50 / 248 |
| No calendar sine/cosine | 0.01278 | 0.286 | 0.02294 | 0.003197 | 144 / 248 |
| No rain features | 0.00465 | 0.323 | 0.01301 | 0.003198 | 50 / 248 |

The prior-report feature looks vulnerable to reporting shift in the marginal audit, yet removing it almost halves geographic Spearman. It is currently useful for locating recurring report sites. Removing rain roughly halves daily AP, supporting its value for timing. Removing absolute elevation or pump/gate channels gives small daily AP gains but weaker geographic ranking; neither is a clear generalization improvement. Removing calendar features reduces the seasonal **count** underprediction, but loses geographic and daily ranking. No ablation fixes both distribution shift and calibration.

These comparisons are sensitive to the chosen source checkpoint: for example, the calendar-free model selected a later training epoch than the all-feature model. The October–December block has been inspected repeatedly, and only one held-out district was used here. Treat the results as candidates for a leave-one-district-out test, not as a feature-selection decision. A practical next model would keep rain and local report history, test relative terrain in place of absolute height, and represent pumps with capacity/operation data if available. It should be judged on several unseen districts and seasons before changing the map shown to users.

The [ablation metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_feature_ablation_metrics.json) and [840-cell comparison](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_feature_ablation_cell_comparison.csv) contain the full outputs. Rebuild with `.venv/bin/python flood_bangkok_feature_shift_audit.py` and `.venv/bin/python flood_bangkok_feature_ablation.py`.

## Subsequent open-data feature expansion

The [two-season feature-group backtest](FLOOD_BANGKOK_NEW_FEATURES.md) adds LiDAR low points, ESA/OSM surface cover, an OSM canal graph, and quality-screened HII gauge rain to a matched local/transfer tree comparison. **Surface cover** gives the most consistent Bang Kapi→Bang Khae gain across October–December 2024 and 2025. The combination of all four groups gives the best 2025 geographic transfer rank, but worsens daily AP and still underpredicts event counts. That result is for the controlled tree, not the 17-channel CNN in this report.
