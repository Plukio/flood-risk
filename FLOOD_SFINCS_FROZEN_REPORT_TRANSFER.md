# Frozen 10 m depth and direction features for masked Bangkok reports

**Exploratory result, 23–24 October 2026.** The proposed second CNN does use
nearby physics features and signed flow vectors. On the held-out Q4 2025
report queue it does **not** beat the same CNN trained on masked report history
alone. This is a ranking experiment for Traffy reports, not a validation of
flood extent, depth, or probability.

## Transfer path

1. Freeze the corrected [SFINCS depth + direction + mask checkpoint](artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/paired_extension/paired_direction_mask/checkpoint.pt). Run an initially dry, uniform 40 mm/h, 15-minute scenario on actual Bangkok LiDAR. Each trained-size 96 × 96 window produces simulated 10 m depth, signed east/north velocity, and 48 final hidden maps. We retain the central 48 × 48 pixels so overlapping windows cover the raster.
2. For every fully covered 250 m cell, pool **all 25 × 25 predicted pixels**. We use mean and 90th-percentile depth, mean signed east/north velocity, mean speed, and eight principal components of the 48 pooled hidden channels. PCA and channel scaling fit on source districts only. We also include a physics-coverage mask. Using all 625 pixels costs no extra SFINCS-CNN run compared with sampling five points.
3. Append those 14 layers to the existing masked-report image. A **second CNN over the 250 m grid** receives the target and its neighbors. Five depthwise 3 × 3 residual steps give it a seven-cell (1.75 km) radius, so velocity and other features in one cell can alter another cell's score. Its 250 m output is a report-site **ranking score**. The frozen 10 m checkpoint receives no report gradients.
4. Use the same 2023–2024 source-district group-ranking loss and exact target-centered 3 × 3 history mask as the existing masked CNN. The mask removes local raw report counts **and their contributions to all five Gaussian history layers** for positive and comparison patches. Select correction strength on January–September 2025 source districts; score October–December 2025 excluded districts at a daily 10% inspection budget with a 20% strict-isolated reserve.

The 10 m source is the three original study DTMs where available, then cached BMA district LiDAR only where its native pixel size is at most 15 m. Sources between 10 and 15 m are resampled to a 10 m grid; that interpolation creates no new detail. This covers **9,808 of 25,812** candidate report cells. The Q4 queue on this fixed subset contains **149** strict isolated report-positive cell-days in **74** touching groups. All rows below use identical covered cells, report examples, architecture width, random initialization, optimizer, masks, and queue.

## Held-out report ranking

| CNN input arm | Strict isolated cell-days found / 149 | Touching groups found / 74 | Interpretation |
| --- | ---: | ---: | --- |
| Fixed distant report history, no CNN correction | 23.34 (15.66%) | 19.76 (26.70%) | Queue reference on the covered subset. |
| Masked report-history CNN, physics layers zeroed | **37.58 (25.22%)** | **25.58 (34.57%)** | Best tested arm. |
| Depth + frozen hidden features; explicit vectors zeroed | 24.00 (16.11%) | 17.00 (22.97%) | Hidden features can still encode direction; this only removes explicit vectors. |
| Depth + signed direction + frozen hidden features | 32.76 (21.99%) | 21.76 (29.40%) | Above the no-vector arm, below history CNN. |

Fractional hits arise from tied-score allocation at the fixed inspection budget. Correction strength was chosen on source validation separately for each fold and arm, never on Q4. The full five-fold [result JSON](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_sfincs_frozen_spatial_report_mask_fivefold.json) includes source-validation selection, Q4 daily hits, and fold scores. In one small fold the direction-enabled model found 7 sites versus history CNN's 5; a larger fold reversed that gain. The pooled result is the relevant comparison.

## Does the second CNN use neighboring cells?

Yes in the literal model-sensitivity sense. On 128 sampled held-out cells per fold on the first Q4 day, removing **neighboring** physics layers while preserving the target cell's physics changed every tested score; mean absolute score change was **0.091**. Removing the explicit vector and speed layers changed every tested score; mean absolute change was **0.134**. See the [neighbor sensitivity audit](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_sfincs_frozen_spatial_neighbor_audit.json). This shows the CNN has learned to use the inputs. It does **not** show that its learned directional relation is physically correct or helps report ranking: the held-out queue says it does not win here.

We also checked the direction of the learned response. For an eastward vector,
positive flow at the **west** neighbor should matter more as incoming flow than
the same vector at the east neighbor; the northward analogue compares the
south and north neighbors. The incoming-minus-outgoing gradient was positive
in **63.1%** of sampled eastward comparisons but only **39.1%** of northward
ones, with large disagreement across folds. Thus the second CNN responds to
nearby vectors, but it has **not learned a stable physical arrow rule** from
the report target alone.

## Explicit directed-neighbor follow-up

After seeing that sign audit and the first Q4 result, we added a simple
**one-cell incoming-flow proxy**. Mean simulated depth times signed east/north
velocity is shifted toward the indicated neighbor; the CNN also receives
incoming minus outgoing flow. This makes the *input's* neighbor relation
directional, while the report CNN still learns how much it matters. It is a
surface-flow proxy, not a measured flux across a 250 m boundary or a known
drain connection. A matched control has the same architecture and two zeroed
route channels.

| Report-head seed | No-route CNN: cells / groups | Directed-neighbor CNN: cells / groups |
| --- | ---: | ---: |
| 0 | 30 / 19 | **39 / 26** |
| 10,000 | **34.76 / 21** | 32 / **23** |

Both use the same 149 isolated cell-days and 74 groups, with report-head
strength selected on source validation. [Seed 0](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_sfincs_directed_neighbor_mask_fivefold.json)
and [seed 10,000](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_sfincs_directed_neighbor_mask_fivefold_seed_10000.json)
contain daily scores and fold selections. The directed cue wins the matched
cell comparison in one seed and loses in the other. It does **not** establish a
stable improvement. Because we designed this cue after inspecting Q4, even
the seed-0 score of 39 versus the history CNN's 37.58 is an exploratory
diagnostic, not a new independent holdout result.

## Unfreezing the 10 m checkpoint on masked reports

We then took the seed-0 directed-neighbor report CNN and kept **that report
head fixed**. The original 10 m model supplied the first two five-minute
states. At the third step, report ranking gradients flowed through the 250 m
CNN and differentiable pooling of the target's **3 × 3 nearby 250 m cells**
into the last two 10 m convolutions and the depth/velocity heads. All earlier
10 m layers remained frozen. Each geographic fold used 96 sampled source-
district training days from 2023–2024, up to three masked positive groups per
day, and a small penalty for drifting from the original simulator outputs.
Afterward we rebuilt every covered 250 m feature cell, selected report-head
strength on January–September 2025 source districts, and evaluated the same
Q4 held-out districts. A no-change 10 m model reproduced the original tiled
Bang Kapi features exactly; the source-only feature transform also reproduced
the saved features within 0.00001.

| Q4 isolated report capture | Cell-days / 149 | Groups / 74 |
| --- | ---: | ---: |
| Frozen 10 m model + directed-neighbor report CNN | **39.00** | **26.00** |
| Fine-tuned 10 m model, original report-head strength | 34.12 | 22.16 |
| Fine-tuned 10 m model, source-validation-selected strength | 29.72 | 22.00 |

The [five-fold fine-tuning result](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_sfincs_report_unfreeze_fivefold.json)
records the fold-by-fold comparisons and [adapted 10 m checkpoints](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_sfincs_report_unfreeze_fivefold.pt).
At source-selected strength, fold cell hits changed from 4 to 4.76, 6 to
5.96, 10 to 4, 10 to 8, and 9 to 7. Fine-tuning **hurt this held-out report
task**; even keeping the original correction strengths loses 4.88 cell-days.
The report gradients do reach the 10 m layers, but the weak report target did
not preserve a useful transferable physical representation in this run.
This is one adaptation schedule on a Q4 set already inspected; it does not
rule out every partial-unfreezing method.

## Reading and limits

The transfer is structurally valid: a 10 m model produces subcell maps, pooling gives one vector per 250 m cell, and a spatial 250 m CNN can combine neighboring vectors. The unconstrained frozen physical representation makes held-out report ranking worse than the matched masked-history CNN. The oriented proxy has an unstable result across two report-head seeds; unfreezing the 10 m model on weak reports reduces held-out capture further. Keep report history as the current ranking model; the simulator representation remains useful as an exploratory scenario layer, not an operational report predictor.

The 40 mm/h rain scenario is **assumed** and has no observed 10 m starting water state. The simulator uses closed tile edges and uniform assumed roughness in this transfer; it omits drains, pumps, gates, infiltration, and buildings. Its hidden maps were learned to imitate SFINCS, not reports. Unreported comparison cells are unknown rather than verified dry. The SFINCS checkpoint saw synthetic Bang Kapi and Bang Khae terrain during pretraining, although Q4 report labels and the report-head fit are held out in the stated splits. Q4 2025 has been inspected repeatedly in this project, so the comparison is exploratory.

## Reproduce

```sh
.venv/bin/python flood_sfincs_report_features.py
.venv/bin/python flood_sfincs_report_mask_transfer.py
.venv/bin/python flood_sfincs_report_neighbor_audit.py
.venv/bin/python flood_sfincs_report_direction_route.py
.venv/bin/python flood_sfincs_report_direction_route.py --seed-offset 10000
.venv/bin/python flood_sfincs_report_unfreeze.py
```

The [feature audit](artifacts/flood_open_weak_train/multi_target/citywide/sfincs_10m_direction_250m_frozen_features.json) records district coverage and source resolution. Model weights are in the matching `.pt` file beside the result JSON.
