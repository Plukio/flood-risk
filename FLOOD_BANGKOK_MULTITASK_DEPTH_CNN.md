# Joint CNN for flood complaints and recorded road depth

I trained an exploratory **two-output spatial CNN** over the same eleven 250 m district grids. A shared two-layer 3×3 CNN feeds (1) a **Traffy flood-complaint occurrence** head and (2) a **BMA road-flood depth** head. The classification loss uses the complaint-buffer cell-day labels. The depth loss is applied **only at BMA road-report cell-days matched to a point**; all other depths are unknown, not zero.

## Depth-label coverage

The [BMA road-flood archive](https://data.bangkok.go.th/dataset/frd_dds) has depth and road/area text but no event coordinates. I accepted only a **unique exact normalized name match** to a point in the [2026 BMA flood-risk inventory](https://data.bangkok.go.th/dataset/flood_risk26), within the same district polygon. This is deliberately conservative. Across 2022–2025 in the eleven districts, **502 reports have depth**, but only **137 rows** match under that rule; two share a cell-day, leaving **135 labeled cell-days**:

| Split | Matched BMA depth cell-days |
|---|---:|
| Train: 2022–31 July 2025 | 69 |
| Select/check: August–September 2025 | 37 |
| Test: October–December 2025 | 29 |

The test has 4 observations at 10 cm, 10 at 15 cm, 14 at 20 cm, and one at 25 cm. **The 50 cm Chatu Chak report from the broader depth audit did not match a unique inventory point and is not in this test.** The 2026 point inventory was published after the events, and even an exact place-name match does not verify the event's 250 m cell. This is a retrospective location proxy, not a clean spatial ground truth.

## Held-out result

Both networks use the same sampled training scenes and architecture; only the joint network receives the additional masked depth loss. Epoch selection uses August–September **classification average precision**. October–December 2025 supplies no labels for fitting or selection. Rain inputs include the full historical day, so this is a hindcast rather than a pre-storm depth forecast.

| Model | Q4 complaint classification AP | Q4 depth MAE on 29 matched cells |
|---|---:|---:|
| Classification-only CNN | **0.03716** | — |
| Joint classification + depth CNN | 0.03651 | 4.54 cm |
| Constant 15 cm depth, from training median | — | **3.45 cm** |

The classification target's Q4 prevalence is **0.00565** across 580,704 mapped cell-days, so both CNNs rank that proxy better than random. Adding the depth task **slightly lowers classification AP** and produces **worse depth estimates than the constant baseline**. On the validation period, the best depth MAE reached by the joint model in any epoch is **3.115 cm**, also above the constant baseline's **3.035 cm**. This run gives no evidence that the sparse depth labels improve the shared model.

The depth head is **conditional on a matched BMA road report**. It cannot make a trustworthy citywide water-depth map: there are no measured zero-depth labels for other cells, BMA monitors selected major roads, and most report locations remain unplaced. The existing pre-October [six-site depth-capture test](FLOOD_BANGKOK_CNN_DEPTH_CAPTURE.md) is a separate ranking question; it does not turn this depth head into a validated severity model.

## Reproduce

Run `.venv/bin/python flood_bangkok_multitask_depth_cnn.py`. Inspect [metrics and training histories](artifacts/flood_bangkapi_surface/surface_multitask_bma_depth_cnn_metrics.json), [conservatively matched BMA rows](artifacts/flood_bangkapi_surface/surface_multitask_bma_depth_cnn_matched_records.csv), and [29 held-out depth predictions](artifacts/flood_bangkapi_surface/surface_multitask_bma_depth_cnn_Q4_depth_predictions.csv). The script reads cached public BMA, BMA inventory, Traffy, LiDAR, and rain files. It does not infer zero depths from missing BMA records.
