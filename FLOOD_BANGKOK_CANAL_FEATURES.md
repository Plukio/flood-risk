# Canal geometry in the Bangkok flood-report CNN

We added two static canal channels to the 250 m cell CNN: **mapped canal centreline length inside each cell** and **distance from the cell centre to the nearest mapped canal**. Both come from the local [OpenStreetMap Thailand snapshot dated 1 January 2025](https://download.geofabrik.de/asia/thailand.html), using `waterway=canal`. The [extraction audit](data/flood_bangkok_transfer/canal_2025_audit.json) found **2,883 canal ways** inside the eleven-district study extent plus a 5 km buffer. In Bang Khae, 444 of 840 model cells contain a mapped canal line; the cell-centre median distance is 139 m. The [cell comparison table](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_canal_cnn_cell_comparison.csv) includes both canal inputs for inspection.

The features are geometry only. They do not specify canal width, depth, water level, flow direction, blockage, tide, gate position, or pump operation. The 2025 map is applied retrospectively to 2022–2024 training examples. The [BMA canal catalog](https://data.bangkok.go.th/dataset/canal) lists older water-quality readings without usable coordinates. Another [BMA water-source table](https://data.bangkok.go.th/dataset/data_canal) has dimension fields, but the accessible CSV has only 63 rows for district 1011 (Lat Krabang) and no coordinates, so neither table can supply cell-level geometry for this eleven-district comparison. The separate [historical water-level audit and backtest](FLOOD_BANGKOK_WATER_LEVEL_BIAS.md) checks BMA's 2022–2025 river gauge and the 2026-only canal-level files.

## Matched Bang Khae comparison

The transferred CNN trains on ten districts through July 2025, excluding Bang Khae. Bang Kapi August–September selects the checkpoint; Bang Khae August–September fits the local intercept. October–December 2025 is a reused exploratory test with **248** report-positive cell-days across **840** Bang Khae cells. Both CNNs use the same chronological split, optimizer, and 100 m report-buffer labels.

| Model | Cell Spearman | Daily AP | Brier | Predicted / observed cell-days |
|---|---:|---:|---:|---:|
| Transferred CNN, no canal channels | 0.324 | 0.0285 | 0.003186 | 58 / 248 |
| Transferred CNN, **with canal channels** | **0.336** | **0.0331** | 0.003188 | 49 / 248 |
| No canals + local rain-count correction | 0.323 | **0.0401** | **0.003141** | 187 / 248 |
| Canals + local rain-count correction | **0.333** | 0.0390 | 0.003142 | 187 / 248 |

The canal channels improve the transferred CNN's cell rank and daily AP, but its uncorrected count remains far too low. With the same local rain-count correction, they retain a **0.010 Spearman** advantage while daily AP is slightly lower. This is a ranking gain in one district, not evidence that the model understands canal hydraulics.

For a Bang Khae-only CNN trained on local 2022–July 2025 data, canal channels increased daily AP from **0.0274 to 0.0393** and improved Brier from **0.003169 to 0.003145**, but reduced cell Spearman from **0.360 to 0.335** and reduced the predicted count from **135 to 113** versus 248 observed. The mixed result argues against treating canal proximity as a universal fix. The October–December block has already been inspected in earlier work, so these comparisons are exploratory.

The [canal CNN metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_canal_cnn_metrics.json), [rain-count hybrid metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_canal_rate_hybrid_metrics.json), and [hybrid 840-cell table](artifacts/flood_bangkapi_surface/surface_crossdistrict_bangkhae_canal_rate_hybrid_cell_comparison.csv) contain the full results. Rebuild the features with `.venv/bin/python flood_bangkok_canal_features.py`, train with `.venv/bin/python flood_bangkok_spatial_cnn_canals.py`, then run `.venv/bin/python flood_bangkok_canal_rate_hybrid.py`. Before adopting the canal channels, evaluate them on other held-out districts and seasons; adding timestamped canal levels or flow-control state would allow a stronger test of drainage effects.
