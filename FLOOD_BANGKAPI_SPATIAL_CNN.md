# Bang Kapi 2D CNN experiment

[Compare the cell rankings](artifacts/flood_bangkapi_surface/surface_connected_cnn_comparison_map.html) for historical report-footprint days, the earlier cell-feature tree, the 2D CNN and the October–December 2025 outcome. The [per-cell CSV](artifacts/flood_bangkapi_surface/surface_connected_cnn_comparison.csv), [GeoJSON](artifacts/flood_bangkapi_surface/surface_connected_cnn_comparison.geojson) and [metrics](artifacts/flood_bangkapi_surface/surface_connected_cnn_metrics.json) contain the values.

## Setup

Each training example is one day of the **37 × 35 grid** covering Bang Kapi. The 527 district cells have the same 23 inputs as the earlier model: 12 terrain, OpenStreetMap and infrastructure summaries; prior report-footprint days; prior complaint activity; two seasonal terms; and seven rain summaries. An extra channel marks which grid cells are inside the district. Rain is common to all cells on a day because the open historical series comes from one [Open-Meteo point](https://open-meteo.com/en/docs/historical-weather-api). The network has two 3 × 3 convolutions plus a local 1 × 1 path, so a cell's prediction can use a **5 × 5 cell neighborhood** (about 1.25 km across). It sees 250 m summaries, **not the raw 10 m LiDAR pixels or a drainage-flow graph**.

The target is the [connected-report footprint proxy](FLOOD_BANGKAPI_PROXY_PROBABILITY.md): a cell-day is positive if a terrain-connected report area or its 100 m fallback touches the cell. A day with no footprint is negative by this proxy rule, not verified dry ground. Train on January–July 2025, choose the epoch and correct the probability intercept using August–September, then hold out October–December. Report history stops at 30 September. All models and the historical baseline below use the same **241 positive held-out cell-days**. The CNN uses a fixed positive-loss weight of 30 because positives are rare.

## Held-out result

| Rank of October–December positive days per cell | Spearman | Comparable-cell pair agreement |
|---|---:|---:|
| Historical positive days, January–September | **0.490** | 77.7% |
| Cell-feature tree | 0.459 | **78.2%** |
| 2D CNN | 0.368 | 72.6% |

The 2D CNN **does not improve geographic ranking** on this held-out period. Its daily average precision is **0.0214**, versus **0.0198** for the tree; its Brier score is **0.004911**, versus **0.004922** for the tree. Those small daily-score differences do not offset the weaker cell ranking for the stated map-prioritization goal. The CNN predicts 208.4 positive cell-days versus 241 observed, after intercept correction. This single-district result does not establish that spatial features cannot help; it tests this small CNN with the available grid and labels.

The [training and validation loss curve](artifacts/flood_bangkapi_surface/surface_connected_cnn_loss_curve.png) shows positive-weighted binary cross-entropy per valid cell-day; the lower panel shows validation average precision. The [per-epoch values](artifacts/flood_bangkapi_surface/surface_connected_cnn_training_history.csv) mark the checkpoint selected by validation average precision, which is not necessarily the epoch with the lowest validation loss.

The [experimental CNN rain-scenario map](artifacts/flood_bangkapi_surface/surface_connected_cnn_probability_map.html) is included for inspection, not operational rain-to-flood simulation. Doubling duration alone lowers its predicted proxy probability in **81 of 527 cells**. The CNN has no monotonic-rain constraint, and neither model has local rainfall or verified water extent. `flood_probability` remains empty.

Rebuild with `.venv/bin/python flood_bangkapi_spatial_cnn.py`. If the project environment lacks PyTorch or Matplotlib, install them with `uv pip install --python .venv/bin/python torch matplotlib`. The script and map use the cached open-data inputs described in the [surface pilot](FLOOD_BANGKAPI_SURFACE.md).
