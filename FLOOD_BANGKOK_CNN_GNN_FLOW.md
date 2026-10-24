# First Bangkok CNN–GNN flow-graph backtest

We tested whether explicit candidate water paths improve the 250 m citywide flood-complaint model. The target remains a **100 m buffer around a direct Traffy flood complaint touching a cell-day**. A cell with no complaint is zero for this proxy, not a verified dry cell. This is not a measured-inundation or flood-depth model.

## Graph and model

The [graph builder](flood_bangkok_flow_graph.py) creates one node per valid 250 m cell in each of Bangkok's 50 districts. It builds three message relations:

1. **Surface downstream:** one link to the steepest lower of eight adjacent cells if the 250 m terrain means differ by at least 0.05 m.
2. **Surface upstream:** the reverse of that link, so the model can learn an upstream/backwater effect separately.
3. **Canal:** bidirectional links between successive cells crossed by the same `waterway=canal` way in the 1 January 2025 OpenStreetMap Thailand snapshot. The open map does not establish canal flow direction.

The [graph audit](data/flood_bangkok_transfer/flow_graph_250m_v1/audit.json) records **28,309 nodes**, **23,137 directed downstream surface links**, and **17,832 directed canal links** across the district graphs. Cross-district links are absent. The 250 m surface links are terrain-based guesses, not observed water flow. Canal links follow mapped centreline geometry, not verified hydraulic connections to drains, pumps, or outfalls.

The [hybrid implementation](flood_bangkok_cnn_gnn_hybrid.py) loads the saved 17-channel CNN and **freezes it**. Each graph node receives that CNN's local hidden features plus the seven previously built canal, pump, pond, and underpass channels. Two relation-specific graph layers exchange messages along the surface and canal edges. A learned residual adjusts the saved CNN's output logit at each cell. The residual starts at exactly zero, so the initial prediction equals the saved CNN. This first pass does **not** jointly fine-tune the CNN backbone.

## Matched backtest

The graph head fits in the same 42 districts on dates through **31 July 2025**. It selects a checkpoint by Bang Kapi **August–September 2025** daily average precision, calibrates each district's intercept on its own August–September complaints, and tests **October–December 2025**. The same eight districts remain outside CNN and graph-head weight fitting. A matched **no-edge control** trains the same node features and residual architecture, with all graph connections removed.

Cell-rank Spearman is the mean across eight held-out districts between each cell's expected and observed test-period complaint-positive days. Daily AP is the mean district cell-day average precision. The last column sums expected complaint-positive cell-days across the held-out districts; **974** were observed.

| Model | Held-out rank Spearman | Held-out daily AP | Expected / observed cell-days |
|---|---:|---:|---:|
| Jan–Sep local complaint-history rank | **0.318** | — | — |
| Original 17-channel CNN | 0.309 | 0.01945 | 438 / 974 |
| Canal-feature CNN | 0.316 | **0.02298** | 365 / 974 |
| Frozen CNN + node head, **no edges** | 0.310 | 0.01893 | 379 / 974 |
| Frozen CNN + surface/canal graph | 0.306 | 0.01901 | 356 / 974 |

The graph has **no held-out ranking gain**: its mean Spearman is 0.003 below the no-edge control and 0.002 below the original CNN. Its daily AP is 0.00008 above the no-edge control but below both the original and canal-feature CNNs. It beats the original CNN's rank in three of eight held-out districts. The existing canal-feature CNN remains the strongest learned model in this comparison, while the simple historical rank is slightly higher on the eight held-out districts.

The poor graph result does not establish that drainage networks are unhelpful. The open graph lacks geocoded pipes, true cell-to-canal/outfall connections, cross-district links, water direction controlled by gates and tides, and time-matched pump state. Elevation means at 250 m can also miss kerbs, culverts, and small barriers. The October–December 2025 test window has been reused across experiments, so these comparisons are exploratory. The graph does not validate predictions for doubled or longer rain scenarios.

**Later data correction (23 October 2026):** a [public BMA ArcGIS drainage service](FLOOD_BANGKOK_DRAINAGE_GIS.md) has since been found with geocoded 2021 pipe polylines, canals and assets. The graph in this experiment did **not** use them. True inlet/outfall connectivity and historical gate/pump state remain unverified.

## Separate BMA road-flood check

The [BMA major-road flood archive](https://data.bangkok.go.th/dataset/frd_dds) is a second, independent **positive report source**. It has 1,435 rows across 2021–2025, including reported depth, but its event records give road and area text without coordinates. The citywide CNN and graph above were trained on **Traffy complaints only**, not on these BMA events.

For a conservative positive-site check, the [BMA check script](flood_bangkok_bma_positive_check.py) accepts only a unique exact normalized road-area name match to a point in BMA's **2026 flood-risk inventory**, inside the event's district. Across all 50 districts, this locates 325 of 1,397 BMA rows with depth from 2022–2025. For Q4 2025, 89 distinct positive site-days in 30 districts land on the model grids; only **five site-days in two districts** are among the eight geographic holdouts. The 2026 anchors make the locations retrospective and approximate.

The table gives the **mean percentile of each BMA positive site within its district on its event day** (1.0 means highest-ranked). It uses no BMA report for model fitting or model selection.

| Model | 89 Q4 site-days, all matched districts | Five Q4 site-days, geographic holdouts |
|---|---:|---:|
| Jan–Sep complaint-history rank | 0.384 | 0.371 |
| Original CNN | 0.704 | 0.871 |
| Canal-feature CNN | 0.685 | 0.819 |
| CNN–GNN | 0.710 | 0.875 |

The graph ranks these approximate BMA sites slightly higher than the original CNN, while its **Traffy complaint ranking is worse** in the eight-district test. Five held-out BMA positives are too few to claim a BMA advantage. BMA has no dry-cell observations, so positive-site percentiles cannot establish Spearman, AP, calibrated flood probability, flood extent, or depth accuracy. The [BMA check metrics](artifacts/flood_bangkapi_surface/surface_bma_positive_site_check_citywide_v1_metrics.json) and [matched test site-days](artifacts/flood_bangkapi_surface/surface_bma_positive_site_check_citywide_v1_site_days.csv) preserve the result.

## Reproduce and inspect

Run `.venv/bin/python flood_bangkok_flow_graph.py`, then `.venv/bin/python flood_bangkok_cnn_gnn_hybrid.py`, then `.venv/bin/python flood_bangkok_cnn_gnn_hybrid.py --no-edges`, then `.venv/bin/python flood_bangkok_bma_positive_check.py`. The [graph metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_cnn_gnn_flow_v1_metrics.json), [no-edge metrics](artifacts/flood_bangkapi_surface/surface_crossdistrict_cnn_gnn_flow_v1_no_edges_metrics.json), [district comparison](artifacts/flood_bangkapi_surface/surface_crossdistrict_cnn_gnn_flow_v1_district_comparison.csv), and [learning curves](artifacts/flood_bangkapi_surface/surface_crossdistrict_cnn_gnn_flow_v1_learning_curves.png) preserve the complaint backtest.
