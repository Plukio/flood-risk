# Bangkok flood ranking: connections beyond nearby cells

Checked 22 October 2026. I built a citywide canal graph from the 1 January 2025 OpenStreetMap waterways and tested whether a cell's **previous reports elsewhere on its connected canal** add information beyond recent reports in nearby cells. We score report-positive 250 m cell-days in a daily top-10% queue. Unreported cells are unknown, not verified dry.

## What was connected

The preferred graph uses shared OSM node IDs. Two waterways crossing the same 250 m cell connect **only if the OSM ways share a node**. Long segments receive intermediate graph nodes, preserving the OSM junctions. The graph has **52,299 nodes, 52,142 edges, and 661 components**; its largest connected component spans about **65 km**. Of 25,812 Bangkok candidate cells, **24,000 are within 1 km** of a canal node. The [graph audit](artifacts/flood_open_weak_train/multi_target/citywide/osm_shared_node_canal_graph_250m_audit.json) and [builder](flood_citywide_true_canal_graph.py) preserve the source and attachment distances.

Each candidate cell projects its preceding 365-day report count to the nearest canal node within 1.25 km. A normalized walk passes that history along graph edges for 8, 16, 64, or 128 steps. The resulting connected-history feature is added to the [no-rain recent + nearby history score](FLOOD_CITYWIDE_HISTORY_RANKER.md). The weight and walk length are chosen on **January–September 2025**. A control adds report history smoothed by ordinary straight-line distance at 1, 2, or 4 km, with its weight chosen on the same period. Both use identical labels and daily queue budgets.

| Citywide daily top-10% capture | Jan–Sep 2025 selection | Q4 2025 | Q4 outside old patches + 750 m |
| --- | ---: | ---: | ---: |
| Recent + nearby history, no rain | **49.78%** | 48.17% | 46.08% |
| Best true-canal graph variant | 49.75% | **48.41%** | 46.11% |
| Best distance-only control | **50.61%** | 48.12% | **46.55%** |

The graph variant is **0.03 percentage points below** the no-rain score on selection data. It is **0.25 points above** it citywide in Q4, but the paired seven-day-block interval is **−0.14 to +0.66 points**. Outside the old training patches its Q4 gain is only **0.03 points** (interval **−0.32 to +0.24**). Across district-specific Q4 queues, the graph variant beats the no-rain score in 15 districts, loses in 20, and ties in 15. This does **not establish a predictive benefit from canal connectivity**.

The broader distance-only feature wins on selection data, but its Q4 citywide result also fails to improve the no-rain score. This control matters: an apparent gain from a graph feature could simply come from reaching farther than a 250 m neighborhood. A second, coarser [250 m raster graph](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_canal_connected_ranker.json) gave a different Q4 score, showing sensitivity to how crossing canals are connected. The shared-node graph is the more conservative topology; [full metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_canal_connected_ranker_osm_nodes.json) include both geography scopes and all 50 districts.

## Why this is still useful

The graph now gives us a way to send information **along a mapped canal corridor rather than to every cell inside a distance circle**. But OSM connectivity alone is not a hydraulic connection: the nearest canal may not drain a cell, and a gate, pump, culvert, tide level, or outfall can block or reverse the influence. Random-walk steps also do not equal a fixed travel time or flow distance. The present feature is a topology probe, not a flood-flow simulation.

For a meaningful connected-state model, each cell needs a defensible drainage attachment, with time-matched water-level differences across connected gates and operating pump/gate states. Those would let graph messages change by event. Until those data are available and tested, keep the **no-rain recent + nearby history score** as the ranking benchmark and the canal graph as an experimental input. Q4 labels had been inspected previously, so these intervals remain exploratory.

Reproduce with `.venv/bin/python flood_citywide_connected_ranker.py --graph osm_nodes`. The [evaluation code](flood_citywide_connected_ranker.py) also accepts `--graph raster` for the topology sensitivity check.
