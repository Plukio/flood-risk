# Q4 2025 Bangkok report-queue map

Open the [interactive map](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_history_interactive_map.html). It is a self-contained HTML file. The [Bangkok preview](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_history_map_preview.png) shows the default view; the [Bang Kapi preview](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_history_map_detail_preview.png) shows one date with different picks highlighted.

The two maps compare the **nearby-report-history baseline** and the **synthetic-center-plus-block masked CNN** on the same five Q4 2025 held-out district folds. The right map can also show **CNN + terrain neighbors**, an experimental one-hop score average whose blend weight was chosen on source districts. Orange marks 250 m cells with Traffy flood reports. Blue or purple marks cells selected for inspection. “All Q4” shows how often each cell would have been selected across 92 daily queues; choose a date to see one queue. Select a district to zoom and see statistics for that district, while the actual competition for slots remains fold-wide. “Only different picks” removes cells both methods would select.

The top-area slider sets the percentage of candidate cells inspected **per day within each held-out fold**, from 1% to 30%. At every value, 20% of inspection slots are reserved for cells with no own-cell report in the preceding 365 days. This keeps the baseline and CNN at the same resource budget. The card counts are weighted where scores tie at the cutoff; a partial selection appears with partial fill. At the default **10%** inspection area, the exported map exactly reproduces the saved Q4 backtest:

| At 10% area | Nearby history | Masked CNN |
| --- | ---: | ---: |
| All 7,209 report-positive cell-days captured | 44.114% | 44.479% |
| 1,664 first-in-a-year positive cell-days captured | 11.946% | 13.529% |

Switching to **CNN + terrain neighbors** at the same 10% area gives **44.607%** overall capture and **14.083%** first-in-a-year capture. The [continuity test](FLOOD_CITYWIDE_TERRAIN_CONTINUITY.md) explains why that small exploratory gain does not establish physical flow learning; ordinary neighbor smoothing performs equally well.

**Show possible terrain-linked cells** adds cyan outlines around unreported cells next to a same-day reported cell when their 250 m mean DTM heights differ by at most 0.25 m. This is an outcome-derived *hypothesis overlay*, not observed flood extent, not a dry/wet label, and not a score input. The [continuity citywide preview](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_continuity_map_preview.png) and [Bang Kapi detail preview](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_mask_cnn_continuity_detail_preview.png) show the added controls.

The map shows *reported cell-days*, not verified flood extent or flood probability. Unreported cells are unknown rather than observed dry. The scores use only pre-day report history and static terrain/OSM/WorldCover features; Q4 report outcomes are displayed only as an overlay and for evaluation. The cyan outlines use those Q4 outcomes retrospectively and are never model inputs. The CNN mask arm was examined on Q4 in earlier model selection, so its improvement remains exploratory.

Rebuild with [flood_citywide_q4_queue_map.py](flood_citywide_q4_queue_map.py). The [HTML template](flood_citywide_q4_queue_map_template.html) contains the slider and drawing logic. The build checks that the packed rank bounds reproduce the published default backtest, including fractional score ties. The generated HTML was opened in headless Chrome at both citywide and Bang Kapi date/difference settings.
