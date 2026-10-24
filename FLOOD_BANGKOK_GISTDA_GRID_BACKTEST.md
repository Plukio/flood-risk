# GISTDA mapped water on the Bangkok 250 m grid

**Nong Chok follow-up:** The [event-day rain check](FLOOD_BANGKOK_GISTDA_NONG_CHOK_EVENTDAY.md) replaces the synthetic weather on all eight GISTDA-positive dates and audits the unavailable October infrastructure operating state.

I used the [GISTDA daily flood catalog](https://data.go.th/dataset/flood-disaster-data) as a **positive mapped-water layer**. I searched every 2026 daily collection available through 20 October, queried each item's actual polygon asset in a broad Bangkok box, and intersected the polygons with BMA boundaries. Of 52 listed days, 51 had a usable daily item. Eight had polygons inside BMA. The 26 September collection was listed but did not return a usable data item; it is **unchecked**, not a zero-water observation. [Scan inventory](artifacts/flood_bangkapi_surface/gistda_2026_bma_daily_scan.json).

| GISTDA product date | BMA mapped area (km²) | 250 m cells touched |
|---|---:|---:|
| 27 Sep 2026 | 10.60 | 648 |
| 2 Oct | 58.39 | 2,465 |
| 3 Oct | 51.95 | 2,666 |
| 4 Oct | 63.56 | 2,964 |
| 7 Oct | 29.79 | 1,612 |
| 9 Oct | 38.05 | 1,456 |
| 10 Oct | 32.84 | 1,464 |
| 20 Oct | 15.61 | 1,052 |

The grid joins use the exact 250 m cells used by the citywide complaint models, clipped to each district. Every cell has `mapped_wet_area_m2` and an `observation_status`: `mapped_water` when a polygon covers more than 1 m², otherwise **`unknown`**. The computed sum of mapped cell areas matches the clipped source polygons to within 2% on every date. The [8 daily grid CSVs](artifacts/flood_bangkapi_surface/gistda_20261004_grid_250m.csv) include all 28,309 valid citywide cells per date; the [wet-cell GeoJSON](artifacts/flood_bangkapi_surface/gistda_20261004_wet_cells_250m.geojson) and [Nong Chok overlay](artifacts/flood_bangkapi_surface/gistda_20261004_nong_chok_rank_overlay.png) show the spatial result for 4 October. [Full metrics](artifacts/flood_bangkapi_surface/gistda_2026_positive_only_grid_backtest.json).

The saved models used the **same feature definitions across districts, with each district's own mapped values**. The canal CNN includes 250 m BMA terrain summaries, mapped pump/gate proximity, prior complaint days, season/rain summaries, and local OSM canal length/distance; its five additional pump-capacity, pond, and underpass channels were masked. The CNN–GNN starts from a frozen 17-channel terrain/rain CNN, adds those seven static canal/asset channels at graph nodes, and exchanges messages along terrain-inferred and OSM-canal links. The graph is built separately within each district and has no cross-district links. GISTDA polygons were **outcomes for this check**, never model inputs or training labels. The 2026 comparison reuses saved scores from a fixed 1 October 2025 synthetic-rain scene, so it does not update prior complaints, asset operation, or local rainfall to the actual GISTDA dates.

## Positive-only ranking check

For each district with mapped water on each date, I allocated a fixed 5%, 10%, or 20% of its valid cells by rank and measured the fraction of **mapped water area** in that queue. Fractional allocation at score ties gives an expected capture instead of arbitrary cell-ID tie breaking. The baseline is the number of direct Traffy flood-complaint-positive days in each cell during 2025. The saved canal CNN and CNN–GNN ranks use their existing **1 October 2025 spatial scene with synthetic 20 mm / 2 h rain**. Those saved scores are spatial references; they are **not** date-matched 2026 forecasts. Random cell selection has 5%, 10%, and 20% expected area capture at these budgets.

| Ranking | Top 5% area capture | Top 10% | Top 20% |
|---|---:|---:|---:|
| 2025 complaint history | 0.8% | 3.4% | 13.4% |
| Saved canal CNN reference | 1.6% | **5.7%** | 13.2% |
| Saved CNN–GNN reference | 1.9% | 4.3% | 7.4% |

These percentages weight each district-date result by mapped area, totaling **300.8 km² of area-days**; the same place can appear on several dates. On the largest day, 4 October, Nong Chok had 46.20 km² mapped. Its top 10% canal-CNN cells captured 7.4% of that area, against 4.2% for complaint history. The [overlay](artifacts/flood_bangkapi_surface/gistda_20261004_nong_chok_rank_overlay.png) shows much of the mapped water north of the CNN's selected cells.

## What the comparison means

This is a poor match to the **street-complaint target** used to train the CNNs, not proof that the CNNs fail at verified road flooding. GISTDA's source polygons include a `rice_area` field greater than zero on polygons accounting for **93%–99%** of clipped mapped area across the eight dates. This means the *source polygons* intersect rice-area records; it does not prove that every square metre in those polygons is rice land. Their `length_road` field is greater than zero on polygons accounting for **14%–39%** of mapped area. That flag does not identify which wet pixels are road surface or whether a road was impassable. [Source-field audit](artifacts/flood_bangkapi_surface/gistda_2026_bma_landuse_flags.json).

The daily product metadata gives a product date and a source filename such as `rd2_20261020_0613`, but it does not establish a sensor acquisition time, completeness mask, or verified dry footprint. A cell outside a polygon cannot be labeled dry. The missing 2026 complaint-history input also prevents a clean 2026 run of the frozen CNN with date-matched rain. Open-Meteo historical hourly rain was accessible for these dates, so rainfall itself is not the remaining data barrier. We should not calculate dry-cell accuracy, average precision, Spearman against zero-filled cells, or calibrated flood probability from this layer.

**Useful next label:** intersect these polygons with road geometry and independently check a sample of road-adjacent wet cells against timestamped cameras, BMA reports, or field photographs. Keep unobserved cells unknown. The broad GISTDA layer can then support positive road-water examples; an acquisition/coverage mask and verified non-flooded road observations would be needed for a full flood/no-flood validation.

Reproduce with `.venv/bin/python flood_bangkok_gistda_2026_scan.py`, then `.venv/bin/python flood_bangkok_gistda_grid_backtest.py`, `.venv/bin/python flood_bangkok_gistda_landuse_audit.py`, and `.venv/bin/python flood_bangkok_gistda_rank_map.py`. The public example key is read from data.go.th at runtime and not stored in output artifacts.
