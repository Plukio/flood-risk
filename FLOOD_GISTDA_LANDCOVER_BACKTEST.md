# Land cover and Thai SMAP in the six-region flood-map backtest

Run on 22 October 2026. This follows the [six-region GISTDA transfer test](FLOOD_GISTDA_TRANSFER.md) without changing its 250 m cells, eight 2026 product dates, geographic folds, CNN, or positive-area ranking metric. Each of six folds trains on five regions and ranks GISTDA mapped-water area in the sixth. The measure below is **mapped area captured in the highest-ranked 10% of cells**. It is not flood probability or precision: area outside GISTDA polygons is unverified.

## Held-out-region results

| Ranking | 2026, 45 positive patch-days | Five selected 2025 patch-days¹ |
| --- | ---: | ---: |
| Random selection, expected | 10.0% | 10.0% |
| Base CNN: terrain, OSM roads/canals, rain | 24.0% | 20.0% |
| Base CNN + pre-event Sentinel-2 greenness | 25.1% | 17.5% |
| Base CNN + WorldCover **built fraction only** | 22.7% | — |
| Base CNN + six WorldCover cover fractions | **29.2%** | 23.9% |
| Base CNN + six cover fractions + greenness | 26.9% | 22.0% |
| Base CNN + six cover fractions + Thai weekly SMAP | 26.6% | — |
| **WorldCover crop fraction alone; no model** | **32.0%** | **25.4%** |

¹ The 2025 positives were selected after checking GISTDA overlap. The CNNs were trained on 2026 and applied **backward** to 2025; this is an exploratory sensitivity check, not a forward-time forecast.

The six WorldCover channels are 2021 fractions of **built, crop, tree/grass, bare, wetland, and permanent water**. ESA 10 m map tiles covered 100% of all six patches. The same five training dates and two validation dates selected epochs as in the previous CNN; no held-out-region labels entered training or normalization. The [WorldCover extraction audit](artifacts/flood_gistda_multiregion/worldcover_2021_audit.json), [model metrics](artifacts/flood_gistda_multiregion/gistda_cover_cover_metrics.json), [simple baselines](artifacts/flood_gistda_multiregion/gistda_cover_simple_baselines.json), and [2025 stress check](artifacts/flood_gistda_multiregion/gistda_cover_2025_frozen_test.json) preserve the results.

| Held-out region | Base CNN | CNN + cover | Crop fraction alone |
| --- | ---: | ---: | ---: |
| Nong Chok | 26.9% | 30.7% | **37.1%** |
| Pathum Thani | 46.9% | 53.8% | **55.0%** |
| Chachoengsao | 19.9% | 22.1% | **25.1%** |
| Ayutthaya | 23.1% | 26.4% | **27.1%** |
| Nakhon Sawan | 20.5% | 37.0% | **37.5%** |
| Phitsanulok | 37.2% | **45.3%** | 38.6% |

Separate cover classes improve the CNN over its earlier inputs in all six held-out regions. But **crop fraction alone beats that CNN in five of six** and in the mapped-area weighted result. At 250 m, the mapped-positive-area-weighted cells are approximately **79.9% WorldCover crop**, versus **45.0% crop** across the six patches. About **85.2%** of mapped area falls in cells with over 50% crop cover; only **1.9%** falls in cells with over 20% built cover. These are approximate cell-fraction comparisons, not a 10 m polygon overlay. The gain therefore mostly reflects where this GISTDA product maps broad agricultural water. It is not evidence that the CNN has learned which paved Bangkok streets will flood under a given storm. Built fraction alone hurts this target: pavement can generate runoff that collects elsewhere, while mapped water here is concentrated on agricultural land.

For the easier 20 October 2026 test in **already seen regions**, cover CNN capture is **48.1%**; cover plus greenness is **54.5%**. Ranking by each region's earlier GISTDA mapped-water history captures **68.7%** in the same evaluation. The satellite CNN has not displaced that local-history baseline for known regions.

## Thai soil-moisture test

The public [GISTDA weekly SMAP catalog](https://data.go.th/dataset/disaster-platform-drought) supplied 21/28 September and 5 October 2026 products. Their polygon values rasterized across 100% of the six patches. Each event uses only a product created before local midnight on that event day. The 20 October input is **15 days old**; the other inputs are 2–6 days old. [The SMAP audit](artifacts/flood_gistda_multiregion/smap_2026_audit.json) records product dates, creation times, coverage, and age without embedding an API key.

SMAP lowers the geographic result from **29.2% to 26.6%**. Excluding 20 October still gives **29.1% cover versus 26.6% cover + SMAP**, so the stale date alone does not explain the decline. A coarse moisture product may help another target or a longer time series, but this test does not support keeping it in the present CNN.

## Decision for the model

- **Keep WorldCover's separate surface classes** as available, pre-event context. Also keep the simple crop-fraction ranking as a mandatory baseline for GISTDA mapped-water tests.
- **Do not promote built fraction, Sentinel-2 greenness, or SMAP as demonstrated gains** for unseen-region ranking on this target. They can remain candidate inputs for a different, verified urban-flood target.
- **Do not treat GISTDA-unmapped cells as dry.** The current labels and positive-only loss cannot calibrate a probability map or validate a doubled-rain scenario. For urban runoff, test paved area within each cell and its contributing area against dated road-water observations, with road and drain connectivity.

The [source scan](FLOOD_SATELLITE_SOURCE_SCAN.md) explains why the Thai 2 m RGB basemap, latest Thai crop polygons, Dynamic World, GPM, and near-event Sentinel-1 were not inserted in this historical run: their acquisition dates, access requirements, or same-source label leakage need resolving first.

## Reproduce

```bash
.venv/bin/python flood_gistda_multiregion_worldcover.py
.venv/bin/python flood_gistda_multiregion_smap.py
.venv/bin/python flood_gistda_multiregion_cover_train.py
.venv/bin/python flood_gistda_multiregion_cover_2025_test.py
.venv/bin/python flood_gistda_multiregion_cover_baselines.py
```

The extraction scripts require network access. The model uses cached GISTDA labels, rain, OSM, terrain, and Sentinel-2 inputs from the earlier backtest. The `cover_train.py` default runs the SMAP variant and therefore requires the SMAP extraction step first.
