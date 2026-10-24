# GISTDA flood-extent audit for Bangkok

**Update (22 October 2026):** The [250 m grid and positive-only ranking check](FLOOD_BANGKOK_GISTDA_GRID_BACKTEST.md) extends this initial audit across eight 2026 BMA-positive days. It finds that most mapped polygons intersect rice-area records and that the saved street-complaint rankings capture little of the mapped-water area.

I tested the [GISTDA Disaster Platform catalog](https://data.go.th/dataset/flood-disaster-data) against the BMA district boundaries. The catalog publishes an example API key in its resource links. Its STAC endpoints returned `Authentication Required` with `token=`, but **accepted the same public example key with `api_key=`**. The STAC root includes historical daily catalogs through **2025**, although the data.go.th page explicitly lists only the 2022/2023 catalog links. No key is stored in the downloaded artifacts.

The audit queried STAC item search for a broad Bangkok box (`100.30,13.55,100.95,13.98`), then queried each item's **actual GeoJSON flood-polygon asset** in that box. Finally it intersected each polygon with BMA's district boundaries. An item scene footprint touching the box does not mean water was mapped inside Bangkok.

| Product/time | Polygons returned in broad box | Polygons actually intersecting BMA | What they show |
|---|---:|---:|---|
| 2022 daily products | Four days with 34, 48, 77, and 71 polygons | **0** | All broad-box hits were outside BMA. |
| 2023–2024 daily products | No positive broad-box assets found | **0** | No event masks for BMA in the checked daily assets. |
| **7 Oct 2025** daily product | **0** | **0** | BMA recorded 14 major-road floods that day, including two in Bang Kapi. The GISTDA product did not cover them as mapped water. |
| **9 Nov 2025** daily product | 344 | **10** | About **0.150 km²** beside the Chao Phraya in Bang Sue/Bang Phlat. The source field is `OTH_20251109_0001, OTH_20251109_0002.shp`; it does **not** identify a satellite scene. BMA's major-road archive has no report on that date. |
| 11 and 23 Nov 2025 daily products | 10 on each date | **0** | Broad-box polygons lay outside BMA. |
| **20 Oct 2026** daily product | 1,702 | **667** | About **15.61 km²**, mainly Nong Chok (11.40), Lat Krabang (2.80), Khlong Sam Wa (0.77), and Min Buri (0.63 km²). The source field is `rd2_20261020_0613`; its exact sensor and time-zone interpretation were not established from the product metadata. |

The [20 October 2026 extent map](artifacts/flood_bangkapi_surface/gistda_20261020_bangkok_extent_map.png) and [clipped GeoJSON](artifacts/flood_bangkapi_surface/gistda_daily_20261020_bma.geojson) make the eastern-Bangkok footprint inspectable. The [9 November 2025 riverbank map](artifacts/flood_bangkapi_surface/gistda_20251109_bang_sue_map.png) shows why that small historical example needs independent checking before use as a street-flood target.

**Interpretation:** GISTDA supplies real, time-indexed flood-polygon products and recent Bangkok coverage. The mapped polygons are **positive candidate water extents**, not complete verified wet/dry truth. The API does not return a corresponding observed-dry or sensor-coverage mask in these GeoJSON assets; absence of a polygon cannot label a cell dry. The STAC `datetime` on daily items is `00:10Z`, which appears to identify the product day rather than independently confirmed image acquisition time. The 2025 overlap is too small and the only BMA-intersecting source is tagged `OTH`, so this audit does not yet provide dense historical training masks for the 2022–2025 CNN/U-Net backtest.

The immediate useful experiment is to collect **future daily** GISTDA products, keep their positive polygons as partial labels, and obtain the acquisition footprint/quality or no-data mask before assigning negatives. Independently check selected polygons against timestamped photos, cameras, BMA reports, or other imagery; preserve uncertain areas. For 2025, the Bang Sue/Bang Phlat shapes can be investigated as a small case study, while the Bang Kapi 7 October road flood remains unobserved by this product.

Reproduce a day with `.venv/bin/python flood_bangkok_gistda_extent_audit.py YYYY-MM-DD`. The date-specific [2026 audit](artifacts/flood_bangkapi_surface/gistda_daily_20261020_bma_audit.json), [2025 riverbank audit](artifacts/flood_bangkapi_surface/gistda_daily_20251109_bma_audit.json), [2025 Bang Kapi event audit](artifacts/flood_bangkapi_surface/gistda_daily_20251007_bma_audit.json), and [2022–2024](artifacts/flood_bangkapi_surface/gistda_2022_2024_bangkok_flood_asset_audit.json) / [2025](artifacts/flood_bangkapi_surface/gistda_2025_bangkok_flood_asset_audit.json) asset inventories preserve the checks.
