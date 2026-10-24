# Recorded flood depth in the Bangkok backtest districts

Yes: [BMA's major-road flood reports](https://data.bangkok.go.th/dataset/frd_dds) include a `frd_flood_height` field, recorded in centimetres. I checked the October–December 2023–2025 reports for the same 11 districts as the [fixed-budget cell backtest](FLOOD_BANGKOK_INSPECTION_BACKTEST_EXPANDED.md). There are **110 BMA road-flood report rows**, **109 with a depth**. Median recorded depth is **15 cm**; **44/109** are at least 20 cm. The highest is **50 cm**, reported on **13 November 2025** on Ratchadaphisek Road around the Ratchayothin–Lat Phrao junction in Chatu Chak, with a reported flood duration of 4 hours 45 minutes.

| Test quarter | BMA road reports | With depth | Median depth | At least 20 cm | Maximum |
|---|---:|---:|---:|---:|---:|
| Oct–Dec 2023 | 23 | 23 | 15 cm | 6 | 20 cm |
| Oct–Dec 2024 | 14 | 14 | 15 cm | 4 | 20 cm |
| Oct–Dec 2025 | 73 | 72 | 15 cm | 34 | 50 cm |

The recorded values are heavily rounded: **17 reports at 10 cm, 48 at 15 cm, 42 at 20 cm, one at 25 cm, and one at 50 cm**. These are useful *severity categories* in the BMA road archive, but their precision and measurement method are not documented in the cached report rows.

## Coverage by district

These counts cover the three October–December quarters combined, not all flooding in each district.

| District | BMA reports with depth | Median | Maximum | At least 20 cm |
|---|---:|---:|---:|---:|
| Bang Khen | 4 | 15 cm | 20 cm | 1 |
| Bang Kapi | 8 | 15 cm | 20 cm | 1 |
| Chatu Chak | 18 | 17.5 cm | 50 cm | 9 |
| Pra Wet | 14 | 17.5 cm | 20 cm | 7 |
| Suanluang | 16 | 15 cm | 20 cm | 7 |
| Don Mueang | 5 | 15 cm | 20 cm | 1 |
| Bang Khae | 13 | 15 cm | 25 cm | 5 |
| Lak Si | 17 | 15 cm | 20 cm | 6 |
| Saphan Sung | 0 | — | — | 0 |
| Wang Thong Lang | 6 | 15 cm | 20 cm | 2 |
| Bang Na | 8 | 20 cm | 20 cm | 5 |

One Pra Wet report has no depth. Saphan Sung has no BMA major-road report in these quarters, despite Traffy flood complaints. This difference shows why **no BMA report cannot be read as zero depth**.

## Can depth be checked against the ranked cells?

The **2,454 Traffy complaint points** in the backtest have no depth field. The BMA depth rows contain road and area text but no event coordinates, so they cannot be assigned reliably to those 250 m complaint cells across all 11 districts. They also cover selected major roads rather than every street or non-road surface.

There is one small location check: six named Bang Kapi sites flooded in October–December 2025 can be attached to **approximate** cells using BMA's **2026 risk-point inventory**. This inventory postdates the flood events; it is not proof of the exact 2025 wet cell.

| Named Bang Kapi site | Q4 BMA site-days | Recorded depth, cm | Local tree rank | Recent reports rank |
|---|---:|---:|---:|---:|
| Ramkhamhaeng Soi 43 opposite | 1 | 20 | 1 | 2 |
| Bang Kapi intersection | 1 | 15 | 13 | 12 |
| Ramkhamhaeng Soi 33 | 1 | 15 | 17 | 18 |
| Ramkhamhaeng Soi 20 | 1 | 10 | 6 | 9 |
| Hua Mak / Ramkhamhaeng 24 | 2 | 15, 15 | 202 | 439 |
| Lam Sali intersection | 2 | 10, 15 | 141 | 201 |

At a 25-cell budget, **both ranking rules include the same four of six named sites**, including the only 20 cm site; both miss the two sites with 15 cm maxima. Six sites and approximate coordinates are too little evidence to claim that either ranking detects *deeper* floods better.

A separate [pre-October CNN depth-capture comparison](FLOOD_BANGKOK_CNN_DEPTH_CAPTURE.md) now checks the same six sites against a spatial CNN and the recent-report queue at equal cell budgets. At 25 cells they tie on depth-weighted capture; at 10 cells the CNN includes one additional 15 cm site.

An [exploratory joint classification-and-depth CNN](FLOOD_BANGKOK_MULTITASK_DEPTH_CNN.md) also trained on conservatively matched BMA site-days across the eleven districts. Its held-out depth error was worse than a constant 15 cm prediction, reflecting the sparse and selected depth labels rather than a validated water-depth map.

The existing backtest remains a test of **complaint-location capture**, not depth prediction. A depth-weighted cell backtest would require the BMA road locations to be geocoded and manually checked against the grid, with enough matched events outside Bang Kapi. A separate depth prediction model would additionally need observations for low-depth and dry locations, plus event-timed rain and drainage state.

## Reproduce

Run `.venv/bin/python flood_bangkok_depth_audit.py`. Review the [metrics](artifacts/flood_bangkapi_surface/surface_bma_depth_audit_metrics.json), [110 BMA report rows](artifacts/flood_bangkapi_surface/surface_bma_road_depth_q4_2023_2025.csv), and [six-site Bang Kapi comparison](artifacts/flood_bangkapi_surface/surface_bma_depth_bang_kapi_2025_sites.csv). The script reads the cached public BMA CSVs and the already exported rank and site-match files; it does not infer depths for Traffy cells.
