# Can shared water levels explain the Bangkok flood-report bias?

**They are plausible physical inputs, but the available historical river level does not remove the undercount.** The target here is the daily count of 250 m cells touched by a 100 m buffer around a direct Traffy flood report. It is a reporting proxy, not measured inundation. The experiment uses full-day historical rain, so it is an explanatory hindcast or scenario input, not a day-ahead forecast.

## Open water-level data checked

| BMA catalog | Usable dates and measurement | Use in this backtest |
| --- | --- | --- |
| [Inner flood-barrier canal levels](https://data.bangkok.go.th/dataset/water_level_inner_daily) | Seven stations; daily maximum, metres relative to mean sea level; 1 Jan–16 Jul **2026** in the accessible CSV | **No 2022–2025 overlap.** Includes canals relevant to Bang Kapi and the western side, but cannot explain earlier held-out blocks. |
| [Outer flood-barrier canal levels](https://data.bangkok.go.th/dataset/water_level_outer_daily) | Three stations; daily maximum, metres relative to mean sea level; 1 Jan–16 Jul **2026** | No earlier overlap. |
| [Rama VIII Bridge, Chao Phraya](https://data.bangkok.go.th/dataset/river_rama) | Daily maximum river level and flow; 1 Jan 2022–4 Aug 2026; 1,677 rows, including **21 `NULL` levels** in May 2023 | Usable historical **river-boundary proxy**. The catalog does not state its level datum, so do not subtract it from LiDAR elevations or canal levels. |
| [Pak Khlong Talat, Chao Phraya](https://data.bangkok.go.th/dataset/river_daily_pak_khlong) | Daily maximum river level; 1 Jan–4 Aug **2026** | No earlier overlap. An [older Pak Khlong Talat file](https://data.bangkok.go.th/dataset/wl-max-chaophraya-river) spans Jan–Sep 2023 only. |
| [BMA five-minute water-level catalog](https://data.bangkok.go.th/dataset/flood) | Describes historical sensors; accessible catalog supplied a data dictionary but no measurement CSV | Cannot be joined to the current labels from the open catalog. |

The [Rama VIII source CSV](https://data.bangkok.go.th/dataset/acddf9a1-da51-46c6-8964-a61e9a500ae3/resource/f01c4ce7-a185-4b47-9e71-3af9bd95d27c/download/river_rama.csv) is snapshotted in [data/flood_bangkok_transfer/river_rama_daily_2022_2026.csv](data/flood_bangkok_transfer/river_rama_daily_2022_2026.csv). It provides a river level influenced by several processes; it is **not a sea-level gauge** or a local canal reading. BMA published the resource in August 2026, so the historical backtest does not establish when its old readings became available to a live forecaster. Among 2022–2025 model days, 1,439 of 1,461 have a usable prior-day level; missing values remain missing in the model. One isolated −1.00 m reading on 6 Jun 2023 is a possible data-quality issue and has not been silently corrected.

## Matched chronological test

For each district and year, train through July, adjust the predicted count using August–September labels, and test October–December. Compare the existing monotone Poisson model using rain and month with the same model plus **yesterday's river maximum** and the **mean of the preceding three daily maxima**. The river features are unconstrained and use no same-day river maximum; neither uses test labels. Rain features include the observed test day's full rainfall. The 2025 October–December block has already been examined during model development, so the result is exploratory.

| District | Test year | Observed report-positive cell-days | Rain only predicted | Rain + earlier river predicted | Daily MAE, rain → river |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bang Kapi | 2023 | 171 | 41.1 | 42.7 | 1.944 → 1.992 |
| Bang Kapi | 2024 | 82 | 48.1 | **72.5** | 1.074 → **1.060** |
| Bang Kapi | 2025 | 242 | 136.5 | **148.0** | 2.798 → **2.564** |
| Bang Khae | 2023 | 224 | 54.5 | 63.6 | 2.489 → 2.481 |
| Bang Khae | 2024 | 156 | **88.7** | 59.9 | 1.971 → 1.882 |
| Bang Khae | 2025 | 248 | **186.6** | 151.2 | **2.914 → 3.236** |

The prior-day river feature helps some district-years, but **all six blocks still underpredict**. In Bang Khae 2025, it increases October–December underprediction and worsens daily error. There is no evidence here that a single shared river-level input fixes the bias. The [full metrics](artifacts/flood_bangkapi_surface/surface_river_level_count_backtest_metrics.json) include daily Poisson deviance and rainy-day error; the [2025 daily predictions](artifacts/flood_bangkapi_surface/surface_river_level_count_backtest_2025_daily.csv) show which dates changed. Rebuild with `.venv/bin/python flood_bangkok_river_level_backtest.py`.

## More useful water-state feature

The physical quantity for a canal outlet is the **time-matched difference between inside-canal and outside-river levels**, together with gate state and pump capacity. A high outside level can reduce gravity drainage, but a closed gate or running pump changes that relationship. Daily maxima from two stations may occur at different hours; subtracting them would not estimate the actual hydraulic head during a storm. The 2026 canal CSVs can support a future temporal test after report labels and station connectivity are available; they cannot improve a 2022–2025 model without earlier readings.

The follow-up [shared-state experiment](FLOOD_BANGKOK_SHARED_STATE_NEXT.md) tests longer antecedent wetness, lagged non-flood reporting activity, and river-level interactions with static drainage proximity. The available features still do not yield consistently better daily count calibration or transferable cell ranking. A time-matched canal/outside-river head and actual gate/pump operation await timestamped measurements. Any claim about actual flood probability still needs independently observed wet and dry locations, rather than the absence of a report.
