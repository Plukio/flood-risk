# Lak Si flood-model feasibility backtest

Run the reproducible open-data experiment:

```bash
.venv/bin/python flood_laksi_backtest.py
```

The script downloads and caches source data in `data/flood_laksi/` and writes
`metrics.json`, a matched-incident audit, daily training features, 2025
predictions, and `street_ranking_2025.csv` to `artifacts/flood_laksi/`.

## What the model actually predicts

For each of six recurring Lak Si locations and each calendar day, the target is
**whether BMA recorded a road-flood incident there**. No report is an
**unobserved day**, not a verified dry road. The input includes the *complete*
day's observed rain, so this is a retrospective scenario hindcast, not an
advance warning test. The probabilities in the output are probabilities of a
**report under this dataset's reporting process**, not calibrated probabilities
of physical flooding.

The six candidate location names were identified from reports before 2025.
Coordinates are looked up from the later BMA point inventory solely to extract
terrain. Any site appearing for the first time in 2025 is excluded from the
2025 test. The exact BMA report-to-site matches are exported for inspection.

## Inputs and provenance

| Input | Source | Use and limit |
|---|---|---|
| Flood reports | [BMA road-flood files, 2021–2025](https://data.bangkok.go.th/dataset/frd_dds) | 95 Lak Si rows; 86 assigned by explicit place-name rules to the six recurring sites. No verified dry observations. |
| Elevation | [BMA LiDAR ImageServer](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson) | Exported 256 m tiles at 1 m output spacing; terrain features are point elevation and local relief. Native service pixel spacing is 0.2 m; vertical accuracy and datum have not been established. |
| Site coordinates | [BMA flood-risk-point inventory](https://data.bangkok.go.th/dataset/flood_risk26) | Geolocation only. Its later publication date makes it unsuitable for choosing held-out test sites. |
| Rain | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | Hourly `best_match` precipitation, 2021–2025. The six sites reduce to **two** grid cells; this is too coarse for local Bangkok storms. |
| Radar scenes | [Planetary Computer Sentinel-1 GRD catalog](https://planetarycomputer.microsoft.com/dataset/sentinel-1-grd) | Only an overpass-time audit. No SAR flood masks were inferred or used as labels. |

## 2025 holdout: street-location ranking

Models were fitted on 2021–2024 site-days and evaluated on **all of 2025**.
There are 8,766 training site-days with 62 report-positive labels, and 2,190
test site-days with 23 report-positive labels. The latter are 23 of the 29
Lak Si incident rows in 2025; six did not match one of the six pre-2025 sites.

Each site's daily predicted **report** probabilities are summed over 2025 to
produce its expected report-days. The table compares that ordering with the
held-out BMA report count and reported flood duration. Rows are ordered by the
rain + site model; the model never saw 2025 reports during training.

| Predicted rank | Street location | Actual rank¹ | Actual report-days | Actual flood-hours |
|---:|---|---:|---:|---:|
| 1 | Chaeng Watthana, Lotus | 1 | 7 | 5.63 |
| 2 | Chaeng Watthana, M.T.B. 11 | =4 | 2 | 2.67 |
| 3 | Chaeng Watthana, Administrative Court | =2 | 5 | 9.00 |
| 4 | Ngam Wong Wan, Pongphet | =2 | 5 | 14.67 |
| 5 | Chaeng Watthana Soi 10 | =4 | 2 | 3.00 |
| 6 | Ngam Wong Wan, Chin Khet | =4 | 2 | 3.58 |

¹ Actual rank is based on 2025 report-days; equal counts share a rank. The
model's expected report-day counts are in `street_ranking_2025.csv`.

The predicted order has **Spearman rank correlation 0.62** with 2025 reported
flood days. It gets 9 of 11 comparable site pairs in the correct order;
four of 15 pairs have tied observed counts. However, a simple ranking by
2021–2024 report counts achieves **exactly the same ordering and agreement**.
The rain + terrain model scores 0.37. Against total reported flood-hours, the
rain + site model's rank correlation is **0.03**: it misses that Pongphet had
the greatest reported duration. These are only six sites, so the correlations
are descriptive rather than evidence of generalizable skill.

The open rain data mostly distinguishes dry days from rainy days; among rainy
days, it has little power to distinguish which site floods. No-report labels
may hide unreported floods, so these are rankings of recorded incidents, not
measured flood hazard or false alarms. Two of the 23 reported 2025 site-days
had less than 1 mm in the open grid.

The median rainfall field in the matched 2025 BMA incident rows is 74.5 mm,
versus 8.2 mm for the corresponding Open-Meteo grid *calendar day*. These are
different measurement locations and time windows; they cannot be used as a
direct error estimate. The gap is a strong reason to obtain local high-frequency
rain archives before fitting a street-level storm-response model.

## Why a U-Net or CNN is premature

- A U-Net needs event-aligned flood-area masks. The Sentinel-1 catalog has 348
  Lak Si scenes across 2021–2025, but a scene falls within a reported flood
  interval on only **one of 42 distinct report dates**. Three dates have a
  scene within three hours of a reported interval. A coincident scene is not
  automatically a reliable urban flood mask.
- A terrain CNN would see only six distinct LiDAR patches and 62 positive
  training site-days. It would mostly memorize locations. The simpler model
  gives an honest measure of what the current labels support.
- CCTV coordinates are public, but historical camera frames are not included
  in the [BMA CCTV dataset](https://data.bangkok.go.th/dataset/bma-cctv).

## Next data gate

To train and evaluate a real road-flood model, add **historical local rain
gauges/radar at 15–60 minute intervals** and **verified wet/dry observations**
from archived CCTV or an agency incident log. Then define road-segment/storm
labels, keep entire storms and years held out, compare a rainfall-threshold
baseline with a terrain CNN, and use a U-Net only if dense flood-area masks can
be collected for many independent storms.
