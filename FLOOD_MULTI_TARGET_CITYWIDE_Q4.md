# Bangkok-wide Q4 2025 report-ranking diagnostic

Follow-up: the [recent + nearby history ranker](FLOOD_CITYWIDE_HISTORY_RANKER.md) beats the 365-day count baseline on the same citywide queue; the joint CNN result below remains unchanged.

Checked 22 October 2026. The two frozen CNN checkpoints were scored on nine nonoverlapping 20 km tiles covering all 50 Bangkok districts. The evaluation uses 250 m cells and October–December 2025 Traffy report-positive cell-days. Both models received the same static layers, retrospective Open-Meteo rain, and prior 365-day report history. The joint model additionally received satellite-label gradients during training; the urban-only CNN did not.

For each day, the queue selects the highest-scoring 10% of candidate cells. Capture is the fraction of report-positive cell-days in that queue, splitting ties fractionally. These are **reports**, not independently verified water extent; an unreported cell is unknown rather than observed dry.

| Queue | Candidate cells | Q4 positive cell-days | Joint CNN | Urban-only CNN | Prior-year report history | Built cover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Citywide, all Bangkok | 25,812 | 7,209 | 44.1% | 44.0% | 47.2% | 25.0% |
| Citywide, outside training patches and 750 m buffer | 13,302 | 3,605 | 41.9% | 42.3% | 44.9% | 26.3% |

The joint-minus-urban-only difference is **+0.08 percentage points** citywide (exploratory seven-day-block 95% interval −0.96 to +0.88) and **−0.43 points** on new geography (−2.41 to +0.99). Joint trails report history by **3.06 points** citywide (interval for joint minus history −5.83 to −1.78) and **3.01 points** on new geography (−7.08 to −0.92). The satellite targets have **no demonstrated incremental benefit for Bangkok complaint ranking** in this check.

All 50 districts contain Q4 report-positive cells. Within their own daily top-10% queues, joint beats urban-only in 22 districts, loses in 25, and ties in 3. It beats prior-year report history in 16 and loses in 34. Outside the buffered training patches, 37 districts have at least one positive cell-day; joint beats urban-only in 14, loses in 16, and ties in 7. Many district counts are small, so individual differences are noisy.

![District comparison](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_citywide_district_differences.png)

## District detail

The first three capture columns use all cells in each district. The final three use only cells at least 750 m from the Bang Kapi and Nong Chok training patch footprints. A dash means there were no report-positive cell-days in that new-area subset. Each row has its own daily district queue, so district percentages differ from the citywide-queue summary above.

| District | All positives | Joint | Urban-only | History | New-area positives | Joint, new | Urban-only, new | History, new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phra Nakhon | 162 | 13.8% | 14.1% | 16.8% | 162 | 13.8% | 14.1% | 16.8% |
| Dusit | 49 | 46.9% | 46.9% | 49.8% | 49 | 46.9% | 46.9% | 49.8% |
| Nong Chok | 90 | 45.6% | 46.7% | 44.0% | 0 | — | — | — |
| Bang Rak | 34 | 16.2% | 20.6% | 12.1% | 34 | 15.6% | 20.6% | 12.0% |
| Bang Khen | 160 | 47.1% | 45.6% | 54.1% | 85 | 42.4% | 41.2% | 47.7% |
| Bang Kapi | 249 | 32.0% | 30.4% | 36.5% | 0 | — | — | — |
| Parthum Wan | 65 | 21.5% | 18.5% | 17.4% | 20 | 20.0% | 20.0% | 19.3% |
| Pom Prap Sattru Phai | 38 | 32.6% | 32.6% | 27.6% | 38 | 32.6% | 32.6% | 27.6% |
| Phra Khanong | 179 | 27.0% | 26.0% | 25.9% | 0 | — | — | — |
| Min Buri | 50 | 30.0% | 34.0% | 37.9% | 8 | 50.0% | 50.0% | 100.0% |
| Lat Krabang | 230 | 52.6% | 47.7% | 60.2% | 99 | 52.5% | 55.6% | 61.1% |
| Yan na wa | 37 | 13.5% | 16.2% | 29.0% | 8 | 37.5% | 38.8% | 28.4% |
| Samphanthawong | 43 | 19.1% | 14.4% | 24.7% | 43 | 19.1% | 14.4% | 24.7% |
| Phaya Thai | 38 | 23.7% | 26.3% | 19.4% | 12 | 20.0% | 25.0% | 18.7% |
| Thon buri | 49 | 32.7% | 28.6% | 33.3% | 49 | 32.7% | 28.6% | 33.3% |
| Bangkok Yai | 113 | 31.0% | 31.2% | 41.5% | 113 | 31.0% | 31.2% | 41.5% |
| Huai Khwang | 86 | 23.3% | 26.0% | 26.0% | 0 | — | — | — |
| Khlong San | 39 | 10.3% | 8.7% | 15.5% | 39 | 10.3% | 8.7% | 15.5% |
| Taling Chan | 154 | 43.5% | 40.3% | 46.0% | 154 | 43.5% | 40.3% | 46.0% |
| Bangkok Noi | 231 | 26.6% | 26.8% | 35.2% | 231 | 26.6% | 26.8% | 35.2% |
| Bang Khun thain | 348 | 40.5% | 43.7% | 44.2% | 348 | 40.5% | 43.7% | 44.2% |
| Phasi Charoen | 76 | 28.2% | 28.9% | 27.0% | 76 | 28.2% | 28.9% | 27.0% |
| Nong Khaem | 87 | 46.0% | 48.3% | 49.3% | 87 | 46.0% | 48.3% | 49.3% |
| Rat Burana | 41 | 41.5% | 36.6% | 35.5% | 41 | 41.5% | 36.6% | 35.5% |
| Bang Phlat | 85 | 25.9% | 24.7% | 25.2% | 85 | 25.9% | 24.7% | 25.2% |
| Din Daeng | 144 | 13.6% | 17.2% | 17.0% | 0 | — | — | — |
| Bueng Kum | 59 | 17.6% | 16.9% | 23.3% | 0 | — | — | — |
| Sathon | 48 | 12.5% | 16.7% | 33.8% | 45 | 13.3% | 15.6% | 29.0% |
| Bang Sue | 122 | 19.0% | 22.2% | 25.6% | 97 | 19.4% | 19.5% | 22.2% |
| Chatu Chak | 479 | 21.8% | 23.2% | 26.2% | 2 | 0.0% | 0.0% | 0.0% |
| Bang Kho laen | 39 | 20.5% | 20.5% | 24.7% | 39 | 20.5% | 20.5% | 24.7% |
| Pra Wet | 883 | 31.0% | 30.7% | 38.2% | 154 | 35.1% | 32.5% | 37.4% |
| Khlong Toei | 80 | 26.2% | 27.5% | 26.1% | 0 | — | — | — |
| Suanluang | 271 | 31.4% | 29.4% | 33.6% | 0 | — | — | — |
| Chom Thong | 187 | 20.9% | 23.0% | 20.9% | 187 | 20.9% | 23.0% | 20.9% |
| Don Mueang | 244 | 37.1% | 37.6% | 35.6% | 244 | 37.1% | 37.6% | 35.6% |
| Ratchathewi | 25 | 28.0% | 24.0% | 27.8% | 15 | 41.3% | 40.0% | 40.3% |
| Lat Phrao | 99 | 25.3% | 26.3% | 38.6% | 0 | — | — | — |
| Vadhana | 173 | 27.7% | 32.8% | 28.1% | 0 | — | — | — |
| Bang Khae | 244 | 38.0% | 37.7% | 33.8% | 244 | 38.0% | 37.7% | 33.8% |
| Lak Si | 219 | 39.4% | 37.3% | 39.8% | 93 | 26.9% | 25.8% | 27.8% |
| Sai Mai | 41 | 56.1% | 61.0% | 50.7% | 41 | 56.1% | 61.0% | 50.7% |
| Khan Na Yao | 30 | 6.7% | 3.3% | 13.7% | 0 | — | — | — |
| Saphan Sung | 263 | 34.0% | 34.8% | 34.9% | 0 | — | — | — |
| Wang Thong Lang | 109 | 27.3% | 27.5% | 24.5% | 0 | — | — | — |
| Khlong Sam Wa | 42 | 47.6% | 40.5% | 40.3% | 22 | 45.5% | 45.5% | 43.5% |
| Bang Na | 151 | 20.7% | 18.3% | 28.8% | 117 | 25.8% | 23.2% | 36.3% |
| Thawi Watthana | 107 | 43.7% | 42.1% | 51.8% | 107 | 43.7% | 42.1% | 51.8% |
| Thung khru | 180 | 27.1% | 29.4% | 34.7% | 180 | 27.1% | 29.4% | 34.7% |
| Bang Bon | 237 | 49.3% | 49.2% | 52.0% | 237 | 49.3% | 49.2% | 52.0% |

## Data and limits

- Geometry: nine 20 km tiles cover 1,583 km² of the BMA boundary; 25,812 candidate cells intersect BMA at more than 2% area. The buffered new-geography subset has 13,302 candidate cells. A cell touching the 750 m buffer at 50 m raster precision is excluded. This buffer matches the CNN’s three 3×3-convolution receptive-field radius.

- Static layers: SRTM elevation and terrain derivatives, 1 January 2025 OSM roads/canals, and ESA WorldCover 2021 fractions. The nine-tile [build audit](artifacts/flood_open_weak_train/multi_target/citywide/data_audit.json) records coverage.

- Rain: cached Open-Meteo gridded archive sampled at the representative point of the nearest district to each tile’s BMA portion. One rain series per 20 km tile is an approximation. Full calendar-day rain may occur after a complaint.

- Q4 Traffy labels were excluded from both CNNs’ urban training, but the Q4 period was inspected in earlier model development. The joint checkpoint used 2026 satellite labels, so this is a retrospective geography diagnostic rather than a prospective time test. No head gives calibrated flood probabilities or verified rain scenarios.

- The rolling baseline sums only reports in the preceding 365 days and excludes the scored day. The metric measures positive report capture under a fixed queue budget; it does not measure false alarms or true water extent.

## Reproduce

Run `.venv/bin/python flood_multi_target_citywide_data.py` then `.venv/bin/python flood_multi_target_citywide_eval.py`. The [evaluation script](flood_multi_target_citywide_eval.py) and [full JSON](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_citywide_report_ranking.json) preserve daily counts, district results, checkpoints, and source choices.
