# Beating the prior-year report-history queue across Bangkok

Checked 22 October 2026. I selected a ranking recipe using January–September 2025, then scored October–December 2025. All candidate recipes used information from before each scored day except the optional **same-day full-calendar-day rain** term. This is a report-capture comparison, not verified flood extent.

## Result

| Queue and score | Candidate cells | Q4 report-positive cell-days | Captured in daily top 10% |
| --- | ---: | ---: | ---: |
| All Bangkok: preceding 365-day report count | 25,812 | 7,209 | 47.18% |
| All Bangkok: recent + nearby report history, no rain | 25,812 | 7,209 | 48.17% |
| All Bangkok: recent + nearby + rain | 25,812 | 7,209 | 48.36% |
| Outside old patches + 750 m: preceding 365-day report count | 13,302 | 3,605 | 44.92% |
| Outside old patches + 750 m: recent + nearby, no rain | 13,302 | 3,605 | 46.08% |
| Outside old patches + 750 m: recent + nearby + rain | 13,302 | 3,605 | 46.16% |

The selected score beats prior-year history by **1.17 percentage points citywide**, capturing about **85 additional report-positive cell-days** at the same daily budget. Outside the old training patches it gains **1.23 points**, or about **44 additional cell-days**. Exploratory paired seven-day-block 95% intervals for those gains are **+0.63 to +1.69** and **+0.37 to +2.26 points**, respectively. Within their own top-10% district queues, it wins in **31 of 50 districts** (18 losses, one tie); in the 37 districts with positives outside the old patches, it wins in **21** (13 losses, three ties).

The **no-rain score also wins**: +0.99 points citywide and +1.16 points outside old patches. The extra same-day rain term adds only **0.19** and **0.08 points** over that no-rain score; the paired intervals for those increments include zero. The no-rain version is the stronger operational benchmark because it requires no future-day rainfall.

## What the 10% queue means for field work

The 10% cutoff is a **standardized map shortlist**, not a measured daily BMA inspection capacity. The candidate grid contains 25,812 250 m cells, so 10% is about 2,581 shortlisted cells citywide each day. Across 50 districts, the median is 320.5 candidate cells (about 32 shortlisted); Nong Chok has 3,921 candidates (about 392 shortlisted). These counts cannot be treated as feasible daily crew visits without staffing, travel, and visit-time evidence. A field pilot would need a smaller route and staff-hour budget.

The [district-count audit](artifacts/flood_open_weak_train/multi_target/citywide/candidate_cells_by_district_250m.csv) uses the same candidate rule and greatest-overlap district assignment as `flood_citywide_history_ranker.py`: cells with more than 2% Bangkok area in each citywide tile are assigned to one district. Fractional 10% counts are analytical selection weights, not partial visits.

## How the score works

For each 250 m cell, let `H365` and `H90` count report-positive days in the preceding 365 and 90 days, excluding the scored day. Let `G(H365)` smooth the 365-day counts across nearby cells with a Gaussian width of one cell (250 m). The selected retrospective score is:

`log1p(H365) + 0.4 log1p(H90) + 0.4 log1p(G(H365)) + 0.25 log1p(Rday)`

`Rday` is full-calendar-day Open-Meteo rain in millimeters, sampled at one nearest district proxy per 20 km tile. The no-rain score drops the final term. There are no fitted location-specific coefficients: the candidate weights were compared on January–September 2025 capture. Earlier Q4 work informed this experiment, so the test remains exploratory.

| Validation, Jan–Sep 2025 citywide queue | Capture |
| --- | ---: |
| Prior 365-day count | 48.17% |
| Recent + nearby, no rain | 49.78% |
| Recent + nearby + selected rain weight | 50.16% |
| Best tested gradient-boosted tree blend | 48.12% |

The gradient-boosted model used report histories, SRTM terrain, OSM canals/roads, WorldCover land cover, season and rain. It did **not** beat the simple history score on validation and was not selected. The previously tested joint satellite/urban CNN captured **44.13% citywide** and **41.91% outside its old patches** on the same Q4 citywide queues; it still does not beat report history. This result improves complaint-site ranking through a more useful history score, not through demonstrated physical flood prediction.

## District detail

Each district has its own daily top-10% queue. The last three columns restrict the queue and positives to cells outside the old training patches plus a 750 m buffer. Small positive counts yield unstable district percentages.

| District | Q4 positives | History | Selected | New-area positives | History, new | Selected, new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phra Nakhon | 162 | 16.76% | 17.16% | 162 | 16.76% | 17.16% |
| Dusit | 49 | 49.77% | 46.94% | 49 | 49.77% | 46.94% |
| Nong Chok | 90 | 44.05% | 47.78% | 0 | — | — |
| Bang Rak | 34 | 12.13% | 13.23% | 34 | 11.98% | 12.65% |
| Bang Khen | 160 | 54.14% | 54.31% | 85 | 47.71% | 47.06% |
| Bang Kapi | 249 | 36.50% | 36.14% | 0 | — | — |
| Parthum Wan | 65 | 17.44% | 19.38% | 20 | 19.27% | 22.50% |
| Pom Prap Sattru Phai | 38 | 27.63% | 32.10% | 38 | 27.63% | 32.10% |
| Phra Khanong | 179 | 25.95% | 24.58% | 0 | — | — |
| Min Buri | 50 | 37.91% | 42.40% | 8 | 100.00% | 100.00% |
| Lat Krabang | 230 | 60.15% | 62.61% | 99 | 61.14% | 60.61% |
| Yan na wa | 37 | 28.98% | 32.70% | 8 | 28.36% | 37.50% |
| Samphanthawong | 43 | 24.65% | 24.65% | 43 | 24.65% | 24.65% |
| Phaya Thai | 38 | 19.41% | 18.42% | 12 | 18.72% | 25.00% |
| Thon buri | 49 | 33.27% | 38.78% | 49 | 33.27% | 38.78% |
| Bangkok Yai | 113 | 41.52% | 41.68% | 113 | 41.52% | 41.68% |
| Huai Khwang | 86 | 25.96% | 24.07% | 0 | — | — |
| Khlong San | 39 | 15.50% | 20.51% | 39 | 15.50% | 20.51% |
| Taling Chan | 154 | 46.02% | 46.75% | 154 | 46.02% | 46.75% |
| Bangkok Noi | 231 | 35.19% | 35.07% | 231 | 35.19% | 35.07% |
| Bang Khun thain | 348 | 44.17% | 47.96% | 348 | 44.17% | 47.96% |
| Phasi Charoen | 76 | 27.02% | 26.32% | 76 | 27.02% | 26.32% |
| Nong Khaem | 87 | 49.34% | 51.15% | 87 | 49.34% | 51.15% |
| Rat Burana | 41 | 35.52% | 36.59% | 41 | 35.52% | 36.59% |
| Bang Phlat | 85 | 25.16% | 24.71% | 85 | 25.16% | 24.71% |
| Din Daeng | 144 | 16.99% | 15.97% | 0 | — | — |
| Bueng Kum | 59 | 23.31% | 25.42% | 0 | — | — |
| Sathon | 48 | 33.82% | 34.17% | 45 | 29.02% | 31.11% |
| Bang Sue | 122 | 25.64% | 24.67% | 97 | 22.18% | 19.48% |
| Chatu Chak | 479 | 26.23% | 28.00% | 2 | 0.00% | 0.00% |
| Bang Kho laen | 39 | 24.67% | 24.36% | 39 | 24.67% | 24.36% |
| Pra Wet | 883 | 38.23% | 40.02% | 154 | 37.44% | 36.62% |
| Khlong Toei | 80 | 26.14% | 30.00% | 0 | — | — |
| Suanluang | 271 | 33.55% | 35.28% | 0 | — | — |
| Chom Thong | 187 | 20.88% | 21.93% | 187 | 20.88% | 21.93% |
| Don Mueang | 244 | 35.64% | 37.30% | 244 | 35.64% | 37.30% |
| Ratchathewi | 25 | 27.81% | 28.00% | 15 | 40.33% | 40.00% |
| Lat Phrao | 99 | 38.57% | 38.48% | 0 | — | — |
| Vadhana | 173 | 28.09% | 28.44% | 0 | — | — |
| Bang Khae | 244 | 33.85% | 35.57% | 244 | 33.85% | 35.57% |
| Lak Si | 219 | 39.84% | 39.27% | 93 | 27.84% | 27.85% |
| Sai Mai | 41 | 50.69% | 56.10% | 41 | 50.69% | 56.10% |
| Khan Na Yao | 30 | 13.69% | 20.00% | 0 | — | — |
| Saphan Sung | 263 | 34.92% | 36.12% | 0 | — | — |
| Wang Thong Lang | 109 | 24.45% | 23.85% | 0 | — | — |
| Khlong Sam Wa | 42 | 40.32% | 50.00% | 22 | 43.52% | 45.45% |
| Bang Na | 151 | 28.79% | 28.68% | 117 | 36.27% | 38.46% |
| Thawi Watthana | 107 | 51.84% | 46.54% | 107 | 51.84% | 46.54% |
| Thung khru | 180 | 34.68% | 32.22% | 180 | 34.68% | 32.22% |
| Bang Bon | 237 | 51.96% | 51.39% | 237 | 51.96% | 51.39% |

## Limits and next benchmark

Follow-up: the [Bangkok canal-connectivity test](FLOOD_CITYWIDE_CONNECTED_RELATIONS.md) builds a graph reaching beyond nearby cells, but its current static topology feature does not establish a ranking gain over the no-rain score.

- Traffy complaint cells are positive observations. Unreported cells have no verified dry label, and capture of complaints does not prove capture of inundated area.

- The selected rain term can use hours after the complaint and is only constant across each 20 km tile. It does **not** validate a two-times-rain flood scenario. Rain adds a small, statistically uncertain increment here.

- Q4 2025 was inspected during earlier project work, so these paired intervals are exploratory rather than a clean prospective test. Recipe weights were selected before this run’s Q4 scoring, but Q4 is not a pristine untouched period for the project.

- Any next CNN, GNN or rain-scenario model should beat the **no-rain recent + nearby history queue** under the same daily budget, then show added value on independently verified water extent and rainfall scenarios.

## Reproduce

Run `.venv/bin/python flood_citywide_history_ranker.py`. The [script](flood_citywide_history_ranker.py), [complete metrics](artifacts/flood_open_weak_train/multi_target/citywide/q4_2025_history_ranker.json), and [citywide comparison](FLOOD_MULTI_TARGET_CITYWIDE_Q4.md) preserve the split, feature recipe, district results and limitations.
