# Flood-map transfer test: six Thai regions

This experiment asks whether a spatial model trained in several places can rank mapped flood-water locations in a **completely unseen region**. It ranks all surface cells, including off-road land. It does **not** estimate flood probability: GISTDA supplies positive polygons, while cells outside polygons have no verified dry label.

## Data and method

- Six fixed 20 × 20 km patches at 250 m resolution: Nong Chok, Pathum Thani, Chachoengsao, Ayutthaya, Nakhon Sawan, and Phitsanulok. Each patch has 6,400 cells. The last two are farther from Bangkok and include different terrain.
- [GISTDA daily flood products](https://data.go.th/dataset/flood-disaster-data): eight dates from 27 September to 20 October 2026; 45 of the 48 patch-dates contain positive polygons. Five additional positive patch-dates from October–December 2025 were inspected separately and then tried as extra training data. A missing polygon is **unknown**, not dry.
- Static inputs: [SRTM elevation](https://registry.opendata.aws/terrain-tiles/), local relative elevation and slope; [OpenStreetMap](https://www.openstreetmap.org/) road and canal length/distance from the local January 2025 extract.
- Weather: [Open-Meteo historical precipitation](https://open-meteo.com/en/docs/historical-weather-api) at one grid point per patch, represented by daily rain, hourly peaks, wet hours, and previous 24/72-hour rain. Month is encoded cyclically.
- Sentinel experiment: [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) images from January–April 2025, before every label. Red and near-infrared bands produce NDVI. Cloud/shadow pixels are masked with the scene-classification layer. The extra channels are mean NDVI, fraction of valid subpixels with NDVI above 0.3, and valid-pixel fraction. The six patches have 92–98% valid composite coverage.
- Training loss allocates a conditional spatial density to positive GISTDA polygon area inside each patch. Evaluation asks: **what fraction of mapped positive area falls in the highest-ranked 10% of cells?** This is area capture, not precision, verified flood extent, or probability.

For each fold, five regions train the CNN on five 2026 dates; two later dates from those regions select the training epoch. The sixth region is excluded from training and normalization, then evaluated on all eight dates. The 2025 augmentation experiment adds the five positive 2025 patch-dates to training **only when their region is not held out**. The 2025 maps were chosen after checking overlap, so that experiment is exploratory.

## Held-out-region results

Area captured in the top 10% of cells; random selection has an expected value of 10%.

| Held-out region | Base CNN | CNN + Sentinel vegetation | Base + 2025 labels | Sentinel + 2025 labels |
| --- | ---: | ---: | ---: | ---: |
| Nong Chok | 26.9% | 26.8% | 28.6% | 29.0% |
| Pathum Thani | 46.9% | 47.8% | 44.8% | 46.4% |
| Chachoengsao | 19.9% | 17.3% | 19.5% | 15.7% |
| Ayutthaya | 23.1% | 18.6% | 21.5% | 18.8% |
| Nakhon Sawan | 20.5% | 39.4% | 22.1% | 34.8% |
| Phitsanulok | 37.2% | 52.5% | 37.3% | 52.8% |
| **Mapped-area weighted, 45 positive patch-days** | **24.0%** | **25.1%** | **24.1%** | **24.5%** |

The base model beats random selection and a low-elevation ranking (13.5% overall), but the vegetation effect changes by region. On the five separately selected 2025 patch-dates, models trained only on 2026 capture 20.0% without Sentinel and 17.5% with Sentinel. Adding 2025 data from the other cities gives 20.7% and 18.2% respectively. These are older-year stress checks, **not** forward-time forecasts.

The later-date test within already seen regions is much easier: on 20 October 2026, the base model captures 37.1% and the Sentinel model 50.6%; simply ranking cells by earlier GISTDA mapped-water history captures **68.7%**. This is a strong baseline for a city with past maps and shows how much repeated water locations matter.

## What greenness means here

Greenness is **not** a soil-infiltration measurement. High pre-event NDVI alone captures 44.4% of mapped water in Pathum Thani and 53.9% in Phitsanulok at a 10% cell budget; the mapped water often overlaps vegetated floodplain or agricultural land. In Ayutthaya, the association reverses: ranking low-NDVI cells captures 25.2%, versus 20.1% for high-NDVI cells. Treating all green cells as flood-protective would therefore be physically misleading. Sentinel vegetation is a useful land-cover clue, but its meaning depends on local land use and drainage.

## Limits and next decision

All six sites are in Thailand's central/northern plains and were selected after checking positive-map availability. GISTDA's item bounding box is not an observed-dry mask; its polygons are not verified street-water measurements. The products differ in sensor coverage between dates. BMA-clipped GISTDA source filenames name Sentinel-1, Sentinel-2, and other acquisitions with times, so feeding a contemporaneous image from the same source into the model would risk directly reproducing the label. SRTM has limited vertical accuracy on flat streets, the OSM snapshot predates the 2026 events, and a single weather grid point misses within-city rain variation. Whole-day rainfall may include hours after satellite acquisition because exact acquisition times for every polygon were not independently established. Therefore these models cannot yet validate a “double the rain” flood scenario or support calibrated flood probabilities.

The next data improvement is a documented observation footprint or independent wet **and dry** observations for multiple events in a pre-registered distant city. That would let us distinguish model transfer from repeated water geography and satellite coverage. For an operational inspection queue today, compare against local mapped-water history whenever that history exists.

## Reproducible artifacts

- Dataset and source audits: `artifacts/flood_gistda_multiregion/multiregion_data_audit.json`, `sentinel2_2025_audit.json`, `gistda_2025_patch_coverage_scan.json`.
- Model metrics: `gistda_multiregion_training_metrics.json`, `gistda_multiregion_2025_augmented_metrics.json`, `gistda_2025_frozen_model_test.json` in the same directory.
- Figure: `gistda_six_city_holdout_capture.png`; learning curves: `gistda_multiregion_learning_curves.png`.
- Scripts: `flood_gistda_multiregion_data.py`, `flood_gistda_multiregion_sentinel2.py`, `flood_gistda_multiregion_train.py`, `flood_gistda_multiregion_2025_test.py`, `flood_gistda_multiregion_augmented.py`.
