# Better data for a rain-responsive Bangkok flood model

Checked 22 October 2026, following the [province expansion](FLOOD_GISTDA_PROVINCE_EXPANSION.md) and [masked completion test](FLOOD_GISTDA_MASKED_COMPLETION.md). The main question is whether the model can rank *new floodwater* under a rain event in Bangkok, including built surfaces. The existing top-10% GISTDA test measures a different, mostly agricultural mapped-water target.

## What the current target contains

Within the BMA portion of the held-out Nong Chok patch, the eight 2026 GISTDA dates contain **226.23 km² of mapped-area-days**. At the 250 m cell scale, WorldCover fractions show:

| Cell context | Share of mapped positive area | Share of candidate BMA cells |
| --- | ---: | ---: |
| More than 50% crop | **91.3%** | 42.0% |
| More than 75% crop | 76.2% | 25.3% |
| More than 20% built | **2.1%** | 24.4% |
| More than 50% built | 0.06% | 7.0% |

These are **cell-level associations**, not a 10 m overlay of each water polygon. They explain why ranking by crop cover captures 44.3% of mapped water in the top 10% of cells. Repeating or masking these maps increases training views without adding much evidence about paved-street flooding or rain response.

## Priority order

| Priority | Data upgrade | What it would unlock | Open-data status and immediate check |
| --- | --- | --- | --- |
| **1** | **Reliable observation target:** acquisition time, sensor, valid footprint, and water/non-water mask for each [GISTDA daily product](https://data.go.th/dataset/flood-disaster-data) | Separate observed dry land from unobserved land; learn *temporary* water at a known time and align rain before it | The checked STAC item has GeoJSON positive polygons and a vector-tile asset. Its `datetime` is a product-day value; `created` is publication time. Neither is verified acquisition time. No valid-observation footprint is exposed. Trace source scene IDs and coverage before another flood-probability training run. |
| **2** | **Urban outcome labels:** [BMA major-road flood records](https://data.bangkok.go.th/dataset/frd_dds), [Traffy flood reports](https://data.bangkok.go.th/dataset/traffy-fondue-2568), and time-stamped visible wet/dry ground from accessible public cameras | Train and test the street-flood question rather than agricultural extent; use BMA depth as a separate severity check | BMA provides 1,435 positive road records for 2021–25, often with depth, but road/area text rather than event coordinates. Traffy provides many geolocated positives but reporting bias and no verified dry cells. [FloodCheck camera streams](FLOOD_BANGKOK_VERIFIED_EXTENT_PLAN.md) have no open historical frame archive. Start with conservatively located positives and keep unknown areas unknown. |
| **3** | **Storm timing and spatial rain:** quality-screened [HII hourly gauges](https://tiservice.hii.or.th/opendata/data_catalog/hourly_rain) and, where feasible, radar or gridded rain for gaps | Replace one patch-wide estimate and full-day rain with rainfall up to the observation cutoff, including intensity, duration, and antecedent rain | HII's September 2026 archive is accessible. Among seven catalog stations within 20 km of Nong Chok, five September files returned data, but each had only 23 valid hourly values on 27 September under the existing strict daily rule. October 2026 monthly files returned 404 on 22 October; recheck later. The BMA catalog result found here is monthly district rain, too coarse for an hourly event model. |
| **4** | **Urban terrain and surface features:** [BMA 10 m LiDAR DTM](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson), dated built cover, roads, canals, and connected contributing area | Better represent low points and runoff-producing surfaces | LiDAR is already usable inside Bangkok; the province GISTDA CNN used 30 m SRTM instead. Dated [Sentinel-2](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) and [Dynamic World](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1) can check changes since 2021 WorldCover, but rainy-season cloud and acquisition dates need screening. Better static features will not repair a mismatched label. |
| **5** | **Time-matched drainage state:** connected canal/outfall levels, gate position, pump operation, tide | Explain why the same rain produces different water persistence | BMA's open canal-level files begin in 2026 and contain daily maxima; the public pump/gate files describe equipment, not historical operation. They cannot yet supply a same-hour connected-gate state for historical backtests. |

## Most useful next experiment with open data

1. **Build a target audit before retraining.** For each GISTDA water product, record source scene, acquisition time, valid observed footprint, land-cover mix, and overlap with independent BMA or Traffy positive sites. If the source has no footprint or acquisition time, retain positive-only ranking; do not label outside-polygons dry.
2. **Run separate tests for open-land water and urban reports.** Keep the original fixed Bangkok GISTDA ranking for comparability, but add an urban target using conservatively located BMA road events and Traffy reports. Do not combine their scores into one claimed flood probability. Hold out whole storms and districts, and compare against crop cover, built cover, and report history.
3. **Use rain only before each label's timestamp.** Where gauges are missing, use a gridded estimate with its spatial and temporal resolution recorded. Add a rain-ablation and event-intensity test; changed predictions under doubled rain are sensitivity, not validation of that counterfactual.

The main gating item is **better labels with observation times and coverage**. More spatial tiles, more masks, or a larger CNN cannot identify a rainfall effect if the target mostly records persistent crop-area water and the true observation time is unknown.
