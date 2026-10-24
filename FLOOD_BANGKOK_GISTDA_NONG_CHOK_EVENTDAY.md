# Nong Chok GISTDA dates with event-day rain

The [earlier Nong Chok overlay](artifacts/flood_bangkapi_surface/gistda_20261004_nong_chok_rank_overlay.png) ranked cells using a saved **1 October 2025** canal-CNN scene with synthetic 20 mm / 2 h rain. I reran that same saved complaint-proxy CNN on all eight 2026 GISTDA-positive dates in Nong Chok, replacing the month and seven rain summaries with [Open-Meteo historical hourly precipitation](https://open-meteo.com/en/docs/historical-weather-api) requested at the district's representative point. Open-Meteo returned a grid point around **13.8137°N, 100.8364°E**, rather than a dense rain map across the district. The input is full-day historical rain in Bangkok local time, so this is an **explanatory hindcast**, not a forecast available before the satellite observation. The [4 October event-rain overlay](artifacts/flood_bangkapi_surface/gistda_20261004_nong_chok_eventday_rain_overlay.png), [daily metrics](artifacts/flood_bangkapi_surface/gistda_2026_nong_chok_eventday_metrics.json), and [cell scores](artifacts/flood_bangkapi_surface/gistda_2026_nong_chok_eventday_scores.csv) preserve the result.

The model was trained with the current year's per-cell prior flood-complaint days, but a 2026 Nong Chok extract was not available in the BMA open Traffy catalog checked on 22 October. I therefore ran two **input sensitivity cases**: the 1 October 2025 prior-complaint map held as a spatial surrogate, and all prior counts set to zero. Neither is verified 2026 history. Static terrain, OSM canal geometry, and BMA pump/gate locations remain mapped for Nong Chok. The saved model has no input for pump/gate operation, canal level, or a connected pipe network, and BMA's open daily canal-level CSVs checked on 22 October end on **16 July 2026**.

| GISTDA date | Nong Chok rain that day | Previous 72 h | Mapped water | Top 10% capture, 2025 prior surrogate | Top 10% capture, zero prior |
|---|---:|---:|---:|---:|---:|
| 27 Sep | 47.3 mm | 160.9 mm | 7.66 km² | 4.7% | 15.7% |
| 2 Oct | 1.2 mm | 4.5 mm | 37.83 km² | 7.6% | 14.0% |
| 3 Oct | 3.1 mm | 3.0 mm | 37.04 km² | 7.3% | 13.2% |
| **4 Oct** | **1.5 mm** | **4.9 mm** | **46.20 km²** | **6.6%** | **12.6%** |
| 7 Oct | 1.8 mm | 26.1 mm | 22.29 km² | 4.8% | 10.9% |
| 9 Oct | 0.5 mm | 24.0 mm | 28.15 km² | 6.5% | 10.9% |
| 10 Oct | 5.4 mm | 3.5 mm | 23.30 km² | 5.9% | 12.8% |
| 20 Oct | 3.2 mm | 26.5 mm | 11.40 km² | 1.9% | 6.5% |

The original synthetic-rain 4 October top-10% capture was **7.4%**. Date-matched rain changes the spatial ranking only moderately with the same prior map (Spearman **0.952** against the original rank). The large change under zero prior (12.6% capture) demonstrates that the unavailable 2026 per-cell complaint-history channel prevents a unique 2026 model evaluation. Zero is a sensitivity bound, not an observed value. The 27 September preceding 72-hour total, 160.9 mm, also exceeds the maximum 111.4 mm in the model's 2022–July 2025 training rain summaries.

The low 4 October district-point rain does not mean the GISTDA polygon is wrong or that all water arose that day. **67.5%** of its Nong Chok mapped area also appears in the 2 October product, and **71.4%** appears on 3 October. These overlaps show repeated mapped water, but the catalog does not provide a verified acquisition time, dry/coverage mask, or cause. A single gridded rain point is also a coarse summary for the large Nong Chok district. The GISTDA products often include rice-area records, whereas the CNN learned street-complaint occurrence.

**Infrastructure status:** This run uses Nong Chok-specific mapped structures, but they are static proxies from earlier catalogs. The open BMA pump/gate location catalog was last modified in 2023; the pump-capacity catalog is 2023; the canal-level CSVs end in July 2026. The [HII open water-level archive](https://tiservice.hii.or.th/opendata/data_catalog/water_level/2026/) has September station files but its October 2026 folder was empty when checked on 22 October. I found no public October 2026 pump/gate operating-state records in BMA's catalog. Therefore the map cannot be called a weather-and-infrastructure-state-matched hindcast. Date-matched rain alone does not resolve the target mismatch or missing operational inputs.

Reproduce with `.venv/bin/python flood_bangkok_gistda_nong_chok_eventday.py` after the GISTDA grid backtest. The [cached Open-Meteo response](data/flood_bangkok_transfer/rain_1003_2026_gistda_events.json) keeps the exact hourly series used.
