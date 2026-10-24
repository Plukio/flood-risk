# Can open satellite imagery improve the Bang Kapi ranking?

Run `.venv/bin/python flood_bangkapi_imagery_audit.py`. The reproducible
[date-by-date scene audit](artifacts/flood_bangkapi_imagery_audit/scene_timing_by_report_date.csv)
compares BMA's 2025 Bang Kapi flood start/end times with open
[Sentinel-1 RTC](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc)
and [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a)
scene metadata over the 13 named locations. Counts below are for 12 distinct
BMA report dates, not 44 individual report rows. Times are converted to
Bangkok time (UTC+7). Scene intersection and timing do **not** establish local
cloud-free pixels or flooded-road visibility.

| 2025 archive check | Sentinel-1 radar | Sentinel-2 optical |
|---|---:|---:|
| Scenes intersecting the Bang Kapi study box during 2025 | 84 | 87 |
| BMA report dates with a scene on the same local date | 3 of 12 | 6 of 12 |
| Dates with a pass during any reported flood interval | **1 of 12** | **0 of 12** |
| Same-day scenes with scene-wide cloud cover below 20% | Not applicable | **0 of 6 dates** |

The one concurrent Sentinel-1 pass was **7 October 2025 at 18:29**, during a
reported Bang Kapi intersection flood. The [road/radar pilot](FLOOD_ROAD_DATA_PILOT.md)
found no darkening candidate at a road midpoint in that 1 km² tile. Radar
backscatter can change for many reasons, and flooded road strips are often
narrower than its approximately 10 m pixels. Sentinel-1 can still help find
larger inundated open areas when pass timing works; it cannot currently label
most Bang Kapi road events in this archive.

Sentinel-2 daytime passes on report dates occurred **before** the recorded
flood intervals, and their scene-wide cloud fractions were 51–100%. The
cloud percentage is for the full scene, not a measurement over a given road.
There were **four pre-September 2025 scenes below 10% scene-wide cloud**, so
pre-event optical imagery is available for *static* context. One near-clear
example is 15 January 2025. Useful candidate features are vegetation cover,
built-up/impervious surface proxies, open water and changes in construction
around each drainage catchment. These require cloud/shadow masking and should
be computed only from imagery acquired before the prediction date. Optical
imagery cannot reveal a blocked underground pipe or pump operation.

For *storm input*, [GPM IMERG](https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V07)
provides half-hourly satellite precipitation on an approximately 0.1° grid.
That can add rain timing where a gauge is missing, but a grid cell is roughly
10 km across, much larger than the spacing between these Bang Kapi locations.
It cannot by itself rank adjacent streets. A quality-checked local rain gauge
or weather-radar rainfall grid would be more useful for street scenarios.

## What to add next

1. **Rain over time:** Use quality-checked hourly gauges and a gridded rain
   estimate as a fallback. Keep 1-hour peak, duration, total and previous-day
   rain. Compare each source against known storm records before fitting.
2. **Drainage and outfall:** Join each road's LiDAR-derived contributing area
   to canal, gate and pump locations. BMA has an open
   [pump/gate station table](https://data.bangkok.go.th/dataset/water-station)
   with coordinates and capacity. Its
   [drain-pipe table](https://data.bangkok.go.th/dataset/drain-pipe) has pipe
   dimensions and road/from/to names, but no coordinates in the CSV we
   inspected; many rows are continuation or subtotal rows. It was published
   in 2026, so it needs temporal checks before a 2025 backtest. Neither table
   supplies historical pump-on/off or blockage state.
   A later [BMA ArcGIS drainage audit](FLOOD_BANGKOK_DRAINAGE_GIS.md) found a
   separate public **geocoded 2021 pipe layer**; the CSV's missing geometry
   does not mean pipe paths are unavailable. The GIS still lacks verified
   street-inlet and operating-outfall links.
3. **Static satellite context:** Derive cloud-masked vegetation and built-up
   summaries from the clear pre-season Sentinel-2 scenes over each **upstream
   catchment**, not only a circle around an inventory point. Try these as
   additions to recent-report and drainage baselines, with the same
   September–December time split.
4. **Better outcomes:** Obtain time-stamped road photos, field checks or an
   open geotagged complaint stream to verify both wet and dry roads. BMA
   reports are positives; a missing report is not a dry-road observation.

With only ten sites in the forward Bang Kapi check, a CNN or U-Net would have
far more parameters than useful labels. Expand the site and storm sample across
districts, hold out whole storms and districts, and require an improvement over
the existing **Spearman 0.528 recent-report baseline** before claiming better
street ranking. Test doubled-rain scenarios separately against storms with
different measured intensities and durations; no satellite image in this audit
validates that counterfactual.
