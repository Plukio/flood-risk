# Bang Kapi cell-level flood-map pilot

The user-defined alternative that treats **no flood footprint as dry** now has a [connected-report probability map and backtest](FLOOD_BANGKAPI_PROXY_PROBABILITY.md). It joins reports close in time and space through low terrain, using a 100 m radius when a connected area cannot be inferred. Its percentages refer to that report-based proxy. The independently observed surface-flooding probability discussed below remains unavailable.

The pilot covers **527 cells of 250 m** across Bang Kapi, including off-road land. Each cell has BMA LiDAR elevation, OpenStreetMap road/building/waterway/amenity summaries, nearby BMA pump and gate locations, and optional Sentinel-2 surface summaries. Hourly historical rainfall is shared across the district. The [feature map](artifacts/flood_bangkapi_surface/surface_feature_map.html) shows these inputs.

The current model ranks cells by the chance of **at least one Traffy flood complaint on a day**, conditional on rainfall. It does **not** estimate the chance of surface flooding. A cell with no complaint was not checked and confirmed dry. We therefore leave `flood_probability` empty in every cell. A 250 m cell may include both a road and nearby parcels; its score is not a separate prediction for each land use within it.

## Open data and labels

| Input | Source and use |
|---|---|
| Terrain | [BMA DTM LiDAR ImageServer](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson), sampled at 10 m to summarize each cell's elevation. Ground elevation alone cannot describe an underpass deck, hidden pipe or pump operation. |
| Land use and infrastructure | [OpenStreetMap Thailand extract](https://download.geofabrik.de/asia/thailand.html), dated 1 January 2025: roads, building footprints, waterways and amenities. [BMA pump/gate locations](https://data.bangkok.go.th/dataset/water-station) provide proximity features, not historical operating state. |
| Rain | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), hourly precipitation at one point near Bang Kapi. Features include daily total, hourly and 3-hour peaks, wet hours, and previous 24/72 hours. This misses within-district rainfall differences. |
| Static satellite context | [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a), cloud-masked scene from 15 January 2025: vegetation and built-up proxies. It is not an observation of a 2025 flood. See the [scene timing audit](FLOOD_BANGKAPI_IMAGERY.md). |
| Outcome | [BMA Traffy Fondue 2025](https://data.bangkok.go.th/dataset/traffy-fondue-2568), direct flood-category, dated, geocoded reports. The retained local extract excludes comments, addresses, photo URLs and raw ticket IDs. The 329 Bang Kapi reports occupy 143 cells and 282 cell-report days. Four preventive or pump-work complaints were excluded. |

The displayed BMA road-report anchors are a separate reference overlay. Some anchor coordinates came from a 2026 inventory, so they do not validate a prospective 2025 forecast.

## Chronological backtest

Train on January–August 2025 cell-days and hold out September–December 2025. The held-out set has **150 complaint-positive cell-days** among **64,294 cell-days**. The ranking target is the number of complaint days per cell in the held-out period. `Recent reports` means the count observed before September.

| Ranking input | Spearman versus held-out cell counts | Pairwise order agreement | Predicted positive cell-days |
|---|---:|---:|---:|
| Recent reports | 0.234 | 61.5% | — |
| Terrain, OSM, pumps, rain and report history | **0.326** | **74.6%** | 60.9 |
| Same model plus Sentinel-2 features | 0.287 | 71.6% | 56.4 |

The satellite features slightly raised daily average precision (0.01308 to 0.01333) but **worsened geographic rank**. The map therefore uses the model without satellite. It underpredicted the 150 observed positive cell-days, so its raw complaint scores are not calibrated. Monthly rank checks for the selected model: September 0.226, October 0.210, November 0.208; December had only one positive cell-day and does not support a stable rank estimate. Metrics are in [surface_model_metrics.json](artifacts/flood_bangkapi_surface/surface_model_metrics.json).

## Scenario map

The [interactive scenario map](artifacts/flood_bangkapi_surface/surface_cell_complaint_scenario_map.html) colors **relative complaint-score percentile** among the 527 cells. Its selector compares 4 mm/h for two hours, 8 mm/h for two hours, 4 mm/h for four hours, and 8 mm/h for four hours, with zero rain in the preceding 72 hours and September seasonality. The [scenario table](artifacts/flood_bangkapi_surface/surface_cell_complaint_scenarios.csv) and [GeoJSON](artifacts/flood_bangkapi_surface/surface_cell_complaint_scenarios.geojson) contain every cell and the raw model scores; their `flood_probability` fields remain empty.

The model's rank changes almost not at all when either intensity **or** duration doubles alone (Spearman approximately 1 against the base scenario). Doubling both changes rank more (approximately 0.945 against base). These are responses of a fitted complaint model, **not validated rain-to-flood counterfactuals**. Coarse common rainfall, sparse labels and tree split thresholds limit the scenario response.

## Path to actual cell flood probabilities

Define an event and outcome, for example `water depth > 10 cm somewhere in this cell during this hour`. Collect time-aligned **wet and dry** observations for sampled cells from public flood-depth sensors, archived and geolocated CCTV frames where available, or verified field/photo checks. Sample unreported cells too; otherwise reporting propensity remains confounded with flooding. Add local rain/radar estimates and drainage or pump state, then train and calibrate against whole held-out storms and districts. Compare both rank and probability calibration. Do not deploy a probability legend until those tests pass.

To rebuild the model from the cached open data, run `.venv/bin/python flood_bangkapi_surface_model.py`. Grid construction and the minimal Traffy extract are in `flood_bangkapi_surface_grid.py` and `flood_bangkapi_traffy_extract.py`.

## Open-data wet/dry audit, 21 October 2026

Run `.venv/bin/python flood_bangkapi_probability_evidence.py` for a fresh source check. Its [evidence map](artifacts/flood_bangkapi_surface/surface_probability_evidence_map.html), [GeoJSON](artifacts/flood_bangkapi_surface/surface_probability_evidence.geojson) and [audit](artifacts/flood_bangkapi_surface/surface_probability_evidence_audit.json) distinguish prior reports from independently checked wet/dry cell observations. All 527 `flood_probability` values remain empty.

The [BMA road-flood archive](https://data.bangkok.go.th/dataset/frd_dds) supports 42 named-site report-days in Bang Kapi during 2025, listed in [this dated site table](artifacts/flood_bangkapi_surface/approximate_bma_wet_site_days_2025.csv). Its coordinates are retrospective 2026 inventory anchors: 39 site-days have an anchor within the district boundary, across nine 250 m cells; three anchors fall just outside. These are positive **site** observations with spatial uncertainty, not exact cell labels or dry observations.

[GISTDA FloodCheck's public CCTV response](https://floodcheck.gistda.or.th/cctv) listed 1,308 catalog cameras at the audit time, 371 marked online. A 22 October 2026 coordinate check placed only **21 of those online cameras inside BMA districts**; its four Bang Kapi cameras were all offline. Their `is_flooded=false` flags cannot establish dry conditions, and the response has no observation timestamp or archived frame. The [road-surface depth endpoint](https://floodcheck.gistda.or.th/floodroad) returned HTTP 403 without credentials. GISTDA's [flood-extent catalog](https://data.go.th/dataset/flood-disaster-data) requires API authentication. The 2025 [satellite scene timing audit](FLOOD_BANGKAPI_IMAGERY.md) found one Sentinel-1 pass during a reported Bang Kapi flood interval and no Sentinel-2 pass during one; the road/radar pilot could not verify the known wet road in that pass. BMA's [five-minute water-level catalog](https://data.bangkok.go.th/dataset/flood) describes canal sensors rather than surface wet/dry labels and exposes only a data dictionary in the open catalog.

This audit found **zero verified dry cell-events**. As a result, the probability gate fails. The map shows evidence coverage and keeps probability unavailable; it must not convert unreported cells or offline camera defaults into dry labels. The next usable open-data path is prospective collection of timestamped, reviewed road-camera frames where feeds are online, combined with sampled off-road wet/dry checks and flooded observations across multiple storms and districts. A percentage map then needs a held-out storm/district calibration check.
