# Bangkok road, LiDAR, rain and radar data pilots

Run the reproducible open-data join:

```bash
.venv/bin/python flood_road_data_pilot.py --site soi10
.venv/bin/python flood_road_data_pilot.py --site bangkapi
```

It uses the cached Thailand OpenStreetMap PBF at
`data/thailand-250101.osm.pbf`, downloads source files into
`data/flood_road_pilot/`, and writes outputs into
`artifacts/flood_road_pilot/` for Soi 10 and
`artifacts/flood_road_pilot/bangkapi/` for Bang Kapi. Each pilot is a
1 km × 1 km square. The first is around Chaeng Watthana Soi 10 in Lak Si
(100.578930° E, 13.887480° N). BMA reported
flooding there on **13 September 2025, 17:00–18:20**. Sentinel-1 passed at
18:29 Bangkok time, nine minutes after the reported end.

## What we actually obtained

| Layer | Pilot result | File |
|---|---|---|
| OSM roads + LiDAR | 717 pieces of 10–50 m from 280 OSM ways; all 717 have valid 1 m terrain samples | `roads_lidar_radar.geojson` |
| Satellite comparison | Same ascending Sentinel-1 orbit on 1 and 13 September 2025; 10 m VV backscatter change | `sentinel1_rtc_change_10m.tif` |
| Possible radar darkening | Nine of 10,000 radar pixels pass a deliberately simple screen; polygons are *unverified candidates* | `radar_darkening_candidates.geojson` |
| Government hourly rain | HII stations BKK021 and BKK002, 2021–2025 files where available | `hii_hourly_rain_2021_2025.csv`, `hii_stations.csv` |
| Source and coverage audit | Scene IDs, station gaps, bounds and counts | `audit.json` |

The road GeoJSON stores OSM class and name, bridge/tunnel tags, median and
low-end road terrain elevation, nearby side elevation, and radar change at
each piece's midpoint. These are **road-centreline terrain features**; the
LiDAR DTM is ground elevation and does not give the driving surface of a
bridge or tunnel. `terrain_not_road_surface` flags such OSM ways. The
published vertical datum and road-surface accuracy were not checked.

## Hourly government gauges

The [Hydro-Informatics Institute (HII) hourly rain archive](https://tiservice.hii.or.th/opendata/data_catalog/hourly_rain/)
has downloadable station-by-month CSVs and a
[station metadata file](https://data.hii.or.th/dataset/40afcc66-a44c-4f64-8242-ab8d555b342d/resource/daeae663-3aea-4995-8a45-8f114e189a31/download/0all_stn_metadata.csv).
Although its metadata calls BKK021 and BKK002 water-level stations, their
hourly-rain files contain actual `rainfall_1h` observations.

| HII station | Distance from Soi 10 | Valid hourly readings, 2021–2025 | 13 Sep 2025, readings stamped 15:00–18:00 |
|---|---:|---:|---:|
| BKK021, Wat Bang Bua | 3.82 km | 31,511 | 12.6 mm |
| BKK002, Lak Hok | 9.04 km | 27,685 | 7.8 mm |

Both stations lack all of 2024 and December 2023 in this public archive: 13
missing months each. Their files also contain missing hourly values in the
available months. HII documents `-999`, `9999`, `999999`, and `-` as missing;
the script removes these and negative values. It also flags a wet-season month
with at least 480 readings that are **all zero** as a suspect gauge month, and
removes identical duplicate station-hours. The CSV timestamps have no UTC
offset, so the event comparison treats them as Bangkok local time **without
independent timezone verification**.

The BMA Soi 10 incident row records 63.5 mm. Its gauge location and rainfall
window are not specified in the public row, so 63.5 mm cannot be directly
compared with either HII four-hour sum. The difference reinforces the need for
closer, well-documented rainfall observations before predicting individual
street floods. The BMA open catalog's
[district rainfall dataset](https://data.bangkok.go.th/dataset/rainfall)
is monthly and cannot replace these hourly observations.

## Why the satellite layer is not a wet-road label

The script uses [Sentinel-1 RTC](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc)
on the same orbit and pass direction for the event and reference. The event
product is georeferenced radar backscatter, so it works through clouds and at
night. It is **one snapshot at 18:29**, not a time series during the storm.
The reference date had about 1.0–1.4 mm at the two HII gauges, so it is a
relatively dry comparison, not proven dry at every pixel.

The optional screen flags a 10 m pixel when event VV is below −15 dB and at
least 3 dB lower than the reference. It finds nine pixels across the 1 km²
tile. Only one road-piece midpoint lands on a flagged pixel, and that piece is
on the **elevated Uttaraphimuk expressway**. This is direct evidence that
radar darkening must not be treated as a flooded-road truth label here. Urban
buildings, bridges, vehicle changes, radar geometry, surface roughness and
speckle can change backscatter. Many flooded road strips are narrower than a
10 m pixel, and the overpass may miss short-lived flooding entirely.

Use the TIFF and polygon layer as an investigation queue in QGIS alongside
the road GeoJSON. Do not use the polygon layer as positive training masks for
a U-Net or as proof of wet roads.

## Transfer check: Bang Kapi during an active report

The second pilot covers Bang Kapi intersection at 100.648336° E,
13.766525° N. BMA recorded flooding there on **7 October 2025,
18:15–20:15**; Sentinel-1 passed at **18:29**, during the reported interval.
It is compared with a same-orbit 25 September scene. The pipeline extracted
646 road pieces from 238 OSM ways and found valid LiDAR terrain for all 646.

Only **one of 10,000** radar pixels passes the darkening screen, and no road
midpoint does. Thus the screen also fails to show the reported road flood in
this active-event example. A 10 m radar image cannot reliably establish
whether a particular narrow street was flooded.

| HII station | Distance | Readings stamped 17:00–20:00 on 7 Oct 2025 | Data-quality finding |
|---|---:|---:|---|
| BKK008, Bang Kapi | 3.54 km | Excluded | Every hour of October 2025 says zero; flagged as suspect. |
| BKK021, Wat Bang Bua | 11.74 km | 3.6 mm | Valid readings, distant from the site. |
| BKC008, Bang Nam Phueng | 12.66 km | 9.2 mm | Valid event readings, distant from the site. |

The BMA Bang Kapi incident row lists 99.5 mm, with no published matching
gauge location or accumulation window. It is not directly comparable with
these four-hour station sums. In the HII archive, the BKC008 September 2024
file contains 720 identical duplicate hours and an extra header row; the
script removes them. Its September 2021 all-zero month is also flagged.
These checks are recorded in `bangkapi/audit.json`.

## Using BMA reports as outcomes

The BMA files are useful **positive report labels**. They contain district,
road and sublocation text, start/end times, and usually recorded flood depth.
Across 2021–2025 there are 1,435 report rows; 1,422 have a flood-height value
and 1,433 have a reported duration. Simple, explicit place-name matching finds
five Soi 10 reports (three through 2024 and two in 2025), two reports opposite
the Lak Si telecom building (both in 2025), and three Bang Kapi-intersection
reports (all in 2025).

These reports can label **named sites or road corridors**. They do not yet
label individual 10–50 m OSM pieces reliably. The later BMA risk-point
inventory places the Soi 10 point about **161 m from the named Chaeng Watthana
main-road centreline** in this extract, but about 21 m from a Soi 10 side-road
piece. The report's road field says Chaeng Watthana and its area field says
the mouth of Soi 10. Assigning this report to one nearby piece without manual
review would confuse two different roads. The Bang Kapi inventory point is
about 45 m from the nearest OSM Nawamin piece, still too imprecise to identify
one 10–50 m piece with confidence.

The earlier six-site Lak Si backtest **did** rank against BMA reports: its
2025 site-count ranking had Spearman agreement 0.62, but a simple ranking by
past report counts performed equally well. That small site-level result does
not transfer to the 1,363 road pieces in these two pilot tiles. A road-piece
rank correlation should be reported only after the BMA locations have been
matched and audited at road-piece resolution. An absent report is not a
verified dry road.

## Sources and next data gate

- Roads: [OpenStreetMap Thailand extract](https://download.geofabrik.de/asia/thailand.html), cached PBF dated 1 January 2025. OpenStreetMap contributors, ODbL.
- Terrain: [BMA DTM LiDAR ImageServer](https://cpudgiapp.bangkok.go.th/image/rest/services/DTM_LiDAR/ImageServer?f=pjson), exported at 1 m for this tile.
- Flood timing and rainfall field: [BMA road-flood reports](https://data.bangkok.go.th/dataset/frd_dds), 2025 resource.
- Site coordinate: [BMA flood-risk-point inventory](https://data.bangkok.go.th/dataset/flood_risk26), used only for geolocation; it was published later than the event.
- Rain: [HII government dataset catalog](https://data.go.th/dataset/hii-rainfall) and its hourly station archive above.
- Radar: [Microsoft Planetary Computer Sentinel-1 RTC](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc), with same-orbit scenes on 1 and 13 September 2025.

To train a transferable road-flood model, first geocode and audit the BMA
report location text to these OSM pieces. Obtain time-stamped wet **and dry**
road observations, ideally verified images or field records, and local rain
stations or radar rainfall for storms the model did not see. Then backtest
road rankings with whole storms and districts held out. The current dataset
builds the spatial and temporal join; it does not yet supply trustworthy
street-level flood labels.
