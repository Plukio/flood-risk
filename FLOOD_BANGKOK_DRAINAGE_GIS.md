# Bangkok drainage paths: newly found public GIS layer

**Audited 23 October 2026.** This corrects the earlier conclusion based on the
[BMA drain-pipe CSV](https://data.bangkok.go.th/dataset/drain-pipe). That CSV
has road/from/to names and dimensions but no coordinates. A separate, publicly
queryable [BMA ArcGIS drainage service](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer?f=pjson)
**does provide pipe and canal geometries**. The pipe layer describes a 2021
(B.E. 2564), 1:4,000 compilation from scans, CAD and shapefiles in WGS84 UTM
zone 47N. It has no time field or evidence of later updates.

## What is actually mapped

| BMA ArcGIS layer | Queryable features | What it supplies | Connection limit |
| --- | ---: | --- | --- |
| [5: storm-drain pipes](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer/5?f=pjson) | **81,574** polylines | Geometry, main/secondary type, dimension, length, district, road, `F_ROAD`/`TO_ROAD`, endpoint bottom-elevation text, source and year | No inlet point or explicit pump/outfall edge; digitized endpoints need topology checks |
| [6: canals](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer/6?f=pjson) | **3,496** polylines | Named canal geometry; some width/depth fields | No observed flow direction, level or connection state |
| [7: pump equipment](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer/7?f=pjson) | **273** points | Named water-control buildings and `POWER` | No dated on/off or discharge observations; may overlap other pump catalogs |
| [0: pump stations](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer/0?f=pjson) | **91** points | Station name and stream | Different inventory and extent from layer 7; do not add counts as unique facilities |
| [1: BMA gates](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer/1?f=pjson) | **77** points | Gate name, location and size text | No event-time opening or inside/outside head |
| [2: irrigation gates](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer/2?f=pjson) | **377** points | A second gate inventory, covering a wider extent | Separate ownership and likely overlap need checking |
| [3: drainage tunnels](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer/3?f=pjson) | **7** polylines | Mapped tunnel routes | Construction/operation vintage is not specified by this layer |
| [9: pipe-jacking works](https://cpudgiapp.bangkok.go.th/arcgis/rest/services/Thematic/Drainage/MapServer/9?f=pjson) | **12** polylines | Additional large pipe routes, dimension and discharge fields | The layer description dates these to 2020; status is unclear |

Every pipe row has `YEAR=2564` in the service. An ArcGIS count query finds
**40,977** of the 81,574 pipes with both endpoint bottom-elevation fields
different from the missing-value marker `-`. The fields are text and still need
numeric and datum validation. They offer a **candidate** gravity-flow
direction for some segments; they do not by themselves tell us pump or gate
behavior. A search across the 67 publicly listed BMA ArcGIS Map/Feature
services found no separately named street inlet, manhole-point or outfall
layer. Absence from that service list is not proof BMA lacks an internal one.

## Bang Kapi connection audit and map

I queried the pipe records with district code `1006` (Bang Kapi) and nearby
canals, pumps, gates and tunnels. The [reproducible audit script](flood_bangkok_drainage_gis_audit.py)
saved the [source geometry](artifacts/flood_open_weak_train/multi_target/citywide/bangkapi_drainage_gis_2021_source.geojson),
[candidate pipe–canal points](artifacts/flood_open_weak_train/multi_target/citywide/bangkapi_candidate_pipe_canal_connections.geojson),
[counts and rules](artifacts/flood_open_weak_train/multi_target/citywide/bangkapi_drainage_gis_2021_audit.json),
and an [interactive inspection map](artifacts/flood_open_weak_train/multi_target/citywide/bangkapi_drainage_gis_2021_map.html).

| Bang Kapi check | Result |
| --- | ---: |
| Pipe polylines | **2,454**: 935 main, 1,519 secondary |
| Pipes with two parseable bottom elevations | **955** (39%) |
| Pipe endpoints with a *different* pipe endpoint within 5 m | **4,240 / 4,908** |
| Pipes whose `TO_ROAD` names a canal or river | **325** |
| Those whose last geometry endpoint is within 20 m of a BMA canal | **305** |
| Those whose nearest canal name also matches `TO_ROAD` after normalization | **254** |
| Groups of the 254 candidate points after 20 m clustering | **129** |
| Pump equipment, BMA gates, tunnel lines in pipe bounding box +500 m | **32, 4, 1** |

For the 955 pipes with both numeric invert fields, the median recorded
start-minus-end bottom elevation is **0.052 m**; the 5th–95th percentile is
**−0.211 to +0.594 m**. The geometry order and datum have not been independently
checked, so a positive difference is a direction *candidate*, not a verified
flow direction. Most Bang Kapi pipes cite `Map Scan , CAD File` as their
source. Endpoint proximity is also uncertain: crossings, small drawing gaps,
parallel pipes, and connections beyond the district boundary can change the
topology.

The **129 green map markers are candidate pipe–canal connections, not confirmed
outfalls**. A marker is created only when the pipe's `TO_ROAD` names a canal,
its last endpoint is within 20 m of a BMA canal line, and the nearest canal
has the same normalized name. Nearby candidates are clustered at 20 m. Some
pipe ends also meet another pipe; a canal-side junction could feed a further
pipe rather than discharge directly. Field plans, outlet structures, and gate
or pump records would be needed to verify each outfall.

## What this changes for the flood model

We can build a **network candidate** with pipe segments, dimensions, partial
invert elevations, named pipe-to-canal connections, tunnels, and geocoded
pumps/gates. This gives stronger mapped geometry than LiDAR D8 slopes or OSM
canal lines alone. To move from the first static-feature test below toward a
hydraulic network model, the remaining checks are:

1. Validate the first test's 5 m endpoint snaps against plans, preserving
   ambiguous parallel/crossing pipes and source-quality flags. Do not connect
   every nearby crossing automatically.
2. Verify its `TO_ROAD` name-and-distance candidate canal endpoints as real
   outlets. Keep `named + spatial`, `spatial only`, and `unmatched` confidence
   levels separate.
3. Add gravity direction only where both invert values are valid and the drop
   exceeds measurement tolerance. Represent pumps, gates and tunnels as
   separate edge types with **unknown historical state**.
4. Verify 250 m cell-to-pipe attachments as real inlet links; no street-inlet
   point layer was found. Then test dynamic upstream response and report
   history against the [BMA-road-pretrained CNN](FLOOD_CITYWIDE_BMA_ROAD_BAGS.md)
   and matched shuffled-edge/distance controls on a fresh season.

A [first citywide model test](FLOOD_CITYWIDE_BMA_DRAINAGE_NETWORK_TEST.md) has now
cached the 2021 pipe and canal geometries, snapped endpoints within 5 m, and
given a CNN a static path-to-candidate-canal feature alongside pipe, pump and
gate proximity. On the same held-out Q4 report-site queue, the real-feature
adapter averaged **66.93/332** isolated positive cells across three seeds,
below the frozen BMA-road CNN's **68.85/332**. Candidate link placement beat a
district-stratified shuffled-link control on average but did not improve the
frozen model consistently. This is a graph-derived CNN feature, not a verified
hydraulic GNN.

The geometry does not yet validate flood extent or a doubled-rain simulation.
The important missing observations are street inlets, verified hydraulic
junctions/outfall status, event-time pump/gate operation, canal and river
levels, and independently checked wet **and** dry locations. The GIS was
compiled for 2021, so newer roads and drains also need a vintage check.
