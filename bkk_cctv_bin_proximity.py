"""Compare public BMA bin coordinates with public CCTV point inventories.

Run: .venv/bin/python bkk_cctv_bin_proximity.py
Outputs an interactive map and a machine-readable summary in artifacts/.
Distances are straight-line UTM distances between listed coordinates, not
evidence that a camera sees a bin or that either asset is operational.
"""

from __future__ import annotations

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely import contains_xy, transform as transform_shape, union_all
from shapely.geometry import shape


BIN_CSV = (
    "https://data.bangkok.go.th/dataset/"
    "a0747b8e-8c0b-4965-937d-53c900b73708/resource/"
    "3c0c78f6-1337-4ca7-9366-5d9046be7f47/download/env_gps_bin.csv"
)
LEGACY_CCTV_CSV = (
    "https://data.bangkok.go.th/dataset/"
    "9d4d773f-c626-4578-b42d-e1833c81b35d/resource/"
    "0d5af6a8-5747-4b16-913a-8e0455e37280/download/bma-cctv.csv"
)
CCTV_LAYER = (
    "https://cpudgiportal.bangkok.go.th/arcgis/rest/services/"
    "Hosted/CCTV_Points/FeatureServer/3"
)
DISTRICT_LAYER = (
    "https://cpudgiportal.bangkok.go.th/arcgis/rest/services/"
    "OKR2568/HighValueDataset/FeatureServer/5"
)
DISTRICTS = ["เขตดินแดง", "เขตสัมพันธวงศ์", "เขตปทุมวัน"]
THRESHOLDS_M = [25, 50, 100, 200, 500]
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def plausible_bangkok(lon: float, lat: float) -> bool:
    """A broad range screen; this is not an administrative boundary test."""
    return 100.3 <= lon <= 100.95 and 13.4 <= lat <= 14.1


def get_csv(url: str, encoding: str) -> list[dict[str, str]]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return list(csv.DictReader(io.StringIO(response.content.decode(encoding))))


def get_cctv_points() -> list[dict]:
    response = requests.get(
        f"{CCTV_LAYER}/query",
        params={"where": "1=1", "returnCountOnly": "true", "f": "json"},
        timeout=30,
    )
    response.raise_for_status()
    count = response.json()["count"]

    def page(offset: int) -> list[dict]:
        response = requests.get(
            f"{CCTV_LAYER}/query",
            params={
                "where": "1=1",
                "outFields": "objectid_1,cctv_lat,cctv_long,เขต",
                "returnGeometry": "false",
                "resultOffset": offset,
                "resultRecordCount": 2000,
                "orderByFields": "objectid_1",
                "f": "json",
            },
            timeout=40,
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            raise RuntimeError(result["error"])
        return [item["attributes"] for item in result["features"]]

    with ThreadPoolExecutor(max_workers=5) as executor:
        parts = list(executor.map(page, range(0, count, 2000)))
    rows = [row for part in parts for row in part]
    if len(rows) != count:
        raise RuntimeError(f"CCTV query returned {len(rows)} of {count} rows")
    return rows


def normalize_district(value: str) -> str:
    name = str(value or "").strip().removeprefix("เขต").strip()
    name = {"ป้อมปราบฯ": "ป้อมปราบศัตรูพ่าย", "ตลิ่งชั่น": "ตลิ่งชัน"}.get(name, name)
    return "เขต" + name


def get_district_buffers(transformer: Transformer) -> tuple[dict, object]:
    response = requests.get(
        f"{DISTRICT_LAYER}/query",
        params={"where": "1=1", "outFields": "DISTRICT_N", "f": "geojson"},
        timeout=60,
    )
    response.raise_for_status()
    features = response.json()["features"]
    if len(features) != 50:
        raise RuntimeError(f"Expected 50 district polygons, got {len(features)}")
    polygons = {item["properties"]["DISTRICT_N"]: shape(item["geometry"]) for item in features}
    buffers = {
        name: transform_shape(polygon, transformer.transform, interleaved=False).buffer(100)
        for name, polygon in polygons.items()
    }
    city_buffer = transform_shape(union_all(list(polygons.values())), transformer.transform, interleaved=False).buffer(100)
    return buffers, city_buffer


def clean_bins(
    rows: list[dict[str, str]], transformer: Transformer, district_buffers: dict, city_buffer: object
) -> tuple[dict, int, int, int]:
    sites: dict[tuple[float, float], dict] = {}
    invalid = 0
    outside_city = 0
    district_mismatch = 0
    for row in rows:
        try:
            lon, lat = float(row["Lon_gi_tude"]), float(row["La_ti_tude"])
        except (ValueError, TypeError):
            invalid += 1
            continue
        if not plausible_bangkok(lon, lat):
            invalid += 1
            continue
        x, y = transformer.transform(lon, lat)
        if not contains_xy(city_buffer, x, y):
            outside_city += 1
            continue
        key = (lon, lat)
        site = sites.setdefault(
            key,
            {"districts": set(), "types": set(), "location": row["Bin_locate"], "records": 0},
        )
        district = normalize_district(row["Dis_trict"])
        if district in district_buffers and contains_xy(district_buffers[district], x, y):
            site["districts"].add(district)
        else:
            district_mismatch += 1
        site["types"].add(row["Bin_type"])
        site["records"] += 1
    return sites, invalid, outside_city, district_mismatch


def clean_cameras(
    rows: list[dict], transformer: Transformer, district_buffers: dict
) -> tuple[dict[tuple[float, float], dict], int, int]:
    sites: dict[tuple[float, float], dict] = {}
    invalid = 0
    district_mismatch = 0
    for row in rows:
        try:
            lon, lat = float(row["cctv_long"]), float(row["cctv_lat"])
        except (ValueError, TypeError):
            invalid += 1
            continue
        if not plausible_bangkok(lon, lat):
            invalid += 1
            continue
        district = normalize_district(row["เขต"])
        x, y = transformer.transform(lon, lat)
        if district not in district_buffers or not contains_xy(district_buffers[district], x, y):
            district_mismatch += 1
            continue
        site = sites.setdefault((lon, lat), {"id": row["objectid_1"], "records": 0})
        site["records"] += 1
    return sites, invalid, district_mismatch


def clean_legacy_cameras(rows: list[dict[str, str]]) -> tuple[set[tuple[float, float]], int]:
    sites: set[tuple[float, float]] = set()
    invalid = 0
    for row in rows:
        try:
            lon, lat = float(row["long"]), float(row["lat"])
        except (ValueError, TypeError):
            invalid += 1
            continue
        if plausible_bangkok(lon, lat):
            sites.add((lon, lat))
        else:
            invalid += 1
    return sites, invalid


def main() -> None:
    transformer = Transformer.from_crs(4326, 32647, always_xy=True)
    district_buffers, city_buffer = get_district_buffers(transformer)
    bins_raw = get_csv(BIN_CSV, "cp874")
    old_raw = get_csv(LEGACY_CCTV_CSV, "utf-8-sig")
    cameras_raw = get_cctv_points()
    bins, bad_bins, outside_city_bins, mismatched_bin_rows = clean_bins(
        bins_raw, transformer, district_buffers, city_buffer
    )
    cameras, bad_cameras, mismatched_camera_rows = clean_cameras(
        cameras_raw, transformer, district_buffers
    )
    old_cameras, bad_old_cameras = clean_legacy_cameras(old_raw)

    camera_keys = list(cameras)
    camera_xy = np.array([transformer.transform(*point) for point in camera_keys])
    tree = cKDTree(camera_xy)
    old_keys = list(old_cameras)
    old_tree = cKDTree(np.array([transformer.transform(*point) for point in old_keys]))

    def compare(points: list[tuple[float, float]], index: cKDTree) -> tuple[np.ndarray, np.ndarray]:
        xy = np.array([transformer.transform(*point) for point in points])
        return index.query(xy)

    def stats(distances: np.ndarray) -> dict:
        return {
            "bin_sites": len(distances),
            "median_nearest_m": round(float(np.median(distances)), 1),
            "within_m": {
                str(radius): int(np.count_nonzero(distances <= radius))
                for radius in THRESHOLDS_M
            },
        }

    city_points = list(bins)
    city_distances, _ = compare(city_points, tree)
    city_old_distances, _ = compare(city_points, old_tree)
    payload = {
        "checked_at_bangkok": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds"),
        "method": "Unique exact coordinates; broad range and district-boundary check with 100 m tolerance; nearest straight-line distance in EPSG:32647; no line-of-sight test.",
        "sources": {"bins_csv": BIN_CSV, "cctv_gis": CCTV_LAYER, "legacy_cctv_csv": LEGACY_CCTV_CSV, "district_boundaries": DISTRICT_LAYER},
        "quality": {
            "bin_records": len(bins_raw), "bin_invalid_range_or_parse": bad_bins,
            "bin_rows_outside_city": outside_city_bins,
            "bin_rows_with_unmatched_district": mismatched_bin_rows,
            "bin_unique_sites": len(bins), "cctv_gis_records": len(cameras_raw),
            "cctv_gis_invalid_range_or_parse": bad_cameras,
            "cctv_gis_rows_excluded_district_mismatch": mismatched_camera_rows,
            "cctv_gis_rows_retained": sum(site["records"] for site in cameras.values()),
            "cctv_gis_unique_sites": len(cameras),
            "cctv_gis_max_records_one_coordinate": max(site["records"] for site in cameras.values()),
            "legacy_cctv_records": len(old_raw), "legacy_cctv_invalid_range_or_parse": bad_old_cameras,
            "legacy_cctv_unique_sites": len(old_cameras),
        },
        "city": {"gis_cctv": stats(city_distances), "legacy_cctv_csv": stats(city_old_distances)},
        "districts": {},
    }

    map_data = {}
    for district in DISTRICTS:
        points = [point for point, site in bins.items() if district in site["districts"]]
        distances, indices = compare(points, tree)
        old_distances, _ = compare(points, old_tree)
        points_xy = np.array([transformer.transform(*point) for point in points])
        nearby_camera_indices = set()
        for nearby in tree.query_ball_point(points_xy, 300):
            nearby_camera_indices.update(nearby)
        camera_indices = nearby_camera_indices | set(indices.tolist())
        old_indices = set()
        for nearby in old_tree.query_ball_point(points_xy, 300):
            old_indices.update(nearby)
        payload["districts"][district] = {
            "gis_cctv": stats(distances),
            "legacy_cctv_csv": stats(old_distances),
            "gis_cctv_sites_within_300m_of_any_bin": len(nearby_camera_indices),
        }
        map_data[district] = {
            "stats": payload["districts"][district],
            "bins": [
                {
                    "lat": point[1], "lon": point[0], "distance_m": round(float(distances[i]), 1),
                    "camera_lat": camera_keys[int(indices[i])][1],
                    "camera_lon": camera_keys[int(indices[i])][0],
                    "location": bins[point]["location"],
                    "types": sorted(bins[point]["types"]),
                    "source_records": bins[point]["records"],
                }
                for i, point in enumerate(points)
            ],
            "cameras": [
                {"lat": camera_keys[i][1], "lon": camera_keys[i][0], **cameras[camera_keys[i]]}
                for i in sorted(camera_indices)
            ],
            "legacy_cameras": [
                {"lat": old_keys[i][1], "lon": old_keys[i][0]}
                for i in sorted(old_indices)
            ],
        }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "bkk_cctv_bin_proximity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html = HTML.replace("__MAP_DATA__", json.dumps(map_data, ensure_ascii=False).replace("<", "\\u003c"))
    (ARTIFACTS / "bkk_cctv_bin_proximity.html").write_text(html, encoding="utf-8")
    print(json.dumps({"quality": payload["quality"], "city": payload["city"], "districts": payload["districts"]}, ensure_ascii=False, indent=2))


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bangkok bins and CCTV proximity</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
*{box-sizing:border-box}body{margin:0;font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;color:#172538;background:#f3f6f9}
header{padding:20px 24px;background:#12304a;color:white}h1{font-size:23px;margin:0 0 4px}header p{margin:0;color:#cee0eb;max-width:900px}
.toolbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:14px 24px;background:white;border-bottom:1px solid #dbe3e9}
select{font:inherit;padding:7px 10px;border:1px solid #aab9c7;border-radius:7px}label{display:flex;gap:6px;align-items:center}
.stats{display:flex;gap:10px;flex-wrap:wrap;padding:14px 24px}.card{background:white;border:1px solid #dbe3e9;border-radius:9px;padding:10px 14px;min-width:135px}.card b{display:block;font-size:19px}.card span{font-size:12px;color:#4c6071}
#map{height:67vh;min-height:480px;margin:0 24px 18px;border:1px solid #c8d5df;border-radius:8px}
.note{padding:0 24px 20px;color:#4c6071;font-size:13px}.legend{background:white;padding:8px 10px;border-radius:7px;box-shadow:0 1px 8px #3453;line-height:1.75}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
@media(max-width:600px){#map{margin:0 10px 12px;height:65vh}.toolbar,.stats,header,.note{padding-left:12px;padding-right:12px}}
</style></head><body>
<header><h1>Bangkok bin sites near listed CCTV points</h1><p>Choose a district to inspect cleaned bin locations, CCTV sites, and the nearest coordinate to each bin. Distances are straight-line metres.</p></header>
<div class="toolbar"><label>District <select id="district"></select></label><label><input id="showCameras" type="checkbox" checked> Large CCTV inventory</label><label><input id="showLegacy" type="checkbox"> Older 238-row CSV</label></div>
<div class="stats" id="stats"></div><div id="map"></div>
<div class="note">A nearby CCTV coordinate does not establish that the camera works, faces the bin, or provides a usable video feed. Bin and camera coordinates contain repeats; each marker here is one exact coordinate. CCTV records whose listed district is over 100 m from their coordinate were excluded. The older CSV covers a much smaller camera inventory. Map tiles require internet.</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const data=__MAP_DATA__;
const districtSelect=document.getElementById('district');
const names={'เขตดินแดง':'Din Daeng','เขตสัมพันธวงศ์':'Samphanthawong','เขตปทุมวัน':'Pathum Wan'};
Object.keys(data).forEach(name=>{const o=document.createElement('option');o.value=name;o.textContent=names[name]+' ('+name+')';districtSelect.appendChild(o)});
const map=L.map('map',{preferCanvas:true});L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'© OpenStreetMap contributors'}).addTo(map);
const binLayer=L.layerGroup().addTo(map),cameraLayer=L.layerGroup().addTo(map),legacyLayer=L.layerGroup(),selectionLayer=L.layerGroup().addTo(map);
const escapeHtml=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function color(d){return d<=50?'#0f9f6a':d<=100?'#e4aa16':d<=200?'#e2762f':'#d24a4a'}
function card(value,label){return '<div class="card"><b>'+value+'</b><span>'+label+'</span></div>'}
function render(){
 const item=data[districtSelect.value];binLayer.clearLayers();cameraLayer.clearLayers();legacyLayer.clearLayers();selectionLayer.clearLayers();
 item.cameras.forEach(c=>L.circleMarker([c.lat,c.lon],{radius:2.6,color:'#2d64aa',fillColor:'#2d64aa',fillOpacity:.55,weight:0}).bindPopup('CCTV site<br>Inventory records at this coordinate: '+c.records).addTo(cameraLayer));
 item.legacy_cameras.forEach(c=>L.circleMarker([c.lat,c.lon],{radius:5,color:'#8050aa',fillOpacity:0,weight:2}).bindPopup('Older CCTV CSV coordinate').addTo(legacyLayer));
 const bounds=[];
 item.bins.forEach(b=>{
  const marker=L.circleMarker([b.lat,b.lon],{radius:5,color:'#fff',weight:1.2,fillColor:color(b.distance_m),fillOpacity:.94}).addTo(binLayer);
  marker.bindPopup('<b>Bin site</b><br>'+escapeHtml(b.location)+'<br>Nearest listed CCTV: '+b.distance_m.toFixed(1)+' m<br>Bin types: '+escapeHtml(b.types.join(', '))+'<br>Source rows at this coordinate: '+b.source_records);
  marker.on('click',()=>{selectionLayer.clearLayers();L.polyline([[b.lat,b.lon],[b.camera_lat,b.camera_lon]],{color:'#202b3a',weight:2,dashArray:'5,5'}).addTo(selectionLayer);L.circleMarker([b.camera_lat,b.camera_lon],{radius:8,color:'#202b3a',fillOpacity:0,weight:2}).addTo(selectionLayer)});
  bounds.push([b.lat,b.lon]);
 });
 if(bounds.length)map.fitBounds(bounds,{padding:[25,25]});
 const s=item.stats.gis_cctv,old=item.stats.legacy_cctv_csv;
 document.getElementById('stats').innerHTML=card(s.bin_sites,'unique bin coordinates')+card(s.within_m['50']+' ('+(100*s.within_m['50']/s.bin_sites).toFixed(1)+'%)','within 50 m of large inventory')+card(s.within_m['100']+' ('+(100*s.within_m['100']/s.bin_sites).toFixed(1)+'%)','within 100 m')+card(s.within_m['200']+' ('+(100*s.within_m['200']/s.bin_sites).toFixed(1)+'%)','within 200 m')+card(old.within_m['100'],'within 100 m of older CSV');
}
const legend=L.control({position:'bottomright'});legend.onAdd=()=>{const div=L.DomUtil.create('div','legend');div.innerHTML='<b>Bin to nearest CCTV</b><br><span class="swatch" style="background:#0f9f6a"></span>0–50 m<br><span class="swatch" style="background:#e4aa16"></span>50–100 m<br><span class="swatch" style="background:#e2762f"></span>100–200 m<br><span class="swatch" style="background:#d24a4a"></span>Over 200 m<br><span class="swatch" style="background:#2d64aa"></span>CCTV site';return div};legend.addTo(map);
districtSelect.addEventListener('change',render);
document.getElementById('showCameras').addEventListener('change',e=>e.target.checked?cameraLayer.addTo(map):map.removeLayer(cameraLayer));
document.getElementById('showLegacy').addEventListener('change',e=>e.target.checked?legacyLayer.addTo(map):map.removeLayer(legacyLayer));
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
