"""Count AI for Good recurrence classes within the BMA Bang Kapi polygon.

Run: .venv/bin/python flood_bangkapi_global_radar_audit.py
Requires the cached N12E099 recurrence GeoTIFF. The source README defines
0 as nodata, 1 as an exclusion mask, and 2+ as detected flood months + 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

from flood_bangkok_transfer import load_districts


ROOT = Path(__file__).resolve().parent
TILE = ROOT / "data/flood_bangkapi_surface/ai4g_N12E099_recurrence_2014_2024.tif"
OUTPUT = ROOT / "artifacts/flood_bangkapi_surface/ai4g_recurrence_bangkapi_audit.json"
SOURCE = "https://huggingface.co/datasets/ai-for-good-lab/ai4g-flood-dataset"


def main() -> None:
    district = next(d for d in load_districts() if d["code"] == "1006")
    projector = Transformer.from_crs(32647, 4326, always_xy=True)
    polygon = shapely.transform(district["polygon"], projector.transform,
                                interleaved=False)
    with rasterio.open(TILE) as source:
        if source.crs.to_epsg() != 4326:
            raise ValueError("Unexpected recurrence-tile CRS")
        window = from_bounds(*polygon.bounds, transform=source.transform).round_offsets().round_lengths()
        pixels = source.read(1, window=window)
        inside = geometry_mask([polygon.__geo_interface__], pixels.shape,
                               source.window_transform(window), invert=True)
        values, counts = np.unique(pixels[inside], return_counts=True)
    classes = {str(int(value)): int(count) for value, count in zip(values, counts)}
    total = int(inside.sum())
    result = {"source": SOURCE, "source_readme": f"{SOURCE}/raw/main/README.md",
              "tile": str(TILE.relative_to(ROOT)), "district": "Bang Kapi",
              "pixels_inside_district": total, "value_counts": classes,
              "excluded_value_1_fraction": round(classes.get("1", 0) / total, 4),
              "flood_detection_pixels_value_2_or_more": sum(
                  count for value, count in classes.items() if int(value) >= 2),
              "interpretation": "No usable positive flood pixels; zero is nodata and one is excluded, not verified dry."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
