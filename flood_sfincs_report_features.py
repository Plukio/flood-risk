"""Pool a frozen 10 m SFINCS CNN onto the Bangkok 250 m report grid.

The scenario is a closed-tile, initially dry 40 mm/h, 15-minute storm. It is
an assumed susceptibility probe, not a reconstruction of a report-day storm.
Only LiDAR exports with native resolution <=15 m (plus the three original
10 m study rasters) are used. Coarser exports are never presented as 10 m data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from flood_bangkok_transfer import CACHE as BMA_CACHE
from flood_citywide_history_ranker import SHAPE, load_grid
from flood_multi_target_citywide_data import X0, Y0, ROWS, TILE_M, OUT
from flood_sfincs_direction_pretrain import FlowMaskCNN
from flood_sfincs_lidar_trial import DTM as BANG_KAPI_DTM
from flood_sfincs_physics_model import water_projection
from flood_sfincs_surface_multiarea import REGIONS

CHECKPOINT = Path("artifacts/flood_bangkapi_surface/sfincs_direction_pretrain/paired_extension/paired_direction_mask/checkpoint.pt")
OUT_NPZ = OUT / "sfincs_10m_direction_250m_frozen_features.npz"
OUT_JSON = OUT / "sfincs_10m_direction_250m_frozen_features.json"
SPECIAL = {"1006": BANG_KAPI_DTM,
           "1040": REGIONS["bang_khae"]["dtm"],
           "1011": REGIONS["lat_krabang"]["dtm"]}
YTOP = Y0 + ROWS*TILE_M
RATE = 40.
STEPS = 3
BLOCK = 48
HALO = 24
BATCH = 16


def cropped_10m(path: Path):
    """Read full 250 m cells strictly inside the source's footprint."""
    with rasterio.open(path) as src:
        if src.crs.to_epsg() != 32647:
            raise ValueError(f"Unexpected CRS: {path}")
        left = max(X0, math.ceil(src.bounds.left/250)*250)
        right = min(X0+SHAPE[1]*250, math.floor(src.bounds.right/250)*250)
        top = min(YTOP, math.floor(src.bounds.top/250)*250)
        bottom = max(YTOP-SHAPE[0]*250, math.ceil(src.bounds.bottom/250)*250)
        if right <= left or top <= bottom:
            return None
        height, width = (top-bottom)//10, (right-left)//10
        z = np.full((height,width), np.nan, np.float32)
        reproject(src.read(1), z, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=from_origin(left,top,10,10), dst_crs=src.crs,
                  src_nodata=src.nodata, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        if not np.isfinite(z).all():
            return None
        return z, (YTOP-top)//250, (left-X0)//250, float(src.res[0])


def forward_features(model: FlowMaskCNN, image: torch.Tensor):
    hidden = F.silu(model.entry(image))
    for layer in model.layers:
        hidden = F.silu(hidden+.2*layer(hidden))
    return (water_projection(model.depth_head(hidden),image),
            model.velocity_head(hidden), hidden)


def scenario_mosaic(model: FlowMaskCNN, z: np.ndarray, device: str,
                    frozen_initial_model:FlowMaskCNN|None=None):
    """Run trained-size 96x96 windows; use their central 48x48 outputs."""
    h,w = z.shape
    padded = np.pad(z, ((HALO,BLOCK+HALO),(HALO,BLOCK+HALO)),mode="edge")
    positions = [(r,c) for r in range(0,h,BLOCK) for c in range(0,w,BLOCK)]
    output = np.empty((51,h,w),np.float32)  # depth mm, signed u/v, 48 trunk maps
    for start in range(0,len(positions),BATCH):
        coords=positions[start:start+BATCH]
        patches=np.stack([padded[r:r+96,c:c+96] for r,c in coords])
        center=np.clip((patches-np.median(patches,axis=(1,2))[:,None,None])/.5,-6,6)
        relief=np.clip((patches-np.stack([gaussian_filter(p,3) for p in patches]))/.1,-6,6)
        x=torch.zeros((len(coords),7,96,96),dtype=torch.float32,device=device)
        x[:,0]=torch.from_numpy(center).to(device)
        x[:,1]=torch.from_numpy(relief).to(device)
        x[:,4]=RATE/60
        previous=torch.zeros_like(x[:,2:3])
        current=torch.zeros_like(previous)
        with torch.no_grad():
            for step in range(STEPS):
                x[:,2:3]=previous/.05
                x[:,3:4]=current/.05
                active=(frozen_initial_model if step<STEPS-1 and
                        frozen_initial_model is not None else model)
                depth,velocity,hidden=forward_features(active,x)
                previous,current=current,depth
            selected=torch.cat((depth*1000,velocity,hidden),1)
            selected=selected[:,:,HALO:HALO+BLOCK,HALO:HALO+BLOCK].cpu().numpy()
        for j,(r,c) in enumerate(coords):
            rh,cw=min(BLOCK,h-r),min(BLOCK,w-c)
            output[:,r:r+rh,c:c+cw]=selected[j,:,:rh,:cw]
    return output


def pool_250m(mosaic: np.ndarray):
    channels,h,w=mosaic.shape
    if h%25 or w%25:
        raise ValueError("10 m raster is not aligned to full 250 m cells")
    blocks=mosaic.reshape(channels,h//25,25,w//25,25)
    mean=blocks.mean(axis=(2,4))
    depth_p90=np.quantile(blocks[0],.9,axis=(1,3)).astype(np.float32)
    speed=np.hypot(blocks[1],blocks[2]).mean(axis=(1,3))
    return np.concatenate((mean[:1],depth_p90[None],mean[1:3],
                           speed[None],mean[3:]),axis=0).astype(np.float32)


def build_features(data,model:FlowMaskCNN,device:str,
                   frozen_initial_model:FlowMaskCNN|None=None):
    physics=np.zeros((53,*SHAPE),np.float32)
    coverage=np.zeros(SHAPE,bool)
    rows=[]
    for district_id,district in enumerate(data["districts"]):
        code=district["code"]
        path=SPECIAL.get(code,BMA_CACHE/f"terrain_{code}.tif")
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            resolution=float(src.res[0])
        if resolution>15.01:
            continue
        prepared=cropped_10m(path)
        if prepared is None:
            continue
        z,row,col,native_m=prepared
        pooled=pool_250m(scenario_mosaic(model,z,device,
                                        frozen_initial_model))
        rh,cw=pooled.shape[1:]
        district_mask=(data["district_id"][row:row+rh,col:col+cw]==district_id)&\
                      data["candidate"][row:row+rh,col:col+cw]
        view=physics[:,row:row+rh,col:col+cw]
        view[:,district_mask]=pooled[:,district_mask]
        coverage[row:row+rh,col:col+cw]|=district_mask
        rows.append({"code":code,"name":district["name_en"],
                     "native_resolution_m":round(native_m,3),
                     "candidate_cells":int(district_mask.sum()),
                     "source":str(path)})
        print(code,district["name_en"],"cells",int(district_mask.sum()),flush=True)
    return physics,coverage,rows


def main():
    torch.set_num_threads(4)
    data=load_grid()
    device="mps" if torch.backends.mps.is_available() else "cpu"
    model=FlowMaskCNN().to(device).eval()
    model.load_state_dict(torch.load(CHECKPOINT,map_location="cpu",
                                     weights_only=True)["state"])
    physics,coverage,rows=build_features(data,model,device)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT_NPZ,features=physics,coverage=coverage,
                        candidate_flat=data["candidate_flat"])
    audit={"checkpoint":str(CHECKPOINT),"scenario":{"start":"dry",
           "rain_mm_per_hour":RATE,"minutes":STEPS*5,"roughness_n":.035},
           "teacher":"frozen 10 m SFINCS depth+direction+mask CNN",
           "tile":"96x96 trained-size crop, central 48x48 retained",
           "pool":"full 25x25 pixels per 250 m cell: mean depth, p90 depth, mean signed east/north velocity, mean speed, 48 mean hidden channels",
           "covered_candidate_cells":int(coverage.sum()),
           "total_candidate_cells":int(data["candidate"].sum()),
           "districts":rows,
           "limitations":["Synthetic closed-tile rain response, not observed flood or report-day water",
                          "Cached district LiDAR at 10-15 m is resampled to 10 m; this creates no new detail",
                          "Uniform assumed roughness omits roads, buildings, drains and pumps",
                          "Cells outside <=15 m source footprints have no physics features"]}
    OUT_JSON.write_text(json.dumps(audit,indent=2)+"\n")
    print("saved",OUT_NPZ,"covered",int(coverage.sum()),flush=True)


if __name__=="__main__":
    main()
