"""Candidate-cell adjacency from BMA DTM, with no inferred hard barriers."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from flood_citywide_route_pretrain import terrain
from flood_citywide_history_ranker import SHAPE

HEIGHT_SCALE_M = .25
DISPLAY_MAX_GAP_M = .25


def links(data):
    z,_,_,dem_audit=terrain(data)
    flat=data["candidate_flat"]
    lookup=np.full(int(np.prod(SHAPE)),-1,np.int32)
    lookup[flat]=np.arange(len(flat),dtype=np.int32)
    rr,cc=np.unravel_index(flat,SHAPE)
    origin=[]
    target=[]
    gap=[]
    for dr,dc in ((1,0),(0,1)):
        good=(rr+dr<SHAPE[0])&(cc+dc<SHAPE[1])
        src=np.flatnonzero(good)
        dst=lookup[flat[src]+dr*SHAPE[1]+dc]
        inside=dst>=0
        src,dst=src[inside],dst[inside]
        distance=np.abs(z.ravel()[flat[src]]-z.ravel()[flat[dst]])
        finite=np.isfinite(distance)
        origin.extend(src[finite].tolist())
        target.extend(dst[finite].tolist())
        gap.extend(distance[finite].tolist())
    a=np.asarray(origin,np.int32)
    b=np.asarray(target,np.int32)
    dz=np.asarray(gap,np.float32)
    if not len(a) or len(a)!=len(b):
        raise ValueError("No valid terrain adjacency")
    row=np.r_[a,b]
    col=np.r_[b,a]
    ordinary=csr_matrix((np.ones(len(row),np.float32),(row,col)),shape=(len(flat),len(flat)))
    physical=np.exp(-np.square(dz/HEIGHT_SCALE_M)).astype(np.float32)
    terrain_weight=csr_matrix((np.r_[physical,physical],(row,col)),shape=ordinary.shape)
    audit={"cell_edges":int(len(a)),
           "similar_height_edges_at_025m":int((dz<=DISPLAY_MAX_GAP_M).sum()),
           "height_gap_quantiles_m":np.quantile(dz,[.1,.25,.5,.75,.9,.95]).round(4).tolist(),
           "height_scale_m":HEIGHT_SCALE_M,
           "display_max_gap_m":DISPLAY_MAX_GAP_M,
           "source":dem_audit["dem"]["source"] if "dem" in dem_audit else "cached BMA DTM",
           "barrier_state":"Unknown: no complete walls, kerbs, culverts or time-matched gate/pump states"}
    return ordinary,terrain_weight,a,b,dz,audit
