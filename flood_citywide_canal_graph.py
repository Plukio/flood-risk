"""Rasterize OSM canal continuity across the Bangkok 250 m evaluation grid.

This is an undirected topology proxy. It does not encode gate state, pump
operation, canal level, actual drainage catchments, or hydraulic direction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from flood_bangkapi_surface_grid import OSM_PBF
from flood_multi_target_citywide_data import COLS, OUT, ROWS, TILE_M, X0, Y0, specs_and_masks
from flood_open_weak_data import LocalWays


PIXEL_M = 250
HEIGHT, WIDTH = ROWS*80, COLS*80
N_PIXELS = HEIGHT*WIDTH
GRAPH_FILE = OUT / "osm_canal_graph_250m.npz"
AUDIT_FILE = OUT / "osm_canal_graph_250m_audit.json"


def flat_cell(x: float, y: float) -> int:
    col = int(np.floor((x-X0)/PIXEL_M))
    row = int(np.floor((Y0+ROWS*TILE_M-y)/PIXEL_M))
    return row*WIDTH+col if 0 <= row < HEIGHT and 0 <= col < WIDTH else -1


def sampled_cells(coords: np.ndarray):
    previous = -1
    for a,b in zip(coords[:-1],coords[1:]):
        distance = float(np.hypot(*(b-a)))
        count = max(1,int(np.ceil(distance/75)))
        for step in range(count+1):
            if step == 0 and previous >= 0:
                continue
            xy = a+(b-a)*(step/count)
            cell = flat_cell(float(xy[0]),float(xy[1]))
            if cell != previous:
                yield previous,cell
                previous=cell


def build(candidate: np.ndarray) -> dict:
    if candidate.shape != (HEIGHT,WIDTH):
        raise ValueError("Candidate grid has changed")
    OUT.mkdir(parents=True,exist_ok=True)
    specs,_=specs_and_masks()
    handler=LocalWays(specs)
    handler.apply_file(str(OSM_PBF),locations=True)
    edges=set()
    cells=set()
    for line in handler.canal:
        coords=np.asarray(line.coords,dtype=np.float64)
        for old,new in sampled_cells(coords):
            if new >= 0:
                cells.add(new)
            if old >= 0 and new >= 0 and old != new:
                edges.add((min(old,new),max(old,new)))
    nodes=np.asarray(sorted(cells),np.int32)
    if len(nodes) < 500 or len(edges) < 500:
        raise ValueError("Too few rasterized OSM canal nodes/edges")
    lookup=np.full(N_PIXELS,-1,np.int32)
    lookup[nodes]=np.arange(len(nodes),dtype=np.int32)
    edge_nodes=np.asarray([(lookup[a],lookup[b]) for a,b in sorted(edges)],np.int32)
    row=np.concatenate((edge_nodes[:,0],edge_nodes[:,1],np.arange(len(nodes))))
    col=np.concatenate((edge_nodes[:,1],edge_nodes[:,0],np.arange(len(nodes))))
    adj=csr_matrix((np.ones(len(row),np.float32),(row,col)),shape=(len(nodes),len(nodes)))
    walk=csr_matrix(adj.multiply(1/np.asarray(adj.sum(axis=1)).ravel()[:,None]))
    n_components,labels=connected_components(adj,directed=False)
    component_sizes=np.bincount(labels)
    largest=int(np.argmax(component_sizes))
    component_nodes=nodes[labels==largest]
    y,x=np.divmod(component_nodes,WIDTH)
    span=float(np.hypot((x.max()-x.min())*PIXEL_M,(y.max()-y.min())*PIXEL_M))/1000
    candidate_flat=np.flatnonzero(candidate.ravel())
    cy,cx=np.divmod(candidate_flat,WIDTH)
    ny,nx=np.divmod(nodes,WIDTH)
    tree=cKDTree(np.column_stack((nx,ny)).astype(np.float32)*PIXEL_M)
    distance,nearest=tree.query(np.column_stack((cx,cy)).astype(np.float32)*PIXEL_M)
    nearest=nearest.astype(np.int32)
    distance=distance.astype(np.float32)
    np.savez_compressed(GRAPH_FILE,nodes=nodes,edge_nodes=edge_nodes,
                        nearest=nearest,distance_m=distance,
                        candidate_flat=candidate_flat.astype(np.int32))
    audit={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source":OSM_PBF.name,"osm_canal_ways":len(handler.canal),
           "raster_nodes":len(nodes),"undirected_edges":len(edges),
           "components":int(n_components),
           "largest_component_nodes":int(component_sizes.max()),
           "largest_component_span_km":round(span,2),
           "candidate_cells":len(candidate_flat),
           "candidate_within_500m":int((distance<=500).sum()),
           "candidate_within_1000m":int((distance<=1000).sum()),
           "median_candidate_distance_m":round(float(np.median(distance)),1),
           "graph_file":str(GRAPH_FILE)}
    AUDIT_FILE.write_text(json.dumps(audit,indent=2)+"\n")
    return audit


def load(candidate: np.ndarray) -> dict:
    if not GRAPH_FILE.exists():
        build(candidate)
    with np.load(GRAPH_FILE) as saved:
        nodes=saved["nodes"]
        edges=saved["edge_nodes"]
        nearest=saved["nearest"]
        distance=saved["distance_m"]
        candidate_flat=saved["candidate_flat"]
    if not np.array_equal(candidate_flat,np.flatnonzero(candidate.ravel())):
        raise ValueError("Canal graph candidate cells changed")
    row=np.concatenate((edges[:,0],edges[:,1],np.arange(len(nodes))))
    col=np.concatenate((edges[:,1],edges[:,0],np.arange(len(nodes))))
    adj=csr_matrix((np.ones(len(row),np.float32),(row,col)),shape=(len(nodes),len(nodes)))
    walk=csr_matrix(adj.multiply(1/np.asarray(adj.sum(axis=1)).ravel()[:,None]))
    return {"nodes":nodes,"walk":walk,"nearest":nearest,
            "distance_m":distance,"candidate_flat":candidate_flat}


def propagate(graph: dict, counts: np.ndarray, steps: tuple[int,...]=(8,16)) -> dict[int,np.ndarray]:
    """Mean prior reports per projected cell after walking along canal edges."""
    closest=graph["nearest"]
    distance=graph["distance_m"]
    weights=np.clip(1-distance/1250,0,1).astype(np.float32)
    active=weights>0
    n=len(graph["nodes"])
    numerator=np.bincount(closest[active],weights=counts[active]*weights[active],minlength=n).astype(np.float32)
    denominator=np.bincount(closest[active],weights=weights[active],minlength=n).astype(np.float32)
    result={}
    walk=graph["walk"]
    for step in range(1,max(steps)+1):
        numerator=walk @ numerator
        denominator=walk @ denominator
        if step in steps:
            mean=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
            result[step]=(np.log1p(mean[closest])*weights).astype(np.float32)
    return result


if __name__=="__main__":
    from flood_citywide_history_ranker import load_grid
    data=load_grid()
    print(build(data["candidate"]))
