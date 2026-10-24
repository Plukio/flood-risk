"""OSM shared-node canal graph for the Bangkok 250 m cell grid.

Crossing lines connect only when OSM shares a node. Long OSM segments receive
intermediate graph nodes, preserving their original topology.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from flood_bangkok_canal_graph_features import CanalGraph
from flood_bangkapi_surface_grid import OSM_PBF
from flood_citywide_canal_graph import HEIGHT, WIDTH, PIXEL_M
from flood_multi_target_citywide_data import COLS, OUT, ROWS, TILE_M, X0, Y0


GRAPH_FILE=OUT/"osm_shared_node_canal_graph_250m.npz"
AUDIT_FILE=OUT/"osm_shared_node_canal_graph_250m_audit.json"


def build(candidate:np.ndarray):
    if candidate.shape!=(HEIGHT,WIDTH):
        raise ValueError("Candidate grid changed")
    OUT.mkdir(parents=True,exist_ok=True)
    bounds=(X0-1000,Y0-1000,X0+COLS*TILE_M+1000,Y0+ROWS*TILE_M+1000)
    handler=CanalGraph(bounds)
    handler.apply_file(str(OSM_PBF),locations=True)
    osm_ids=list(handler.graph.nodes)
    index={node:i for i,node in enumerate(osm_ids)}
    coords=[handler.coords[node] for node in osm_ids]
    edges=[]
    for first,second,attrs in handler.graph.edges(data=True):
        a=np.asarray(handler.coords[first],np.float64)
        b=np.asarray(handler.coords[second],np.float64)
        length=float(np.hypot(*(b-a)))
        n=max(1,math.ceil(length/250))
        previous=index[first]
        for step in range(1,n):
            point=a+(b-a)*(step/n)
            new=len(coords)
            coords.append((float(point[0]),float(point[1])))
            edges.append((previous,new))
            previous=new
        edges.append((previous,index[second]))
    coords=np.asarray(coords,np.float32)
    edges=np.asarray(edges,np.int32)
    n_nodes=len(coords)
    if n_nodes<1000 or len(edges)<1000:
        raise ValueError("Too few OSM canal graph nodes/edges")
    row=np.concatenate((edges[:,0],edges[:,1],np.arange(n_nodes)))
    col=np.concatenate((edges[:,1],edges[:,0],np.arange(n_nodes)))
    adj=csr_matrix((np.ones(len(row),np.float32),(row,col)),shape=(n_nodes,n_nodes))
    n_components,labels=connected_components(adj,directed=False)
    component_sizes=np.bincount(labels)
    largest=coords[labels==np.argmax(component_sizes)]
    span=float(np.hypot(*(largest.max(axis=0)-largest.min(axis=0))))/1000
    candidate_flat=np.flatnonzero(candidate.ravel())
    cy,cx=np.divmod(candidate_flat,WIDTH)
    centres=np.column_stack((X0+(cx+.5)*PIXEL_M,
                             Y0+ROWS*TILE_M-(cy+.5)*PIXEL_M))
    distance,nearest=cKDTree(coords).query(centres)
    distance=distance.astype(np.float32)
    nearest=nearest.astype(np.int32)
    np.savez_compressed(GRAPH_FILE,coords=coords,edge_nodes=edges,
                        nearest=nearest,distance_m=distance,
                        candidate_flat=candidate_flat.astype(np.int32))
    audit={"built_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source":OSM_PBF.name,"osm_canal_ways":handler.ways,
           "original_osm_nodes":len(osm_ids),"raster_interval_nodes":n_nodes-len(osm_ids),
           "graph_nodes":n_nodes,"edges":len(edges),"components":int(n_components),
           "largest_component_nodes":int(component_sizes.max()),
           "largest_component_span_km":round(span,2),
           "candidate_cells":len(candidate_flat),
           "candidate_within_500m":int((distance<=500).sum()),
           "candidate_within_1000m":int((distance<=1000).sum()),
           "median_candidate_distance_m":round(float(np.median(distance)),1),
           "topology":"OSM shared node IDs only; crossings in the same 250m cell do not automatically connect",
           "limitations":"No surveyed drainage direction, pipe connection, gate/pump state or outfall",
           "graph_file":str(GRAPH_FILE)}
    AUDIT_FILE.write_text(json.dumps(audit,indent=2)+"\n")
    return audit


def load(candidate:np.ndarray):
    if not GRAPH_FILE.exists():
        build(candidate)
    with np.load(GRAPH_FILE) as z:
        coords,edges,nearest,distance,candidate_flat=(z[k] for k in
            ("coords","edge_nodes","nearest","distance_m","candidate_flat"))
    if not np.array_equal(candidate_flat,np.flatnonzero(candidate.ravel())):
        raise ValueError("Canal graph candidate cells changed")
    n=len(coords)
    row=np.concatenate((edges[:,0],edges[:,1],np.arange(n)))
    col=np.concatenate((edges[:,1],edges[:,0],np.arange(n)))
    adj=csr_matrix((np.ones(len(row),np.float32),(row,col)),shape=(n,n))
    walk=csr_matrix(adj.multiply(1/np.asarray(adj.sum(axis=1)).ravel()[:,None]))
    return {"nodes":np.arange(n,dtype=np.int32),"coords":coords,
            "walk":walk,"nearest":nearest,"distance_m":distance,
            "candidate_flat":candidate_flat}


if __name__=="__main__":
    from flood_citywide_history_ranker import load_grid
    print(build(load_grid()["candidate"]))
