import warnings
import numpy as np
import pandas as pd
from functools import partial
from itertools import chain
from typing import Any, cast
from anndata import AnnData
from scanpy import logging as logg
from collections.abc import Iterable 
from anndata.utils import make_index_unique
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import ( SparseEfficiencyWarning, block_diag, csr_matrix, isspmatrix_csr, spmatrix)

from ._utils import _save_data

def spatial_neighbors(
    adata, sample_key, coord_type, n_neighs=10, radius=None, delaunay=True,
    spatial_key='spatial', n_rings=1,
    percentile = None, set_diag = False, key_added = "spatial",
    copy = False
): 
    if coord_type is None:
        if radius is not None:
            logg.warning(
                f"Graph creation with `radius` is only available when `coord_type =generic` specified. "
                f"Ignoring parameter `radius = {radius}`."
            )
        coord_type = 'grid' if Key.uns.spatial in adata.uns else 'generic'
    else:
        coord_type = coord_type

    if sample_key is not None:
        adata.obs[sample_key] = pd.Categorical(adata.obs[sample_key])
        libs = adata.obs[sample_key].cat.categories
        make_index_unique(adata.obs_names)
    else:
        libs = [None]

    _build_fun = partial(
        _spatial_neighbor,
        spatial_key=spatial_key,
        coord_type=coord_type,
        n_neighs=n_neighs,
        radius=radius,
        delaunay=delaunay,
        n_rings=n_rings,
        set_diag=set_diag,
        percentile=percentile,
    )

    if sample_key is not None:
        mats: list[tuple[spmatrix, spmatrix]] = []
        ixs: list[int] = []
        for lib in libs:
            ixs.extend(np.where(adata.obs[sample_key] == lib)[0])
            mats.append(_build_fun(adata[adata.obs[sample_key] == lib]))
        ixs = cast(list[int], np.argsort(ixs).tolist())
        Adj = block_diag([m[0] for m in mats], format="csr")[ixs, :][:, ixs]
        Dst = block_diag([m[1] for m in mats], format="csr")[ixs, :][:, ixs]
    else:
        Adj, Dst = _build_fun(adata)

    neighs_key = f"{key_added}_neighbors" 
    conns_key = f"{key_added}_connectivities"
    dists_key = f"{key_added}_distances"

    neighbors_dict = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": {
            "n_neighbors": n_neighs,
            "coord_type": coord_type,
            "radius": radius
        },
    }

    if copy:
        return Adj, Dst

    _save_data(adata, attr="obsp", key=conns_key, data=Adj)
    _save_data(adata, attr="obsp", key=dists_key, data=Dst)
    _save_data(adata, attr="uns", key=neighs_key, data=neighbors_dict)


def _spatial_neighbor(
    adata, spatial_key, coord_type, n_neighs = 6, radius = None,
    delaunay = False, n_rings = 1, set_diag = False, percentile = None
): 
    coords = adata.obsm[spatial_key]
    with warnings.catch_warnings():
        if coord_type == "grid":
            Adj, Dst = _build_grid(
                coords, n_neighs=n_neighs, n_rings=n_rings, delaunay=delaunay, set_diag=set_diag,
            )
        elif coord_type == "generic":
            Adj, Dst = _build_connectivity(
                coords, n_neighs=n_neighs, radius=radius, delaunay=delaunay, return_distance=True, set_diag=set_diag,
            )
        else:
            raise NotImplementedError(f"Coordinate type `{coord_type}` is not yet implemented.")

    if coord_type == "generic" and isinstance(radius, Iterable):
        minn, maxx = sorted(radius)[:2]
        mask = (Dst.data < minn) | (Dst.data > maxx)
        a_diag = Adj.diagonal()

        Dst.data[mask] = 0.0
        Adj.data[mask] = 0.0
        Adj.setdiag(a_diag)

    if percentile is not None and coord_type == "generic":
        threshold = np.percentile(Dst.data, percentile)
        Adj[Dst > threshold] = 0.0
        Dst[Dst > threshold] = 0.0

    Adj.eliminate_zeros()
    Dst.eliminate_zeros()      

    return Adj, Dst


def _build_grid(
    coords, n_neighs , n_rings, delaunay = False, set_diag = False) :
    if n_rings > 1:
        Adj: csr_matrix = _build_connectivity(
            coords, n_neighs=n_neighs, neigh_correct=True, set_diag=True, delaunay=delaunay, return_distance=False,
        )
        Res, Walk = Adj, Adj
        for i in range(n_rings - 1):
            Walk = Walk @ Adj
            Walk[Res.nonzero()] = 0.0
            Walk.eliminate_zeros()
            Walk.data[:] = i + 2.0
            Res = Res + Walk
        Adj = Res
        Adj.setdiag(float(set_diag))
        Adj.eliminate_zeros()

        Dst = Adj.copy()
        Adj.data[:] = 1.0
    else:
        Adj = _build_connectivity(
            coords, n_neighs=n_neighs, neigh_correct=True, delaunay=delaunay, set_diag=set_diag,
        )
        Dst = Adj.copy()
    Dst.setdiag(0.0) 
    return Adj, Dst


def _build_connectivity(
    coords, n_neighs, radius = None,
    delaunay = False, neigh_correct = False, set_diag = False, return_distance = False,) :
    N = coords.shape[0]
    if delaunay:
        tri = Delaunay(coords)
        indptr, indices = tri.vertex_neighbor_vertices
        Adj = csr_matrix((np.ones_like(indices, dtype=np.float64), indices, indptr), shape=(N, N))

        if return_distance:
            # fmt: off
            dists = np.array(list(chain(*(
                euclidean_distances(coords[indices[indptr[i] : indptr[i + 1]], :], coords[np.newaxis, i, :])
                for i in range(N)
                if len(indices[indptr[i] : indptr[i + 1]])
            )))).squeeze()
            Dst = csr_matrix((dists, indices, indptr), shape=(N, N))
            # fmt: on
    else:
        r = 1 if radius is None else radius if isinstance(radius, int | float) else max(radius)
        tree = NearestNeighbors(n_neighbors=n_neighs, radius=r, metric="euclidean")
        tree.fit(coords)

        if radius is None:
            dists, col_indices = tree.kneighbors()
            dists, col_indices = dists.reshape(-1), col_indices.reshape(-1)
            row_indices = np.repeat(np.arange(N), n_neighs)
            if neigh_correct:
                dist_cutoff = np.median(dists) * 1.3  # there's a small amount of sway
                mask = dists < dist_cutoff
                row_indices, col_indices, dists = (
                    row_indices[mask],
                    col_indices[mask],
                    dists[mask],
                )
        else:
            dists, col_indices = tree.radius_neighbors()
            row_indices = np.repeat(np.arange(N), [len(x) for x in col_indices])
            dists = np.concatenate(dists)
            col_indices = np.concatenate(col_indices)

        Adj = csr_matrix(
            (np.ones_like(row_indices, dtype=np.float64), (row_indices, col_indices)),
            shape=(N, N),
        )
        if return_distance:
            Dst = csr_matrix((dists, (row_indices, col_indices)), shape=(N, N))

    # radius-based filtering needs same indices/indptr: do not remove 0s
    Adj.setdiag(1.0 if set_diag else Adj.diagonal())
    if return_distance:
        Dst.setdiag(0.0)
        return Adj, Dst

    return Adj
