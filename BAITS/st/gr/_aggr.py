from typing import List, Tuple, cast, Union
from tqdm.auto import tqdm
from scipy.sparse import spdiags
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sps
import warnings
warnings.filterwarnings('ignore') 


def str2list(value: Union[str, list]) -> list:
    """Check whether value is a string. If so, converts into a list containing value."""
    return [value] if isinstance(value, str) else value


def _hop(adj_hop, adj, adj_visited=None):
    adj_hop = adj_hop @ adj    

    if adj_visited is not None:
        adj_hop = adj_hop > adj_visited  # Logical not for sparse matrices
        adj_visited = adj_visited + adj_hop
    return adj_hop, adj_visited


def _mul_broadcast(mat1, mat2):
    return spdiags(mat2, 0, len(mat2), len(mat2)) * mat1
    
def _normalize(adj):
    deg = np.array(np.sum(adj, axis=1)).squeeze()
    with warnings.catch_warnings():
        # If a cell doesn't have neighbors deg = 0 -> divide by zero
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        deg_inv = 1 / deg
    deg_inv[deg_inv == float("inf")] = 0
    return _mul_broadcast(adj, deg_inv) 

def _setdiag(array, value):  
    if isinstance(array, sps.csr_matrix):
        array = array.tolil()
    array.setdiag(value)
    array = array.tocsr()
    if value == 0:
        array.eliminate_zeros()
    return array

def _aggregate(adj, x, method):
    if method == "mean":
        return _aggregate_mean(adj, x)
    elif method == "var":
        return _aggregate_var(adj, x)
    else:
        raise NotImplementedError
###
def _aggregate_mean(adj, x):  
    return adj @ x

def _aggregate_var(adj, x):
    mean = adj @ x
    mean_squared = adj @ (x * x)
    return mean_squared - mean * mean

def _aggregate(adj, x, method):
    if method == "mean":
        return _aggregate_mean(adj, x)
    elif method == "var":
        return _aggregate_var(adj, x)
    else:
        raise NotImplementedError
###
def _aggregate_neighbors(
    adj: sps.spmatrix,
    X: np.ndarray,
    nhood_layers: list,
    aggregations:  "mean",
    disable_tqdm: bool = True,
) -> np.ndarray:
    adj = adj.astype(bool)
    adj = _setdiag(adj, 0)
    adj_hop = adj.copy() 
    adj_visited = _setdiag(adj.copy(), 1) 

    Xs = []
    for i in tqdm(range(0, max(nhood_layers) + 1), disable=disable_tqdm):
        if i in nhood_layers:
            if i == 0:
                Xs.append(X)
            else:
                if i > 1:
                    adj_hop, adj_visited = _hop(adj_hop, adj, adj_visited)
                adj_hop_norm = _normalize(adj_hop) 
                
                for agg in aggregations:
                    x = _aggregate(adj_hop_norm, X, agg)
                    
                    Xs.append(x)
    if sps.issparse(X):
        return sps.hstack(Xs)
    else:
        return np.hstack(Xs)


def aggregate_neighbors(
    adata,
    n_layers, 
    aggregations =  "mean",
    connectivity_key='spatial_connectivities',
    use_rep = 'X_pca', 
    sample_key = 'sample',
    out_key  = "X_cellcharter",
    copy: bool = False,  
) :
    """
    Aggregate the features from each neighborhood layers and concatenate them
    
    Parameters
    -----------
    adata : AnnData
        The annotated data matrix (single-cell dataset in AnnData format).
    n_layers : int or list
        The number of neighborhood layers to aggregate. If an integer is given, 
        it is converted into a list ranging from 0 to `n_layers`.
    aggregations : str or list, default="mean"
        The type of aggregation(s) to perform (e.g., "mean", "sum").
    connectivity_key : str, optional
        Key in `adata.obsp` that contains the spatial connectivity matrix.
    use_rep : str, optional
        Key in `adata.obsm` that specifies which representation (feature matrix) to use. 
        If `None`, the function uses `adata.X`.
    sample_key : str, optional
        Key in `adata.obs` that identifies sample/grouping information. If `None`, all 
        cells are considered together.
    out_key : str, default="X_cellcharter"
        Key under which the aggregated features are stored in `adata.obsm`.
    copy : bool, default=False
        If True, returns the aggregated feature matrix instead of modifying `adata`.
    
    Returns
    --------
    - If `copy` is True, returns the aggregated feature matrix as a numpy array or sparse matrix.
    - Otherwise, modifies `adata.obsm` in place, storing the aggregated matrix under `out_key`.
    """
    connectivity_key = connectivity_key
    sample_key = sample_key
    aggregations = str2list(aggregations)

    X = adata.X if use_rep is None else adata.obsm[use_rep]

    if isinstance(n_layers, int):
        n_layers = list(range(n_layers + 1))
    
    if sps.issparse(X):
        X_aggregated = sps.dok_matrix(
            (X.shape[0], X.shape[1] * ((len(n_layers) - 1) * len(aggregations) + 1)), dtype=np.float32     
        )
    else:
        X_aggregated = np.empty(
            (X.shape[0], X.shape[1] * ((len(n_layers) - 1) * len(aggregations) + 1)), dtype=np.float32
        )

    if sample_key in adata.obs:
        samples = adata.obs[sample_key].unique()    
        sample_idxs = [adata.obs[sample_key] == sample for sample in samples]
    else:
        sample_idxs = [np.arange(adata.shape[0])]

    for idxs in tqdm(sample_idxs, disable=(len(sample_idxs) == 1)):
        X_sample_aggregated = _aggregate_neighbors(
            adj=adata[idxs].obsp[connectivity_key],
            X=X[idxs],
            nhood_layers=n_layers,
            aggregations=aggregations,
            disable_tqdm=(len(sample_idxs) != 1),
        )
        X_aggregated[idxs] = X_sample_aggregated

    if isinstance(X_aggregated, sps.dok_matrix):
        X_aggregated = X_aggregated.tocsr()

    if copy:
        return X_aggregated

    adata.obsm[out_key] = X_aggregated