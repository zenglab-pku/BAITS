from sklearn.cluster import DBSCAN
import scanpy as sc
import anndata as ad
import numpy as np

def dbscan_cluster(adata_tmp, eps=50, min_samples=5, plot=True,spot_size=50, cmap='tab20_r', title='BLA_id'):  
    """
    Perform DBSCAN clustering on spatial coordinates in an AnnData object.

    Parameters
    ----------
    adata_tmp
        The AnnData object containing spatial coordinates in `obsm['spatial']`.
    eps
        The maximum distance between two samples for one to be considered 
        as in the neighborhood of the other. Default is 50.
    min_samples
        The number of samples in a neighborhood for a point to be considered 
        as a core point. Default is 5.
    plot : bool, optional (default=True)
        If True, generates a spatial scatter plot visualizing the identified clusters. 
    spot_size : float, optional (default=50)
        Marker size for individual cells in the spatial visualization plot.

    Returns
    -------
    AnnData
        The updated AnnData object with a new column `BLA_id` 
        in `obs`, containing the DBSCAN cluster labels.

    """
    # Extract spatial coordinates for clustering
    tmp_res = adata_tmp[adata_tmp.obs['filtered_coords']].copy()
    filtered_positions = tmp_res.obsm['spatial']
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(filtered_positions)
    
    # Add DBSCAN cluster labels to the AnnData object
    adata_tmp.obs['BLA_id'] = np.nan
    adata_tmp.obs.loc[tmp_res.obs.index, 'BLA_id'] = labels
    
    # Visualize the clustering results
    if plot:
        sc.pl.spatial(adata_tmp, color='BLA_id', spot_size=spot_size, cmap=cmap,
                      legend_fontsize='medium', legend_fontweight='normal', 
                      title=title, legend_loc='on data')
    
    # Return the updated AnnData object with clustering labels
    return adata_tmp
    