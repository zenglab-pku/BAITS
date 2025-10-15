from sklearn.cluster import DBSCAN
import scanpy as sc
import anndata as ad

def dbscan_cluster(adata_tmp, eps=50, min_samples=5, plot=True):  
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

    Returns
    -------
    AnnData
        The updated AnnData object with a new column `Bcell_aggregate_index` 
        in `obs`, containing the DBSCAN cluster labels.
        A visualization of the clustering result is also generated.

    """
    # Extract spatial coordinates for clustering
    filtered_positions = adata_tmp.obsm['spatial']
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(filtered_positions)
    
    # Add DBSCAN cluster labels to the AnnData object
    adata_tmp.obs['Bcell_aggregate_index'] = labels
    
    # Visualize the clustering results
    if plot:
        sc.pl.spatial(adata_tmp, color='Bcell_aggregate_index', spot_size=50, cmap='tab20_r', title='', legend_loc='on data')
    
    # Return the updated AnnData object with clustering labels
    return adata_tmp
    