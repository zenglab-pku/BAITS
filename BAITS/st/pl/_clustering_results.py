import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import anndata as ad

def clustering_results(adata, cluster_column, color_dict, spot_size, plot=True):
    """
    Plots the clustering results, displaying the cluster labels for each data point using the specified colors.
    
    Parameters
    ----------
    adata: AnnData object
        containing spatial information and clustering labels.
    - cluster_column: str
        the name of the column containing the cluster labels.(eg. the name of the column in use_rep in cluster_Auto_k)
    - color_dict: dict
        a dictionary specifying the color for each cluster label (optional).
    - spot_size: int
        the size of the data points (default is 50).
    """

    color_dict = {}
    cluster_lst = np.sort(list(set(adata.obs[cluster_column])))
    for i in range(len(cluster_lst)): 
        color_dict[cluster_lst[i]] = color_lst[i]
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Use Scanpy's spatial plotting function
    if plot:
        sc.pl.spatial(adata, color=[cluster_column], ax=ax, show=False, spot_size=spot_size)
    
    # Display the plot
    plt.title(f"Clustering results for {cluster_column}")  #这里的title可以改一下
    plt.show()
