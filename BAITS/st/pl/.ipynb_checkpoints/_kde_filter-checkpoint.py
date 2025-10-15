import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def kde_filter(adata_tmp, score_name, figsize=(16, 4), spot_size=50):
    """
    Plot original image, enrichment score on the first row; binary_mask and mask on the second row.

    Parameters
    ----------
    adata_tmp : AnnData
        AnnData object containing the data and masks.
    score_name : str
        The gene or feature name used for plotting enrichment scores.
    figsize : tuple, optional (default=(10, 10))
        Size of the figure.
    spot_size : int, optional (default=50)
        Size of the points in spatial plots.
    """

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    sc.pl.spatial(adata_tmp, color=score_name, ax=axes[0], show=False, spot_size=spot_size)
    axes[0].set_title("Original Image")
    filtered_adata = adata_tmp[adata_tmp.obs['filtered_coords']]
    sc.pl.spatial(filtered_adata, color=score_name, ax=axes[1], show=False, spot_size=spot_size)
    axes[1].set_title("Filtered Enrichment Score")

    if "binary_mask" not in adata_tmp.uns or "mask" not in adata_tmp.uns:
        raise ValueError("binary_mask or mask is not stored in adata_tmp.uns!") 
    binary_mask = adata_tmp.uns["binary_mask"]
    mask = adata_tmp.uns["mask"]
    axes[2].matshow(binary_mask, cmap=ListedColormap(['silver', 'indianred']))
    axes[2].set_title("Binary Mask")
    axes[3].matshow(mask, cmap=ListedColormap(['silver', 'indianred']))
    axes[3].set_title("Mask")

    plt.tight_layout()
    plt.show()
