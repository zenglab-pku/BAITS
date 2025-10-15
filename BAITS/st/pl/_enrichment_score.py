# import scanpy as sc
# import matplotlib.pyplot as plt

# def enrichment_score(adata_tmp, score_name, if_filtered=False):
#     """
#     Plot the spatial data, either the original or filtered based on the `if_filtered` parameter.
    
#     Parameters
#     ----------
#     adata_tmp: AnnData object
#         The spatial data.
#     score_name: str
#         the name of the score (e.g., Bcell_enrichment) to color the plot.
#     if_filtered: bool
#         whether to plot the filtered data (True) or the original data (False).
#     """
#     fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
#     if if_filtered:
#         # Plot the filtered spatial data
#         if 'filtered_coords' in adata_tmp.obs:
#             filtered_adata = adata_tmp[adata_tmp.obs['filtered_coords']]
#             sc.pl.spatial(filtered_adata, color=score_name, ax=ax, show=False, spot_size=50)
#         else:
#             print("Warning: 'filtered_coords' not found. Plotting original data.")
#             sc.pl.spatial(adata_tmp, color=score_name, ax=ax, show=False, spot_size=50)
#     else:
#         # Plot the original spatial data
#         sc.pl.spatial(adata_tmp, color=score_name, ax=ax, show=False, spot_size=50)
    
#     plt.show()
