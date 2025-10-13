from typing import Optional, Callable, Union
from sklearn.cluster import DBSCAN
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

def batch_process(
    adata: ad.AnnData,
    sample_name: str,
    score_name: Optional[str] = None,
    processing_func: Optional[Callable] = None,
    func_kwargs: Optional[dict] = None,
    verbose: bool = True
) -> ad.AnnData:
    """
    Process AnnData objects in batches by sample.
    
    Parameters
    ----------
    adata
        The AnnData object to process.
    sample_name
        Column name in `obs` that identifies samples.
    score_name
        Optional column name to filter data before processing.
    processing_func
        Function to apply to each sample batch.
    func_kwargs
        Additional arguments for the processing function.
    verbose
        Whether to print progress information.
        
    Returns
    -------
    AnnData
        The concatenated results from all processed samples.
    """
    # Validate inputs
    if sample_name not in adata.obs:
        raise ValueError(f"Sample identifier '{sample_name}' not found in adata.obs")
    
    if score_name is not None and score_name not in adata.obs:
        raise ValueError(f"Score column '{score_name}' not found in adata.obs")
    
    # Initialize function kwargs if not provided
    if func_kwargs is None:
        func_kwargs = {}
    
    # Filter data if score_name is provided
    if score_name is not None:
        adata = adata[adata.obs[score_name] > 0].copy()
    
    # Get unique samples
    samples = set(adata.obs[sample_name])
    processed_samples = []
    
    for sample in samples:
        if verbose:
            print(f'============ Processing sample: {sample} ============')

        # Subset the data for current sample
        adata_sample = adata[adata.obs[sample_name] == sample].copy()
        
        # Apply processing function if provided
        if processing_func is not None:
            adata_sample = processing_func(adata_sample, **func_kwargs)
            if processing_func.__name__ == "kde_filter":  # 通过函数名判断
                try:
                    from B_HIT.st.pl import kde_filter  # 动态导入
                    kde_filter(adata_sample, score_name, figsize=(12, 3), spot_size=50)
                except ImportError:
                    print("Warning: B_HIT.st.pl.kde_filter not available - skipping plotting")

            if processing_func.__name__ == "dbscan_cluster":  # 通过函数名判断
                try:
                    from B_HIT.st.pl import dbscan_cluster  # 动态导入
                    dbscan_cluster(adata_sample, score_name, figsize=(12, 3), spot_size=50)
                except ImportError:
                    print("Warning: B_HIT.st.pl.dbscan_cluster not available - skipping plotting")


        processed_samples.append(adata_sample)
        
        if verbose:
            print(f'============ Finished sample: {sample} ============\n')
    
    # Combine all processed samples
    return ad.concat(processed_samples)


