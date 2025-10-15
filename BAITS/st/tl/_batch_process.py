from typing import Optional, Callable, Union
import anndata as ad

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
        Optional column name that represent the interested features
    processing_func 
        Function to apply to each sample batch.
    func_kwargs
        Additional arguments for the processing function.
    verbose
        Whether to print progress information. 
    Returns
    -------
    AnnData
        The concatenated adata from all processed samples.
    """
    # Validate inputs
    if sample_name not in adata.obs:
        raise ValueError(f"❌ [Error] Sample identifier '{sample_name}' not found in adata.obs")
    
    if score_name is not None and score_name not in adata.obs:
        raise ValueError(f"❌ [Error] Score column '{score_name}' not found in adata.obs")
    
    # Initialize function kwargs if not provided
    if func_kwargs is None:
        func_kwargs = {}
    
    # Filter data if score_name is provided
    # Generally, users wont't be interested in the region with score_name < 0
    if score_name is not None:
        adata = adata[adata.obs[score_name] > 0].copy()
    
    # Get unique samples
    samples = set(adata.obs[sample_name])
    processed_samples = []
    
    for sample in samples:
        if verbose:
            print(f'🔬 Start process sample:\t{sample}') 

        # Subset the data for current sample
        adata_sample = adata[adata.obs[sample_name] == sample].copy()
        
        # Apply processing function if provided
        if processing_func is not None:
            adata_sample = processing_func(adata_sample, **func_kwargs)
            if processing_func.__name__ == "kde_filter":  # 通过函数名判断
                try:
                    from BAITS.st.pl import kde_filter  # 动态导入
                    kde_filter(adata_sample, score_name, figsize=(12, 3), spot_size=50)
                except ImportError:
                    print("👀 [Warning] BAITS.st.pl.kde_filter not available - skipping plotting")

            if processing_func.__name__ == "dbscan_cluster":  # 通过函数名判断
                try:
                    from BAITS.st.pl import dbscan_cluster  # 动态导入
                    dbscan_cluster(adata_sample, score_name, figsize=(12, 3), spot_size=50)
                except ImportError:
                    print("👀 [Warning] BAITS.st.pl.dbscan_cluster not available - skipping plotting")

        processed_samples.append(adata_sample)
        
        if verbose:
            print(f'✅ Sample: {sample} done!')
    
    # Combine all processed samples
    return ad.concat(processed_samples)


