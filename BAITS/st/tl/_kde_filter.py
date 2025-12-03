from collections import defaultdict
import numpy as np
from scipy.ndimage import binary_opening
import multiprocessing as mp
import time
import warnings
warnings.filterwarnings('ignore') 

def kde_worker(args):
    from sklearn.neighbors import KernelDensity
    batch_indices, all_positions, kde_params = args
    bandwidth = kde_params['bandwidth']
    kernel = kde_params['kernel']
    points = kde_params['points']
    sample_weights = kde_params['sample_weights']
    
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(points, sample_weight=sample_weights)
    
    batch_positions = all_positions[batch_indices]
    return batch_indices, np.exp(kde.score_samples(batch_positions))
    
def kde_filter(adata_sample, score_name, threshold_method='percentile', custom_threshold=90,
               high_binSize = 100, default_thread_num=36, clean_mask_size=(3,3),
               plot=True,figsize=(16, 4),spot_size=50 ):
    """
    Filter spatial transcriptomics (ST) data using Kernel Density Estimation (KDE).

    Parameters
    ----------
    adata_tmp : AnnData object
        The input AnnData object containing spatial and score data.
    score_name : str
        The name of the column in `adata_tmp.obs` that contains the scores to be used as weights.
    threshold_method : str
        Method for determining the density cutoff threshold. Available options:
        - 'mean': Use arithmetic mean of density values
        - 'median': Use median density value, robust to outliers
        - 'percentile': Use specified percentile of density distribution
        - 'custom': Use user-provided custom threshold value
    custom_threshold : float, optional (default=None)
        Custom threshold value when `threshold_method='custom'`, or percentile value when `threshold_method='percentile'`. 
        For percentile method, values should be between 0 and 100. 
    high_binSize : int, optional (default: 100)
        The size of the bins used for spatial discretization.
    default_thread_num : int, optional (default: 36)
        The maximum number of threads to use for parallel KDE scoring.
    clean_mask_size : tuple of int, optional (default: (3, 3))
        The size of the structuring element used for binary opening during mask cleaning.
    plot : bool, optional (default=True)
        If True, automatically generates visualization of filtered results.
    figsize : tuple of float, optional (default=(16, 4))
        Figure dimensions (width, height) in inches for the generated visualization. 
        Only applicable when `plot=True`.
    spot_size : float (default=50)
        Size of the scatter point/shape

    Returns
    -------
    AnnData
       The annotated AnnData object.

    """

    if score_name is not None:
        adata_tmp = adata_sample[adata_sample.obs[score_name] > 0].copy()
    
    start_time = time.time()
    positions_original = adata_tmp.obsm['spatial']

    # prepare data
    x = np.array(positions_original[:, 0])
    y = np.array(positions_original[:, 1])
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    Z = np.array(adata_tmp.obs[score_name]).astype(np.float32)
    bandwidth = int(high_binSize / 3)
    points = np.vstack([x, y]).T  

    # keep original locus
    xy_dict = defaultdict(list)   
    for i in range(len(x)):
        Bx = (x[i] - x_min) // high_binSize * high_binSize + x_min
        By = (y[i] - y_min) // high_binSize * high_binSize + y_min
        xy_dict[(Bx, By)].append([x[i], y[i]])
        
    x_grid, y_grid = np.meshgrid(
        np.arange(x_min, x_max, high_binSize), 
        np.arange(y_min, y_max, high_binSize)
    ) 
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    print(f"🔎 Step 1 - Process data:\t\t{time.time() - start_time:.2f} s")
    print(f"📊 Spatial cell number to evaluate:\t\t{len(positions)}") 
    
    # Prepare KDE parrameters
    kde_params = {
        'bandwidth': bandwidth, 'kernel': 'gaussian', 'points': points,'sample_weights': Z
    }
    
    start_time = time.time()
    num_cores = min(mp.cpu_count(), default_thread_num)
    batch_size = len(positions) // num_cores + 1
    batches = [range(i, min(i + batch_size, len(positions))) for i in range(0, len(positions), batch_size)]
    
    tasks = [(batch, positions, kde_params) for batch in batches]
    
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(kde_worker, tasks)
    
    density = np.zeros(len(positions))
    for indices, values in results:
        density[indices] = values
    
    density = density.reshape(x_grid.shape)
    print(f"🏋️ Step 2 - KDE scoring:\t\t{time.time() - start_time:.2f} s")

    start_time = time.time()
    density_flat = density.ravel()
    
    # cutoff selection
    if threshold_method == 'mean':
        threshold = np.mean(density_flat)
        print(f"📊 Using mean threshold:\t\t{threshold:.3f}")
    elif threshold_method == 'median':
        threshold = np.median(density_flat)
        print(f"📊 Using median threshold:\t\t{threshold:.3f}")
    elif threshold_method == 'percentile':
        if custom_threshold is None:
            # defaulting 90th percentile
            threshold = np.percentile(density_flat, 90)
            print(f"📊 Defaulting 90th percentile threshold:\t\t{threshold:.3f}")
        else:
            threshold = np.percentile(density_flat, custom_threshold)
            print(f"📊 Using", custom_threshold, f"th percentile threshold:\t\t{threshold:.3f}") 
    elif threshold_method == 'custom':
        if custom_threshold is not None:
            threshold = custom_threshold
            print(f"📊 Using custom threshold:\t\t{threshold:.3f}")
        else:
            print(f"[Error] Please input custom threshold!")
    else:
        threshold = np.mean(density_flat)
        print(f"📊 Defaulting to mean threshold:\t\t{threshold:.3f}")

    density_thresholded = np.where(density > threshold, density, np.nan)
    binary_mask = ~np.isnan(density_thresholded)
    cleaned_mask = binary_opening(binary_mask, structure=np.ones(clean_mask_size))
    density_cleaned = np.where(cleaned_mask, density, np.nan) 
    mask = ~np.isnan(density_cleaned)
    x_coords = x_grid[mask]
    y_coords = y_grid[mask]

    adata_tmp.uns["binary_mask"] = binary_mask
    adata_tmp.uns["mask"] = mask
    
    # Obtain original locus
    coords = np.vstack([x_coords, y_coords]).T
    orginal_coords = []
    for coord in coords:
        coord_tuple = tuple(coord)
        if coord_tuple in xy_dict:
            orginal_coords.extend(xy_dict[coord_tuple])

    # Obtain legal locus
    original_coords_set = {tuple(row) for row in orginal_coords}
    isin_ = [tuple(x) in original_coords_set for x in positions_original]
    print(f"🔬 Step 3 - Obtain correct coords:\t\t{time.time() - start_time:.2f} s") 
    
    adata_tmp.obs['filtered_coords'] = isin_
    adata_sample.obs['filtered_coords'] = False
    adata_sample.obs.loc[adata_tmp.obs.index,'filtered_coords'] = adata_tmp.obs['filtered_coords'] 

    if plot:
        print(f"🎨 Plot results ") 
        from ..pl import kde_res
        kde_res(adata_tmp, score_name, figsize=figsize,spot_size=spot_size)

    return adata_sample
    