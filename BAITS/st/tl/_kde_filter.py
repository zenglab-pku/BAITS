from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity
from collections import defaultdict
import numpy as np
from scipy.ndimage import binary_opening
import multiprocessing
import time
import warnings
warnings.filterwarnings('ignore') 

def calculate_density(kde, pos):
    return np.exp(kde.score_samples([pos]))
    
def kde_filter(adata_tmp,
               score_name,
               high_binSize = 100,
               default_thread_num=36,
               clean_mask_size=(3,3) ):
    
    """
    Filter spatial transcriptomics (ST) data using Kernel Density Estimation (KDE).

    Parameters
    ----------
    adata_tmp : AnnData object
        The input AnnData object containing spatial and score data.
    score_name : str
        The name of the column in `adata_tmp.obs` that contains the scores to be used as weights.
    high_binSize : int, optional (default: 100)
        The size of the bins used for spatial discretization.
    default_thread_num : int, optional (default: 36)
        The maximum number of threads to use for parallel KDE scoring.
    clean_mask_size : tuple of int, optional (default: (3, 3))
        The size of the structuring element used for binary opening during mask cleaning.

    Returns
    -------
    AnnData
        A subset of the input AnnData object, filtered to retain only the valid spatial coordinates.
    
    """

    # original locations 
    start_time = time.time()
    positions_original = adata_tmp.obsm['spatial']

    # prepare
    x = np.array(positions_original[:, 0]); y = np.array(positions_original[:, 1])
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    Z = np.array(adata_tmp.obs[score_name]).astype(np.float32)
    high_binSize = high_binSize
    bandwidth = int( high_binSize / 3 )
    points = np.vstack([x, y]).T  

    # record original coords
    xy_dict = defaultdict(list)   
    for i in range(len(x)):
        Bx = (x[i] - x_min) // high_binSize * high_binSize + x_min
        By = (y[i] - y_min) // high_binSize * high_binSize + y_min
        xy_dict[(Bx, By)].append([x[i], y[i]])  # 直接追加，无需检查键是否存在
        
    # kde fit
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(points, sample_weight=Z)   
    x_grid, y_grid = np.meshgrid(np.arange(x_min,x_max,high_binSize), np.arange(y_min,y_max,high_binSize)) 
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()]).T     # 	np.vstack 是 np.concatenate 函数的一个方便函数，等价于 np.concatenate，但指定了 axis=0 即在第一个轴（行）上进行堆叠
    x_grid = x_grid.astype(np.float32); y_grid = y_grid.astype(np.float32); positions = positions.astype(np.float32)
    print(f"Step 1 (Prepare): {time.time() - start_time:.2f} s")
    
    # kde score
    start_time = time.time()
    num_cores = min(multiprocessing.cpu_count(), default_thread_num)
    density = Parallel(n_jobs=num_cores)(delayed(calculate_density)(kde, pos) for pos in positions) 
    density = (np.array(density).reshape(x_grid.shape))    
    print(f"Step 2 (kde score): {time.time() - start_time:.2f} s")

    # set density cutoff
    start_time = time.time()
    density_flat = density.ravel()
    # threshold = np.percentile(density_flat, 50) 
    threshold = np.mean(density_flat)                ################### Cutoff is setting as mean value  ###################

    density_thresholded = np.where(density > threshold, density, np.nan)
    binary_mask = ~np.isnan(density_thresholded)
    cleaned_mask = binary_opening(binary_mask, structure=np.ones(clean_mask_size))
    density_cleaned = np.where(cleaned_mask, density, np.nan) 
    mask = ~np.isnan(density_cleaned)
    x_coords = x_grid[mask]
    y_coords = y_grid[mask]

    adata_tmp.uns["binary_mask"] = binary_mask
    adata_tmp.uns["mask"] = mask
    
    coords = np.vstack([x_coords, y_coords]).T
    orginal_coords = []
    for coord in coords:
        orginal_coords += (xy_dict[tuple(coord)])

    # obtain the legal coords
    original_coords_set = {tuple(row) for row in orginal_coords}
    isin_ = [tuple(x) in original_coords_set for x in positions_original]
    print(f"Step 3 (obtain legal coords): {time.time() - start_time:.2f} s")

    adata_tmp.obs['filtered_coords'] = isin_

    return adata_tmp[isin_]