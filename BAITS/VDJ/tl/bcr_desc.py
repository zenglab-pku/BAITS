import numpy as np
import pandas as pd
import itertools

from scipy.spatial import KDTree


def calculate_clone_niche(
    df,
    sample,
    target_clone,
    radius=50,
    scale=2,
    coord_cols=("X", "Y"),
    clone_col="aaClone",
    celltype_col="celltype"
):
    """
    Calculate spatial niche composition around a target BCR clone.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing spatial coordinates and cell type info
    sample : str
        Sample name
    target_clone : str
        Clone name (aaClone)
    radius : float
        Base radius in micrometers
    scale : float
        Spatial scaling factor
    coord_cols : tuple
        Coordinate column names
    clone_col : str
        Clone column name
    celltype_col : str
        Cell type column name
        
    Returns
    -------
    data : dataframe
        Data with near_target_clone column
    niche_prop : series
        Niche composition (proportion)
    niche_count : series
        Niche composition (counts)
    """

    data = df[df['sample'] == sample].reset_index(drop=True)

    coords = data[list(coord_cols)].values
    kdtree = KDTree(coords)

    radius_scaled = radius * scale

    target_indices = data.index[data[clone_col] == target_clone].tolist()

    if len(target_indices) == 0:
        raise ValueError(f"Clone {target_clone} not found in sample {sample}")

    is_near_target = np.zeros(data.shape[0], dtype=bool)

    for target_idx in target_indices:
        target_coord = coords[target_idx]
        neighbors = kdtree.query_ball_point(target_coord, r=radius_scaled)

        for neighbor_idx in neighbors:
            if neighbor_idx != target_idx:
                is_near_target[neighbor_idx] = True

    data = data.copy()
    data['near_target_clone'] = is_near_target

    niche_cells = data[data['near_target_clone']]

    niche_count = niche_cells[celltype_col].value_counts()
    niche_prop = niche_cells[celltype_col].value_counts(normalize=True)

    return data, niche_prop, niche_count


def compute_migraIdx(
    df,
    sample_col='sample',
    clone_col='clone',
    chain_col='Cgene',
    cluster_col='spatial_cluster',
    x_col='X',
    y_col='Y'
):
    """
    Compute migration index between spatial clusters for BCR clones.

    This metric quantifies the degree of clonal overlap between two clusters
    in a given sample and chain, weighted by Shannon entropy of clone distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing clone, chain, cluster, and coordinate info.
    sample_col : str, default='sample'
        Column name for sample identifier.
    clone_col : str, default='clone'
        Column name for clone identifier.
    chain_col : str, default='Cgene'
        Column name for BCR chain.
    cluster_col : str, default='spatial_cluster'
        Column name for spatial cluster.
    x_col : str, default='X'
        Column name for x-coordinate.
    y_col : str, default='Y'
        Column name for y-coordinate.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns:
        ['sample', 'chain', 'cluster_source', 'cluster_target', 'BCR_cross']
        representing migration index between cluster pairs.
    """
    loc_cols = [x_col, y_col]
    unique_xy = df[[sample_col, clone_col, x_col, y_col, cluster_col, chain_col]].drop_duplicates()
    
    groups = [sample_col, chain_col, cluster_col]
    freq = unique_xy.groupby(groups)[clone_col].value_counts(normalize=True).reset_index(name='freq')
    count = unique_xy.groupby(groups)[clone_col].value_counts().reset_index(name='count')
    data = pd.merge(freq, count, on=groups+[clone_col])

    samples = data[sample_col].unique()
    chains = data[chain_col].unique()
    clusters = data[cluster_col].unique()

    clu_coms = list(itertools.combinations(clusters, 2))
    overlap_entropy = {}
    for sample in samples:
        sample_df = data[data[sample_col] == sample]
        for chain in chains:
            chain_df = sample_df[sample_df[chain_col] == chain]
            for clu1, clu2 in clu_coms:
                clu1_clones = set(chain_df.loc[chain_df[cluster_col] == clu1, clone_col])
                clu2_clones = set(chain_df.loc[chain_df[cluster_col] == clu2, clone_col])
                shared_clones = clu1_clones & clu2_clones
                if len(shared_clones) == 0:
                    continue
                tmp = chain_df[(chain_df[clone_col].isin(shared_clones)) &(chain_df[cluster_col].isin([clu1, clu2]))]
                tmp = tmp[['clone', cluster_col, 'count']].pivot_table(index='clone',columns=cluster_col,values='count').fillna(0)

                tmp = tmp.div(tmp.sum(axis=1), axis=0)
                entropy = -np.sum(tmp * np.log2(tmp + 1e-10), axis=1)
                entropy_dict = entropy.to_dict()
                overlap_entropy[(sample, chain, clu1, clu2)] = entropy_dict
                
    migration_dict = {}
    for key, entropy_dict in overlap_entropy.items():
        sample, chain, clu1, clu2 = key
        tmp = data[(data[sample_col] == sample) & (data[chain_col] == chain) & (data[cluster_col].isin([clu1, clu2]))]
        tmp = tmp[tmp[clone_col].isin(entropy_dict.keys())].copy()
        tmp['entropy'] = tmp[clone_col].map(entropy_dict)
        tmp1 = tmp[tmp[cluster_col] == clu1]
        tmp2 = tmp[tmp[cluster_col] == clu2]
        migration_dict[(sample, chain, clu1, clu2)] = np.sum(tmp1['freq'] * tmp1['entropy'])
        migration_dict[(sample, chain, clu2, clu1)] = np.sum(tmp2['freq'] * tmp2['entropy'])
    result = pd.DataFrame.from_dict(migration_dict, orient='index', columns=['BCR_cross'])

    result['sample'] = [x[0] for x in result.index]
    result['chain'] = [x[1] for x in result.index]
    result['cluster_source'] = [x[2] for x in result.index]
    result['cluster_target'] = [x[3] for x in result.index]

    result = result.reset_index(drop=True)
    return result
    



def renyi_entropy(probabilities, alpha_values=range(10)):
    """
    Calculate the Rényi entropy for a given probability distribution and alpha.
    
    Parameters:
    probabilities (list or numpy array): Probability distribution (should sum to 1).
    alpha (float): The order of the Rényi entropy.
    
    Returns:
    float: The calculated Rényi entropy.
    """
    results = {}
    for alpha in alpha_values:
        if alpha == 1:
            entropy = -np.sum(probabilities * np.log(probabilities))
        else:
            entropy = 1 / (1 - alpha) * np.log(np.sum(probabilities**alpha))
        results[alpha] = [entropy]
    return pd.DataFrame.from_dict(results)


def shannon_entropy(p): 
    """
    Compute Shannon entropy of a probability vector.

    Parameters
    ----------
    p : array-like
        Probability vector.

    Returns
    -------
    float
        Shannon entropy value.
    """
    p = p[p > 0]  # Only consider non-zero probabilities
    H = -np.sum(p * np.log2(p))
    return H


def CPK(count): 
    """
    Compute Clonotypes per 1000 cells.

    Parameters
    ----------
    count : array-like
        Clone counts.

    Returns
    -------
    float
        CPK value.
    """
    return len(count)/sum(count) * 1000

    
def normalize_shannon_entropy(p): 
    """
    Compute normalized Shannon entropy (range 0-1).

    Parameters
    ----------
    p : array-like
        Probability vector.

    Returns
    -------
    float
        Normalized entropy value.
    """
    p = p[p > 0]  # Only consider non-zero probabilities
    H = -np.sum(p * np.log2(p)) / np.log2(len(p))
    return H

def gini_index(data): 
    """
    Compute Gini index of a distribution.

    Parameters
    ----------
    data : array-like
        Non-empty data vector.

    Returns
    -------
    float
        Gini index (0=perfect equality, 1=max inequality).
    """
    if len(data) == 0:
        raise ValueError("Input data cannot be empty.")
    sorted_data = np.sort(data)
    n = len(data)
    cumulative_sum = np.cumsum(sorted_data)
    gini_index = (n + 1 - 2 * np.sum(cumulative_sum) / np.sum(sorted_data)) / n
    return gini_index

def Clonality(p):
    """
    Compute clonality metric from Shannon entropy.

    Parameters
    ----------
    p : array-like
        Probability vector.

    Returns
    -------
    float
        Clonality in [0,1].
    """
    C = 1 - shannon_entropy(p) / np.log2(len(p))
    return C

def Clonal_family_diversification(df, sample_col='sample', cluster_col='BCR_familyID', cdr3_col='cdr3nt'):
    """
    Compute clonal family diversification (Gini index of clone sizes per sample).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing sample, cluster, and CDR3 columns.
    sample_col : str
        Column name for sample identifier.
    cluster_col : str
        Column name for BCR family cluster.
    cdr3_col : str
        Column name for CDR3 nucleotide sequence.

    Returns
    -------
    pandas.DataFrame
        DataFrame with sample-wise clonal family diversification index.
    """
    df = df[[sample_col, cluster_col, cdr3_col]].drop_duplicates()
    clone_counts = df.groupby([sample_col, cluster_col]).size().reset_index(name='count') 
    res_df = clone_counts.groupby([sample_col])['count'].apply(lambda x: gini_index(x)).reset_index(name='Clonal_family_diver_index')
    return res_df

def compute_index(function_name, p):
    """
    Wrapper to compute a predefined BCR diversity / entropy index.

    Parameters
    ----------
    function_name : str
        Name of the function to compute, e.g. "shannon_entropy", "Clonality".
    p : array-like or dataframe
        Input probability vector or dataframe as required by the function.

    Returns
    -------
    float or pandas.DataFrame
        Computed index value(s).

    Raises
    ------
    ValueError
        If function_name is not recognized.
    """
    # 使用 globals() 来根据输入的函数名称调用对应的函数 
    functions = {
        'shannon_entropy': shannon_entropy,
        'normalize_shannon_entropy': normalize_shannon_entropy,
        'Clonality': Clonality, 
        'renyi_entropy': renyi_entropy,
        'gini_index': gini_index,
        'CPK':CPK,
        'Clonal_family_diversification': Clonal_family_diversification
    }
    
    # 检查用户输入的函数是否在已定义的函数字典中
    if function_name in functions:
        if function_name=='renyi_entropy':
            return functions[function_name](p)
        else:
            return functions[function_name](p)
    else:
        raise ValueError(f"Function '{function_name}' not recognized.")




