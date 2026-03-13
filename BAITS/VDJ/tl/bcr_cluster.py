import multiprocessing
from functools import partial
import networkx as nx
import pandas as pd
from tqdm.contrib.concurrent import process_map

def cluster_group(cdr3_list, threshold=0.85):
    """
    Cluster a list of CDR3 nucleotide sequences based on sequence identity.

    Two sequences are clustered together if their pairwise identity
    is greater than or equal to the specified threshold.

    Parameters
    ----------
    cdr3_list : list of str
        List of unique CDR3 nucleotide sequences.
    threshold : float, default=0.85
        Minimum pairwise identity required to connect two sequences.

    Returns
    -------
    list of sets
        Each set contains sequences belonging to the same cluster.
    """
    cdr3_list = list(set(cdr3_list))
    L = len(cdr3_list[0])
    
    G = nx.Graph()
    G.add_nodes_from(cdr3_list)

    for i, s1 in enumerate(cdr3_list):
        for s2 in cdr3_list[i+1:]:
            mismatches = sum(c1 != c2 for c1, c2 in zip(s1,s2))
            identity = 1 - mismatches/L
            if identity >= threshold:
                G.add_edge(s1,s2)
    return list(nx.connected_components(G))
    
def process_group_with_neighbor_count(group, threshold=0.85, cdr3nt_col='cdr3nt'):
    """
    Process a grouped BCR dataframe to assign clusters and neighbor counts.

    For each group of sequences sharing the same Vgene, Jgene, and CDR3 length,
    sequences are clustered and the number of neighbors with a single nucleotide
    difference is counted for each sequence.

    Parameters
    ----------
    group : tuple
        Tuple of ((Vgene, Jgene, CDR3_nt_length), dataframe) for the group.
    threshold : float, default=0.85
        Minimum identity threshold used for clustering.
    cdr3nt_col : str, default="cdr3nt"
        Column containing CDR3 nucleotide sequences.

    Returns
    -------
    list of tuples
        Each tuple contains:
        (Vgene, Jgene, sequence, cluster_id, neighbor_count)
    """
    (v_fam, j_fam, cdr3_len), df_group = group
    if len(df_group) < 2:
        # 单序列的 cluster，neighbor count = 0
        return [(v_fam, j_fam, seq, f"{v_fam}_{j_fam}_{cdr3_len}_0", 0) for seq in df_group[cdr3nt_col]]
        
    cdr3_list = df_group[cdr3nt_col].unique().tolist()
    clusters = cluster_group(cdr3_list, threshold=threshold)
    
    result = []
    for cid, cluster in enumerate(clusters):
        cluster_id = f"{v_fam}_{j_fam}_{cdr3_len}_{cid}"
        cluster_list = list(cluster)
        
        for seq in cluster_list:
            neighbor_count = sum(
                sum(c1 != c2 for c1, c2 in zip(seq, other)) == 1
                for other in cluster_list if other != seq
            )
            result.append((v_fam, j_fam, seq, cluster_id, neighbor_count))
    
    return result

def cluster_bcrs(igh_df, threshold=0.85, sample_col=None, Vgene_col='Vgene', Jgene_col='Jgene', cdr3nt_col='cdr3nt', n_cpu=None):
    """
    Cluster BCR sequences across an entire dataset and compute neighbor degrees.

    This function groups sequences by Vgene, Jgene, and CDR3 length, applies
    `process_group_with_neighbor_count` in parallel, and returns the original
    dataframe with added columns for BCR family ID and neighbor degree.

    Parameters
    ----------
    igh_df : pandas.DataFrame
        DataFrame containing BCR sequences.
    threshold : float, default=0.85
        Sequence identity threshold for clustering.
    sample_col : str or None, optional
        Column for sample/library ID. Currently reserved for future use.
    Vgene_col : str, default="Vgene"
        Column containing V gene names.
    Jgene_col : str, default="Jgene"
        Column containing J gene names.
    cdr3nt_col : str, default="cdr3nt"
        Column containing CDR3 nucleotide sequences.
    n_cpu : int or None, optional
        Number of CPUs to use for parallel processing. Defaults to all available.

    Returns
    -------
    pandas.DataFrame
        Original dataframe augmented with:
        - "BCR_familyID": cluster ID assigned to each sequence
        - "Degree": number of neighbors differing by one nucleotide within cluster
    """
    igh_df['CDR3_nt_length'] = igh_df[cdr3nt_col].str.len()
    grouped = igh_df.groupby([Vgene_col, Jgene_col, 'CDR3_nt_length'])
    
    if n_cpu is None:
        n_cpu = multiprocessing.cpu_count()
    
    process_func = partial(process_group_with_neighbor_count, threshold=threshold, cdr3nt_col=cdr3nt_col)
    results = process_map(process_func, list(grouped), max_workers=n_cpu, chunksize=1, desc="Clustering groups")
    
    cluster_map = {}
    neighbor_map = {}
    for r in results:
        for v_fam, j_fam, seq, cid, neighbor_count in r:
            cluster_map[(v_fam, j_fam, seq)] = cid
            neighbor_map[(v_fam, j_fam, seq)] = neighbor_count
    
    cluster_df = pd.DataFrame(
        [(v, j, seq, cluster_map[(v,j,seq)], neighbor_map[(v,j,seq)])
         for (v,j,seq) in cluster_map],
        columns=[Vgene_col, Jgene_col, cdr3nt_col, 'BCR_familyID', 'Degree'] 
    )
    
    return pd.merge(igh_df, cluster_df, how='left', on=[Vgene_col, Jgene_col, cdr3nt_col])
