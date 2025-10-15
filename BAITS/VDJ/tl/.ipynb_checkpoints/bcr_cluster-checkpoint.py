import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from Levenshtein import distance as levenshtein_distance
from itertools import combinations
from collections import defaultdict

def _cluster_cdr3nt(cdr3_sequences, threshold=0.85):
    # Calculate pairwise distances
    n = len(cdr3_sequences)
    dist_matrix = np.zeros((n, n))
    
    for i, j in combinations(range(n), 2):
        dist_matrix[i, j] = levenshtein_distance(cdr3_sequences[i], cdr3_sequences[j])
        dist_matrix[j, i] = dist_matrix[i, j]
    
    condensed_dist = squareform(dist_matrix)
    Z = linkage(condensed_dist, method='average')
    max_dist = max(len(seq) for seq in cdr3_sequences) * (1 - threshold)
    clusters = fcluster(Z, max_dist, criterion='distance')
    
    return {seq: cluster for seq, cluster in zip(cdr3_sequences, clusters)}


def _build_bcr_family(bcr_data, cdr3nt_col, threshold=0.85):
    
    tmp_dd_ss = bcr_data.drop_duplicates(subset=[cdr3nt_col])
    
    if tmp_dd_ss.empty:
        return None
    
    comp_cdr3_nt = tmp_dd_ss[cdr3nt_col].tolist()

    if len(comp_cdr3_nt)==1:
        return  {comp_cdr3_nt[0]: 0 }
    
    family_groups = _cluster_cdr3nt(comp_cdr3_nt, threshold)
    return family_groups
    

def build_bcr_family(bcr_data, Vgene_col, Jgene_col, cdr3nt_col, Cgene_key, Cgene='IGH', threshold=0.85):

    data = bcr_data[[Vgene_col, Jgene_col, cdr3nt_col, Cgene_key]].copy()
    
    # Filter by c_gene
    data = data[data['Cgene'].str.contains(Cgene)].copy()
    
    data = data.drop_duplicates(subset=[Vgene_col, Jgene_col, cdr3nt_col])
    data['CDR3_length'] = (data[cdr3nt_col].str.len()/3).astype(int)
    data['cloneGroup'] = data[Vgene_col] + "_" + data[Jgene_col] + "_" + data[cdr3nt_col].astype(str) 
    data['family'] = ''
    for group in set(data['cloneGroup']): 
        tmp_data = data[data['cloneGroup'] == group].copy()
        result = _build_bcr_family(tmp_data, cdr3nt_col, threshold=0.85)

        if not result:
            continue

        for seq, family_num_idx in result.items():
            family_name = group + '_family_' + str(family_num_idx) 
            mask = (data['cloneGroup'] == group) & (data[cdr3nt_col]==(seq) )  
            data.loc[mask, 'family'] = family_name
            
    data['family_id'] = [ Cgene + '_family_' +x for x in pd.Categorical(data['family']).codes.astype(str)]
    bcr_data = pd.merge(bcr_data, data, on = [Vgene_col, Jgene_col, Cgene_key, cdr3nt_col], how='left')
    
    return bcr_data
    