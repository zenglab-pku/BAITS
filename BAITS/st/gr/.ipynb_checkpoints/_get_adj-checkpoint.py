def get_adj(adata, mode = 'neighbour', k_neighbors = 7, min_distance = 150, self_loop = True):
    """
    Calculate adjacency matrix for ST data.
    
    Parameters
    ----------
    mode : str ['neighbour','distance'] (default: 'neighbour')
        The way to define neighbourhood. 
        If `mode='neighbour'`: Calculate adjacency matrix with specified number of nearest neighbors;
        If `mode='distance'`: Calculate adjacency matrix with neighbors within the specified distance.
    k_neighbors : int (default: 7)
        For `mode = 'neighbour'`, set the number of nearest neighbors if `mode='neighbour'`.
    min_distance : int (default: 150)
        For `mode = 'distance'`, set the distance of nearest neighbors if `mode='distance'`.
    self_loop : bool (default: True)
        Whether to add selfloop to the adjacency matrix.
        
    adj : matrix of shape (n_samples, n_samples)
        Adjacency matrix where adj[i, j] is assigned the weight of edge that connects i to j.
    """
    spatial = adata.obsm["spatial"]   
    if mode == 'distance':
        assert min_distance is not None,"Please set `min_diatance` for `get_adj()`"
        adj = metrics.pairwise_distances(spatial, metric='euclidean')
        adj[adj > min_distance] = 0
        if self_loop:
            adj += np.eye(adj.shape[0])  
        adj = np.int64(adj>0)
        return adj
    
    elif mode == 'neighbour':
        assert k_neighbors is not None,"Please set `k_neighbors` for `get_adj()`"
        adj = kneighbors_graph(spatial, n_neighbors = k_neighbors, include_self = self_loop)
        return adj


    
    