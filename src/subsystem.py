import numpy as np


def _get_cluster_inds(min_dist_inds, i):
    cluster_inds = np.argwhere(min_dist_inds == i)[:,0]
    return cluster_inds


def _get_cluster(nodes, min_dist_inds, i):
    sub_nodes = _get_cluster_inds(min_dist_inds, i)
    cluster = nodes[sub_nodes]
    return cluster


def _get_new_centers(nodes, min_dist_inds, n):
    new_centers = np.empty(shape=(n, 2))

    for i in range(n):
        cluster = _get_cluster(nodes, min_dist_inds, i)
        new_center = cluster.mean(axis=0)
        new_centers[i] = new_center
    
    return new_centers[:,None,:]


def _get_min_dist_inds(nodes, n):
    n_iter = 1000
    
    np.random.seed(0)

    # Initial centers
    center_inds = np.random.choice(
        np.arange(len(nodes)), size=n, replace=False
    )

    centers = nodes[center_inds][:,None,:]
    for _ in range(n_iter):
        dists = np.sqrt(np.square(nodes - centers).sum(axis=2))
        # Assign each node to closest center node
        min_dist_inds = np.argmin(dists, axis=0)
        # Calculate center of mass
        centers = _get_new_centers(nodes, min_dist_inds, n)

    return min_dist_inds


def get_clusters(nodes, n_clusters):
    min_dist_inds = _get_min_dist_inds(nodes, n_clusters)

    clusters = [] 

    for i in range(n_clusters):
        cluster_inds = _get_cluster_inds(min_dist_inds, i)
        clusters.append(cluster_inds)

    return clusters


def _get_subgraph(edges, incidence, sub_nodes):
    border_nodes_mask = np.zeros(len(sub_nodes), dtype=bool)
    n_edges = len(edges)
    sub_edges_mask = np.zeros(n_edges, dtype=bool)
    edge_inds = np.arange(n_edges)
    
    J = np.zeros(incidence.shape)

    for i in range(n_edges):
        e0, e1 = edges[i]
        # Edge crosses the boundary
        if e0 in sub_nodes and e1 not in sub_nodes:
            sub_node_idx = np.argwhere(sub_nodes == e0)[0]
            border_nodes_mask[sub_node_idx] = True
            J[e0, i] = incidence[e0, i]
        # Edge crosses the boundary
        elif e1 in sub_nodes and e0 not in sub_nodes:
            sub_node_idx = np.argwhere(sub_nodes == e1)[0]
            border_nodes_mask[sub_node_idx] = True
            J[e1, i] = incidence[e1, i]
        # Edge entirely within the boundary
        elif e0 in sub_nodes and e1 in sub_nodes:
            sub_edges_mask[i] = True

    border_nodes = sub_nodes[border_nodes_mask]
    sub_edges = edge_inds[sub_edges_mask]

    subgraph = {'sub_nodes': sub_nodes,
                'border_nodes': border_nodes,
                'sub_edges': sub_edges,
                'J': J}

    return subgraph


def _get_M(sub_nodes, n_sub_nodes, n_nodes):
    M = np.zeros((n_sub_nodes, n_nodes))
    inds = np.arange(n_sub_nodes)
    M[inds, sub_nodes] = 1
    
    return M


def _get_I_sub(incidence, sub_nodes, sub_edges):
    # incidence is a sparse matrix
    I_sub = incidence[sub_nodes][:,sub_edges]
    I_sub = I_sub.toarray()
    return I_sub


def _get_Z(M, J, B):
    MJ = M @ J
    Z = M - MJ @ B
    return Z


def get_submatrices(nodes, edges, incidence, sub_nodes, B):
    subgraph = _get_subgraph(edges, incidence, sub_nodes)

    sub_edges = subgraph['sub_edges']

    I_sub = _get_I_sub(incidence, sub_nodes, sub_edges) 
    I_sub_t = I_sub.T
    
    n_nodes, n_sub_nodes = len(nodes), len(sub_nodes)

    M = _get_M(sub_nodes, n_sub_nodes, n_nodes)
    Z = _get_Z(M, subgraph['J'], B)

    submatrices = {'Z': Z,
                   'I_sub': I_sub,
                   'I_sub_t': I_sub_t,
                   'border_nodes': subgraph['border_nodes'],
                   'sub_edges': sub_edges}

    return submatrices
