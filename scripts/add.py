import alphashape
import numba as nb
import numpy as np
from scipy import sparse
import shapely

from src import dataframes, network_utils, param_parser, structure


_NEAREST_NODES_RADIUS = 200
_ADDED_EDGE_WIDTH = 0.5


def _edge_is_within_alphashape(alpha_polygon, new_edge):
    line = shapely.LineString(new_edge)
    return line.within(alpha_polygon)


def _get_alpha_polygon(nodes, alpha):
    alpha_shape = alphashape.alphashape(nodes, alpha)

    # Keep only the largest polygon
    if isinstance(alpha_shape, shapely.MultiPolygon):
        pp = None
        mm = 0
        for i, p in enumerate(alpha_shape.geoms):
            if p.area > mm:
                mm = p.area
                pp = p
        alpha_shape = pp

    alpha_polygon = shapely.geometry.Polygon(alpha_shape.exterior)

    return alpha_polygon


def _update_connectivities(undirected_connectivities, node0, node1):
    undirected_connectivities[node0, node1] = True
    undirected_connectivities[node1, node0] = True
    return undirected_connectivities


@nb.njit()
def _ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


@nb.njit()
def _intersect(A, B, C, D):
    '''Return true if line segments AB and CD intersect'''
    return (_ccw(A, C, D) ^ _ccw(B, C, D)) and (_ccw(A, B, C) ^ _ccw(A, B, D))


@nb.njit()
def _edge_crosses_data_edge(nodes, edge, data_edges):
    for data_edge in data_edges:
        if _intersect(nodes[edge[0]],
                      nodes[edge[1]],
                      nodes[data_edge[0]],
                      nodes[data_edge[1]]):
            return True
    return False


def _get_edge_lengths(nodes, edges):
    edge_lengths = np.linalg.norm(
        nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1
    )
    return edge_lengths


def _get_undirected_connectivities(edges_with_exterior, n_nodes):
    cs = sparse.lil_matrix((n_nodes, n_nodes), dtype=np.bool_)

    cs[edges_with_exterior[:, 0], edges_with_exterior[:, 1]] = True
    cs[edges_with_exterior[:, 1], edges_with_exterior[:, 0]] = True

    return cs


def _get_nearest_nodes_inds(node, data_nodes):
    dists = np.linalg.norm(data_nodes - node, axis=1)
    nearest_nodes_inds = np.where(dists < _NEAREST_NODES_RADIUS)[0]
    return nearest_nodes_inds


def _add_edges(data_nodes, n_nodes, undirected_connectivities,
               alpha_polygon, all_edges, n_edges_added):
    # Number of edges added
    n = 0
    # Number of tries to add an edge per node iteration
    n_tries = 0
    max_tries = 1000

    while True:
        n_tries += 1

        node_idx = np.random.randint(n_nodes)
        node = data_nodes[node_idx]

        nearest_nodes_inds = _get_nearest_nodes_inds(node, data_nodes)
        nearest_node_idx = np.random.choice(nearest_nodes_inds)

        # Skip if the edge already exists, or if the indices are the same
        if undirected_connectivities[node_idx, nearest_node_idx]:
            continue
        elif node_idx == nearest_node_idx:
            continue
        else:
            new_edge = np.array([node_idx, nearest_node_idx])

            if _edge_crosses_data_edge(data_nodes, new_edge, all_edges):
                continue
            elif not _edge_is_within_alphashape(
                alpha_polygon, data_nodes[new_edge]
            ):
                continue
            else:
                all_edges = np.vstack([all_edges, new_edge])
                undirected_connectivities = _update_connectivities(
                    undirected_connectivities, node_idx, nearest_node_idx
                )
                n += 1

                if n%100 == 0:
                    print(f'{n} / {n_edges_added}')

                if n == n_edges_added:
                    break

                # Reset
                n_tries = 0

        if n_tries == max_tries:
            raise AssertionError('Not enough edges could be added.')

    return all_edges, n


def _get_all_edge_widths(n_added_edges, data_edge_widths):
    added_edge_widths = _ADDED_EDGE_WIDTH * np.ones(n_added_edges)
    edge_widths = np.concatenate([data_edge_widths, added_edge_widths], axis=0)
    return edge_widths


def main():
    dirs = structure.Directories()
    params = param_parser.get_params()
    im_name = params['im_name']

    input_fname = dirs.reduced / im_name
    dfs = dataframes.DFs(input_fname, n_edges_added=0)
    data_nodes = dfs.get_nodes()
    degrees = dfs.get_degrees()

    edges_dict = dfs.get_edges_dict()
    data_edges = edges_dict['edges']
    data_edge_lengths = edges_dict['edge_lengths']
    data_edge_widths = edges_dict['edge_widths']

    sim_params = param_parser.get_params()
    n_edges_added = sim_params['n_edges_added']

    if n_edges_added != 0:
        n_nodes = len(data_nodes)

        undirected_connectivities = _get_undirected_connectivities(
            data_edges, n_nodes
        )

        alpha_polygon = _get_alpha_polygon(data_nodes, alpha=0.008)

        edges, n_edges_added = _add_edges(
            data_nodes, n_nodes, undirected_connectivities, alpha_polygon,
            data_edges, n_edges_added
        )

        # Update edge lengths and widths
        edge_lengths = _get_edge_lengths(data_nodes, edges)
        edge_widths = _get_all_edge_widths(n_edges_added, data_edge_widths)
    else:
        edges = data_edges
        edge_lengths = data_edge_lengths
        edge_widths = data_edge_widths

    print(f'Added {n_edges_added} edges.')

    edge_areas = network_utils.get_edge_areas(edge_widths)
    degrees = network_utils.set_degrees(data_nodes, edges)

    output_dir = dirs.added / im_name

    dataframes.save_dfs(
        data_nodes, degrees, edges, edge_lengths, edge_widths, edge_areas,
        output_dir, n_edges_added=n_edges_added
    )


if __name__ == '__main__':
    main()
