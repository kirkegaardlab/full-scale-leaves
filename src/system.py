import numpy as np
from scipy import sparse, spatial

from src import dataframes, network_utils, structure


def _check_dimensions(nx, ny):
    if nx <= 0:
        raise AssertionError('nx must be greater than 0.')
    elif ny%2 == 0 or ny <= 0:
        raise AssertionError('ny must be an odd, positive number.')


class _ArtificialNodes:
    def __init__(self, nx, ny, edge_length) -> None:
        self._nx = nx
        self._ny = ny
        self._edge_length = edge_length

        self.nodes = self._make_nodes()
        self.n_nodes = len(self.nodes)

    def _make_nodes(self):
        _check_dimensions(self._nx, self._ny)

        # Vertical distance between nodes
        y_dist = self._edge_length * np.sin(np.pi / 3)
        # x coords along leaf stem
        x_max = self._nx * self._edge_length
        main_xs = np.arange(0, x_max, self._edge_length)

        # All coords
        x_coords = list(main_xs)
        y_coords = list(np.zeros(self._nx))

        # No. of rows in positive y direction,
        # since rows are added pairwise
        n_pos_rows = int((self._ny - 1) / 2)

        for i in range(n_pos_rows):
            nrows_from_center = i + 1
            n_nodes = int(self._nx - nrows_from_center)

            xs = list(
                (np.arange(n_nodes) + nrows_from_center / 2)
                * self._edge_length
            )

            # Positive ys
            ys = (i+1) * y_dist * np.ones(n_nodes)

            # Append coords for positive row
            x_coords += xs
            y_coords += list(ys)

            # Append coords for negative row
            x_coords += xs
            y_coords += list(-ys)

        coords = np.array([x_coords, y_coords]).T

        return coords


class _Incidence:
    def __init__(self, nodes, n_nodes):
        self._nodes = nodes
        self._n_nodes = n_nodes

        self._connectivity = self._get_connectivity_matrix()

        self.matrix, self.edges, self.edge_lengths = (
            self._get_incidence_matrix()
        )

        self.n_edges = len(self.edge_lengths)

    def _get_neighbors(self):
        """
        Finds neighboring nodes.

                Parameters:
                        nodes (ndarray): All point coordinates.

                Returns:
                        neighbors (ndarray): Indices of neighboring nodes.
                        ranges (ndarray): Index ranges for 'neighbors'.
                            For point k, neighbors[ranges[k]:ranges[k+1]] are
                            the neighboring nodes.
        """
        # Triangulate nodes
        triangulation = spatial.Delaunay(self._nodes)

        # Find neighboring nodes
        ranges, neighbors = triangulation.vertex_neighbor_vertices

        return neighbors, ranges

    def _get_connectivity_matrix(self):
        neighbors, ranges = self._get_neighbors()

        connectivity = np.zeros((self._n_nodes, self._n_nodes))

        # Loop over nodes
        for point_idx in range(len(ranges) - 1):
            # Neighbors of point point_idx
            neighbor_inds = neighbors[ranges[point_idx]:ranges[point_idx+1]]

            # Set connectivity
            connectivity[point_idx, neighbor_inds] = 1

        return connectivity

    def _get_incidence_matrix(self):
        # Defines direction of connectivity
        one_way_connectivity = np.tril(self._connectivity)

        n_edges = int(np.sum(one_way_connectivity))

        edges = []
        # incidence = np.zeros((self._n_nodes, n_edges))
        incidence = sparse.lil_matrix((self._n_nodes, n_edges))
        edge_lengths = np.zeros(n_edges)

        # Edge index
        e = 0
        for v1 in range(self._n_nodes):
            for v2 in range(self._n_nodes):
                if one_way_connectivity[v1, v2]:
                    incidence[v1, e] = 1
                    incidence[v2, e] = -1
                    # Length of edge e
                    edge_lengths[e] = np.linalg.norm(
                        self._nodes[v1] - self._nodes[v2]
                    )

                    # Add vertex indices for each edge
                    edges.append((v1, v2))
                    # Index of next edge
                    e += 1

        edges = np.vstack(edges)
        incidence = incidence.tocsr()

        return incidence, edges, edge_lengths


class _ArtificialSystem:
    def __init__(self, nodes, n_nodes, edges, n_edges, incidence_matrix,
                 edge_lengths, edge_widths_scalar):
        self.nodes = nodes
        self.n_nodes = n_nodes
        self.edges = edges
        self.n_edges = n_edges
        self.degrees = network_utils.set_degrees(nodes, edges)
        self.incidence_matrix = incidence_matrix
        self.edge_lengths = edge_lengths
        self.edge_widths = np.ones(n_edges) * edge_widths_scalar
        self.is_added = np.zeros(n_edges, dtype=bool)

        # Place source node at top center
        mean_x = np.mean(nodes[:,0])
        top_y = np.max(nodes[:,1])
        self.source_idx = np.argmin(
            np.sum(np.abs(nodes - (mean_x, top_y)), axis=1)
        )


def _get_artificial_system(params):
    nx = params['nx']
    ny = params['ny']
    edge_length = params['edge_length']
    edge_widths = params['edge_width']

    artificial_nodes = _ArtificialNodes(nx, ny, edge_length)
    nodes = artificial_nodes.nodes
    n_nodes = artificial_nodes.n_nodes

    incidence = _Incidence(nodes, n_nodes)
    incidence_matrix = incidence.matrix
    edge_lengths = incidence.edge_lengths
    n_edges = incidence.n_edges
    edges = incidence.edges

    artificial_system = _ArtificialSystem(
        nodes, n_nodes, edges, n_edges, incidence_matrix, edge_lengths,
        edge_widths
    )

    return artificial_system


class _InputSystem:
    def __init__(self, params):
        im_name = params['im_name']
        source_pos = params['source_pos']
        n_edges_added = params['n_edges_added']

        input_dir = structure.Directories().added
        self._path_to_input_file = input_dir / im_name

        _input_network_df = dataframes.DFs(
            self._path_to_input_file, n_edges_added=n_edges_added
        )
        self.nodes = _input_network_df.get_nodes()
        self.degrees = _input_network_df.get_degrees()
        
        edges_dict = _input_network_df.get_edges_dict()
        self.edges = edges_dict['edges']
        self.edge_lengths = edges_dict['edge_lengths']
        self.edge_widths = edges_dict['edge_widths']
        self.is_added = edges_dict['is_added']

        # To prevent division by 0
        self.edge_lengths += 1e-10

        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)

        self.incidence_matrix = self._get_incidence_matrix()
        self._source_pos = source_pos
        self.source_idx = self._get_source_idx(self.nodes, self._source_pos)

    def _get_incidence_matrix(self):
        incidence_matrix = sparse.lil_matrix((self.n_nodes, self.n_edges))

        node_inds_1 = self.edges[:,0]
        node_inds_2 = self.edges[:,1]

        for j in range(self.n_edges):
            incidence_matrix[node_inds_1[j], j] = 1
            incidence_matrix[node_inds_2[j], j] = -1

        incidence_matrix = incidence_matrix.tocsr()

        return incidence_matrix

    @staticmethod
    def _get_source_idx(nodes, source_pos):
        match source_pos:
            case 'Top':
                source_idx = np.argmin(nodes[:,0])
            case 'Bottom':
                source_idx = np.argmax(nodes[:,0])
            case 'Left':
                source_idx = np.argmin(nodes[:,1])
            case 'Right':
                source_idx = np.argmax(nodes[:,1])
            case _:
                raise AssertionError('No valid input for source_pos parameter!')

        return source_idx


def _get_input_system(params):
    input_system = _InputSystem(params)
    return input_system


def get(params):
    im_name = params['im_name']

    if im_name == 'artificial':
        system = _get_artificial_system(params)
    # Assumes 'im_name' is a valid filename
    else:
        system = _get_input_system(params)

    return system
