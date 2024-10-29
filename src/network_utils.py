import networkx as nx
import numba as nb
import numpy as np


def set_degrees(nodes, edges):
    """Return array of degrees. Assumes unique edges."""
    n_nodes = len(nodes)

    node_inds, counts = np.unique(edges.flatten(), return_counts=True)
    degrees = np.zeros(n_nodes, dtype=int)
    degrees[node_inds] = counts

    return degrees


@nb.njit
def get_edge_areas(edge_widths):
    edge_areas = np.pi * (edge_widths / 2)**2
    return edge_areas


@nb.njit
def _edge_in_edges(edge, edges):
    for i in range(len(edges)):
        if ((edges[i,0] == edge[0] and edges[i,1] == edge[1])
            or (edges[i,0] == edge[1] and edges[i,1] == edge[0])):
            return True

    return False


@nb.njit
def _get_incident_edges(node_idx, edges):
    cond = (edges[:,0] == node_idx) ^ (edges[:,1] == node_idx)
    incident_edges = edges[cond]
    incident_edge_inds = np.argwhere(cond)[:,0]
    return incident_edges, incident_edge_inds


@nb.njit
def _isolate_two_degree_nodes(degrees, edges, edge_lengths, edge_widths,
                             edge_areas):
    """Isolate two-degree nodes, and connect their neighboring nodes."""
    while True:
        two_degree_nodes = (degrees == 2)
        if not np.any(two_degree_nodes):
            break

        mid_node = np.argmax(two_degree_nodes)

        # Shape = (2,2), (2,)
        incident_edges, incident_edge_inds = _get_incident_edges(
            mid_node, edges
        )

        if incident_edges.shape != (2,2):

            print(incident_edges.shape)
            raise AssertionError(f'Incorrect shape: {incident_edges.shape}')

        # Shape = (2,)
        cond = (incident_edges != mid_node).flatten()
        neighboring_nodes = incident_edges.flatten()[cond]

        edge_idx0 = incident_edge_inds[0]
        edge_idx1 = incident_edge_inds[1]
        node_idx0 = neighboring_nodes[0]
        node_idx1 = neighboring_nodes[1]

        ## Connect if new edge does not already exist, else isolate
        if _edge_in_edges(neighboring_nodes, edges):
            edges[edge_idx0] = mid_node
            degrees[node_idx0] -= 1
            degrees[node_idx1] -= 1
            edge_lengths[edge_idx0] = 0
            edge_widths[edge_idx0] = 0
            edge_areas[edge_idx0] = 0

        else:
            edges[edge_idx0] = neighboring_nodes

            # Update metrics
            edge_lengths[edge_idx0] = (
                edge_lengths[edge_idx0] + edge_lengths[edge_idx1]
            )
            # Based on effective conductivity of new edge
            new_edge_width = (
                1 / ((edge_lengths[edge_idx1] / edge_widths[edge_idx1]**4
                        + edge_lengths[edge_idx0] / edge_widths[edge_idx0]**4)
                    / edge_lengths[edge_idx0])
            )**(1/4)

            edge_widths[edge_idx0] = new_edge_width
            edge_areas[edge_idx0] = get_edge_areas(
                edge_widths[edge_idx0]
            )

        ## Isolate
        edges[edge_idx1] = mid_node

        # Update metrics
        degrees[mid_node] = 0
        edge_lengths[edge_idx1] = 0
        edge_widths[edge_idx1] = 0
        edge_areas[edge_idx1] = 0

    return degrees, edges, edge_lengths, edge_widths, edge_areas


def _get_nodes_to_isolate(G):
    """Make list of nodes not in largest contiguous network."""
    components = nx.connected_components(G)

    components, component_lengths = _get_component_lengths(components)
    largest_component_idx = np.argmax(component_lengths)

    nodes_to_isolate = []

    for i, component in enumerate(components):
        if i != largest_component_idx:
            nodes_to_isolate += list(component)
        # Keep largest connected component
        else:
            continue

    return nodes_to_isolate


def _mask(nodes, edges, degrees):
    G = _make_graph(nodes, edges)

    nodes_to_isolate = _get_nodes_to_isolate(G)

    for node_idx in nodes_to_isolate:
        node_inds = np.argwhere(edges == node_idx)[:,0]

        for row_idx in node_inds:
            edge = edges[row_idx]

            # Do not change isolates
            if edge[0] != edge[1]:
                degrees[edge] -= 1
                edges[row_idx] = node_idx

    return edges, degrees


def _remove_zero_edges(edges, edge_lengths, edge_widths, edge_areas):
    mask = edges[:,0] != edges[:,1]
    edges = edges[mask]
    edge_lengths = edge_lengths[mask]
    edge_widths = edge_widths[mask]
    edge_areas = edge_areas[mask]

    return edges, edge_lengths, edge_widths, edge_areas


def _remove_zero_nodes(nodes, degrees, edges):
    # Adjust indices
    update_array = np.zeros_like(edges)

    for i, degree in enumerate(degrees):
        if degree == 0:
            update_array[edges >= i] += 1
        else:
            continue

    edges = edges - update_array

    # Remove nodes of degree 0
    mask = (degrees != 0)
    nodes = nodes[mask]
    degrees = degrees[mask]

    return nodes, degrees, edges


def clean(nodes, degrees, edges, edge_lengths, edge_widths, edge_areas):
    '''Prepares network for saving.
    
    Isolates two-degree nodes, makes one connected graph, and removes
    zero-degree nodes and isolated edges.
    '''
    print('Merging two-degree nodes...')
    (degrees, edges, edge_lengths, edge_widths,
     edge_areas) = _isolate_two_degree_nodes(
        degrees, edges, edge_lengths, edge_widths, edge_areas
    )

    print('Reducing to one connected graph...')
    edges, degrees = _mask(nodes, edges, degrees)

    print('Removing zero-degree nodes and isolated edges...')
    # Remove edges corresponding to isolated nodes,
    # as well as their lengths, widths, and areas from arrays
    edges, edge_lengths, edge_widths, edge_areas = _remove_zero_edges(
        edges, edge_lengths, edge_widths, edge_areas
    )

    # Remove nodes of degree 0, and change edges accordingly
    nodes, degrees, edges = _remove_zero_nodes(nodes, degrees, edges)

    return nodes, degrees, edges, edge_lengths, edge_widths, edge_areas


def _get_component_lengths(components_gen):
    components = []
    component_lengths = []
    for component in components_gen:
        components.append(component)
        component_lengths.append(len(component))

    return components, component_lengths


def _make_graph(nodes, edges):
    G = nx.Graph()

    n_nodes = len(nodes)
    nodes = np.arange(n_nodes)

    G.add_nodes_from(nodes)
    G.add_edges_from(list(edges))

    return G


def _normalize(model_widths, edge_widths_mean):
    model_widths = model_widths / (model_widths.mean() / edge_widths_mean)
    return model_widths


def get_model_widths(final_conductivities, edge_widths_mean):
    model_widths = 2 * final_conductivities**(1/4)
    model_widths = _normalize(model_widths, edge_widths_mean)
    return model_widths


def get_chi_squared(edge_widths, model_widths):
    chi_squared = (
        (edge_widths - model_widths)**2 / edge_widths
    ).sum()
    return chi_squared


def get_loss(edge_widths, model_widths):
    chi_squared = get_chi_squared(edge_widths, model_widths)
    loss = chi_squared / edge_widths.mean() / len(edge_widths)

    return loss
