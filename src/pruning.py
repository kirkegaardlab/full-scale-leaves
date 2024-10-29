from collections import defaultdict

import numba as nb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
from tqdm import tqdm

from src import network_utils


def _plot_graph(nodes, edges, edge_widths, path_coordinates=None,
               force_color=None):
    print('Plotting...')

    lines = []
    plot_widths = []
    colors = []
    cmap = plt.get_cmap('jet')

    np.random.seed(0)
    for ei, e in enumerate(edges):
        c = cmap(np.random.random())

        if path_coordinates is None:
            x, y = nodes[[e[0], e[1]], 1], nodes[[e[0], e[1]], 0]
            lines.append(((x[0], y[0]), (x[1], y[1])))
            plot_widths.append(edge_widths[ei])
            colors.append(c)
        else:
            for x1, x2 in zip(path_coordinates[ei][1:],
                              path_coordinates[ei][:-1]):
                lines.append((x1[::-1], x2[::-1]))
                plot_widths.append(edge_widths[ei])
                colors.append(c)

    line_segments = LineCollection(
        lines, linewidths=plot_widths, linestyle='solid',
        colors=force_color if force_color is not None else colors
    )
    plt.gca().add_collection(line_segments)

    plt.ylim(nodes[:, 0].min(), nodes[:, 0].max())
    plt.xlim(nodes[:, 1].min(), nodes[:, 1].max())


@nb.njit
def _calc_node_weights(nodes, edges, edge_widths, radius):
    w = np.zeros(len(nodes))
    for ei, e in enumerate(edges):
        d = np.sqrt(
            (nodes[e[0], 0] - nodes[e[1], 0])**2
            + (nodes[e[0], 1] - nodes[e[1], 1])**2
        )
        w[e[0]] = max(w[e[0]], edge_widths[ei]**2 * d)
        w[e[1]] = max(w[e[1]], edge_widths[ei]**2 * d)
    return w


@nb.njit
def _uniform_prune(nodes, edges, node_weights, edge_widths, radius):
    nodes = nodes.copy()
    edges = edges.copy()

    node_weights = node_weights + 1e-7

    print('Merging...')
    r2 = radius**2
    # 0 = untouched, 1 = kept, -1 = deleted
    state = np.zeros(len(nodes), dtype=nb.int8)

    p = 0
    c0 = np.sum(state == 0)

    while (state == 0).any():
        c = np.sum(state == 0)
        if 1 - c / c0 > p:
            print(' ...', int(100 * (1 - c/c0)), '%')
            p += 0.1

        i = np.argmax(node_weights * (state == 0))
        state[i] = 1

        d2 = (nodes[i, 0] - nodes[:, 0])**2 + (nodes[i, 1] - nodes[:, 1])**2
        d2[i] = 1e9
        idx = np.where(d2 < r2)[0]
        for j in idx:
            state[j] = -1
            for ei in range(len(edges)):
                if edges[ei, 0] == j:
                    edges[ei, 0] = i
                if edges[ei, 1] == j:
                    edges[ei, 1] = i
    print(' ... 100 %')

    print('Removing self-loops...')
    mask = edges[:, 0] != edges[:, 1]
    edges = edges[mask, :]
    edge_widths = edge_widths[mask]

    print('Relabelling nodes...')
    old_idxs = np.arange(len(nodes))[state == 1]
    mapping = np.zeros(len(nodes), dtype=nb.int64) - 1
    new_nodes = np.zeros((np.sum(state == 1), 2), dtype=nb.int64)
    count = 0
    for j in range(len(nodes)):
        if state[j] == 1:
            mapping[j] = count
            old_idxs[count] = j
            new_nodes[count, :] = nodes[j, :]
            count += 1

    for ei in range(len(edges)):
        edges[ei, 0] = mapping[edges[ei, 0]]
        edges[ei, 1] = mapping[edges[ei, 1]]


    print('Removing duplicates...')
    for i in range(len(edges)):
        if edges[i, 0] > edges[i, 1]:
            edges[i, 0], edges[i, 1] = edges[i, 1], edges[i, 0]
    idxs = np.argsort((1 + len(new_nodes)) * edges[:, 0] + edges[:, 1])
    edges = edges[idxs, :]
    edge_widths = edge_widths[idxs]
    
    keep = np.zeros(len(edges), dtype=nb.bool_)
    until = np.zeros(len(edges), dtype=nb.int64)
    i = 0
    while i < len(edges):
        keep[i] = True
        j = i + 1
        while j < len(edges):
            if edges[i, 1] != edges[j, 1] or edges[i, 0] != edges[j, 0]:
                break
            j += 1
        until[i] = j
        i = j

    n_edges = np.sum(keep)
    new_edges = np.zeros((n_edges, 2), dtype=nb.int64)
    new_edge_widths = np.ones(n_edges)
    j = 0
    for i in range(len(edges)):
        if keep[i]:
            new_edges[j, :] = edges[i, :]
            new_edge_widths[j] = np.max(edge_widths[i:until[i]])
            j += 1

    return new_nodes, new_edges, new_edge_widths, old_idxs


def _make_graph(edges, nodes, widths, exp=1):
    G = nx.Graph()
    for i in range(len(edges)):
        dx = nodes[edges[i, 0], 0] - nodes[edges[i, 1], 0]
        dy = nodes[edges[i, 0], 1] - nodes[edges[i, 1], 1]
        d = np.sqrt(dx**2 + dy**2) / widths[i]**exp
        G.add_edge(*edges[i], eff_capacity=d)
    return G


def _calc_effective_edge_widths(idxs, old_nodes, old_edges, old_edge_widths,
                               new_nodes, new_edges, exp=1):
    new_edge_widths = np.zeros(len(new_edges))
    old_paths = [np.array([]) for _ in range(len(new_edges))]

    G = _make_graph(old_edges, old_nodes, old_edge_widths, exp=exp)

    for i in tqdm(range(len(new_edges))):
        u, v = new_edges[i]
        try:
            d_path = nx.dijkstra_path_length(
                G, idxs[u], idxs[v], weight='eff_capacity'
            )
        except nx.exception.NetworkXNoPath:
            new_edge_widths[i] = 0
            continue
        d = np.sqrt(
            (new_nodes[u, 0] - new_nodes[v, 0])**2
            + (new_nodes[u, 1] - new_nodes[v, 1])**2
        )
        new_edge_widths[i] = (d / d_path)**(1 / exp)

        path = nx.dijkstra_path(G, idxs[u], idxs[v], weight='eff_capacity')
        old_paths[i] = np.array([old_nodes[x] for x in path])

    return new_edge_widths, old_paths


def _remove_small_weights(edges, edge_widths, path_coordinates,
                         remove_percentile):
    m = np.percentile(edge_widths[edge_widths > 1e-7], remove_percentile)
    mask = edge_widths > m
    path_coordinates = [
        path_coordinates[i] for i in range(len(mask)) if mask[i]
    ] 
    return edges[mask], edge_widths[mask], path_coordinates


def _find_add_missing_edges(old_idxs, old_nodes, old_edges, old_edge_widths,
                           new_nodes, new_edges, new_edge_widths, cutoff=0.75):
    G_old = _make_graph(old_edges, old_nodes, old_edge_widths)
    G_new = _make_graph(new_edges, new_nodes, new_edge_widths)

    node_to_edges = [list() for _ in range(len(new_nodes))]
    for e in new_edges:
        node_to_edges[e[0]].append(e[1])

    checked = defaultdict(lambda: False)
    to_be_added = []
    max_seen = 0
    for i in tqdm(range(len(new_nodes))):
        for j in node_to_edges[i]:
            for k in node_to_edges[j]:
                if i == k or checked[(i, k)]:
                    continue
                checked[(i, k)] = True

                try:
                    d_old = nx.dijkstra_path_length(
                        G_old, old_idxs[i], old_idxs[k], weight='eff_capacity'
                    )
                    d_new = nx.dijkstra_path_length(
                        G_new, i, k, weight='eff_capacity'
                    )
                except nx.exception.NetworkXNoPath:
                    continue

                difference = (d_new - d_old) / d_old
                max_seen = max(difference, max_seen)
                if difference > cutoff:
                    to_be_added.append([i, k])

    to_be_added = np.array(to_be_added)

    n_added = len(to_be_added)
    if n_added != 0:
        new_edges = np.concatenate((new_edges, to_be_added), axis=0)
        new_edge_widths = np.concatenate(
            (new_edge_widths, np.ones(len(to_be_added)))
        )

    print(f' ... added {n_added} new edges to fix double jumps')
    
    return new_edges, new_edge_widths


def _prune(nodes, edges, edge_widths, radius, remove_percentile=1,
          double_jump_check=True):
    print('# PRUNE: calculating node weights')
    node_weights = _calc_node_weights(nodes, edges, edge_widths, radius)
    print('# PRUNE: running uniform prune')
    new_nodes, new_edges, new_edge_widths, old_idxs = _uniform_prune(
        nodes, edges, node_weights, edge_widths, radius
    )

    if double_jump_check:
        print('# PRUNE: tenatively recalculating new widths')
        new_edge_widths, _ = _calc_effective_edge_widths(
            old_idxs, nodes, edges, edge_widths, new_nodes, new_edges
        )

        print('# PRUNE: add missing edges')
        new_edges, new_edge_widths = _find_add_missing_edges(
            old_idxs, nodes, edges, edge_widths, new_nodes, new_edges,
            new_edge_widths
        )

    print('# PRUNE: recalculating new widths')
    new_edge_widths, path_coordinates = _calc_effective_edge_widths(
        old_idxs, nodes, edges, edge_widths, new_nodes, new_edges
    )

    print('# PRUNE: removing small weights')
    new_edges, new_edge_widths, path_coordinates = _remove_small_weights(
        new_edges, new_edge_widths, path_coordinates, remove_percentile
    )

    return new_nodes, new_edges, new_edge_widths, path_coordinates


@nb.njit
def _get_new_edge_lengths(nodes, edges):
    n_edges = len(edges)
    edge_lengths = np.empty(n_edges)

    for i in range(n_edges):
        edge = edges[i]
        incident_nodes = nodes[edge]

        edge_length = np.sqrt(
            ((incident_nodes[0] - incident_nodes[1])**2).sum()
        )
        edge_lengths[i] = edge_length

    return edge_lengths


def _get_new_degrees(nodes, edges):
    degrees = np.zeros(len(nodes))

    unique_nodes, counts = np.unique(edges, return_counts=True)

    degrees[unique_nodes] = counts

    return degrees


def get_pruned(prune_radius, nodes, degrees, edges, edge_lengths, edge_widths,
               edge_areas):
    # Pruned network
    nodes, edges, edge_widths, _ = _prune(
        nodes, edges, edge_widths, prune_radius
    )

    edge_lengths = _get_new_edge_lengths(nodes, edges)
    edge_areas = network_utils.get_edge_areas(edge_widths)
    degrees = _get_new_degrees(nodes, edges)

    return nodes, degrees, edges, edge_lengths, edge_widths, edge_areas
