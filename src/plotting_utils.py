import pickle

from matplotlib.collections import LineCollection


def get_coord_list(nodes, edges):
    coords = nodes[edges]
    coord_list = list(coords)

    return coord_list


def plot_edge_widths(nodes, edges, edge_widths, is_added, scale, alpha, ax,
                     non_added_color='r'):
    conds = [is_added, ~is_added]
    colors = ['k', non_added_color]

    for i, cond in enumerate(conds):
        coord_list = get_coord_list(nodes, edges[cond])

        line_segments = LineCollection(
            coord_list, colors=colors[i], linewidths=scale * edge_widths[cond],
            linestyle='solid', alpha=alpha
        )

        ax.add_collection(line_segments)

    return ax


def flip_coords(coords):
    return coords[:,[1,0]]


def get_output_data(filename):
    with open(filename, 'rb') as f:
        output_data = pickle.load(f)

    return output_data
