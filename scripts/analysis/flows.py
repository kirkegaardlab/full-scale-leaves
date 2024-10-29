import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage.io import imread

from src import (dataframes, network_utils, param_parser, paths, plotting_utils,
                 structure)


def _plot_edges_nodes(ax, im, edges, edge_widths, is_added, nodes, 
                      border_node_coords):
    # Scales input and model widths in plots
    lw_scale = 0.2
    nodes_s = 0.3
    alpha = 0.55
    border_nodes_c = 'k'

    # Show image        
    ax.imshow(im)

    # Edges
    ax = plotting_utils.plot_edge_widths(
        nodes, edges, edge_widths, is_added, lw_scale, alpha, ax
    )

    # Nodes
    ax.scatter(
        nodes[:, 0], nodes[:, 1], s=nodes_s, color='k', edgecolor='None',
        zorder=10
    )

    # Border nodes
    if len(border_node_coords) != 0:
        ax.scatter(
            border_node_coords[:,0], border_node_coords[:,1], s=nodes_s,
            edgecolor='None', color=border_nodes_c, zorder=10
        )

    return ax


def _plot_arrows(ax, model_widths, arrow_positions, arrow_directions):
    width_scale = 0.0003
    headwidth_scale = 5e3

    arrowwidth = width_scale * model_widths.mean()
    headval = headwidth_scale * arrowwidth

    ax.quiver(
        arrow_positions[:,0], arrow_positions[:,1], arrow_directions[:,0],
        arrow_directions[:,1], angles='xy', scale_units='xy', width=arrowwidth,
        scale=1.5, headwidth=1.8 * headval, headlength=headval,
        headaxislength=0.8 * headval, zorder=20
    )

    return ax


def _get_incident_flows(av_final_flows, incidence_matrix):
    av_final_flows = scipy.sparse.diags(av_final_flows)
    incident_flows = incidence_matrix @ av_final_flows

    return incident_flows


def _plot(im_path, model_widths, nodes, edges, is_added, border_node_coords,
          arrow_positions, arrow_directions):
    fig, ax = plt.subplots(figsize=(6,8))
    im = imread(im_path)

    ax = _plot_edges_nodes(
        ax, im, edges, model_widths, is_added, nodes, border_node_coords
    )

    ax = _plot_arrows(ax, model_widths, arrow_positions, arrow_directions)

    fig.tight_layout()
 
    return fig


def _get_border_node_coords(output_data, nodes):
    border_nodes = output_data['border_nodes']
    border_node_coords = nodes[border_nodes]

    return border_node_coords


def _get_arrows(nodes, directed_edges):
    coords = nodes[directed_edges]
    positions = coords[:,0,:]
    directions = coords[:,1,:] - coords[:,0,:]

    return positions, directions


def _get_directed_edges(incidence_matrix, flows, edges):
    incident_flows = _get_incident_flows(flows, incidence_matrix)

    rows, cols = incident_flows.nonzero()

    directed_edges = np.zeros_like(edges) 

    for i in range(len(rows)):
        node_idx, edge_idx = rows[i], cols[i]
        
        # Positive sign means flow out of the node
        flow_direction = np.sign(incident_flows[node_idx, edge_idx])

        if flow_direction == 0:
            raise AssertionError('Zero-output from np.sign')

        # Maps 1 to 0 and -1 to 1
        flow_direction = int((1 - flow_direction) / 2)

        directed_edges[edge_idx, flow_direction] = node_idx 

    return directed_edges


def main():
    plot_params = param_parser.get_params()

    im_name = plot_params['im_name']
    im_path = structure.Directories.images_originals / (im_name + '.tiff')

    filename_maker = paths.FilenameMaker(plot_params)
    input_file = filename_maker.pkl_filename
    output_data = plotting_utils.get_output_data(input_file)

    n_edges_added = plot_params['n_edges_added']

    path_to_dfs = structure.Directories.added / im_name
    dfs = dataframes.DFs(path_to_dfs, n_edges_added)
    nodes = dfs.get_nodes()
    nodes = plotting_utils.flip_coords(nodes)

    edges_dict = dfs.get_edges_dict()
    edges = edges_dict['edges']
    edge_widths = edges_dict['edge_widths']
    is_added = edges_dict['is_added']

    border_node_coords = _get_border_node_coords(output_data, nodes)

    final_conductivities = output_data['final_conductivities']
    model_widths = network_utils.get_model_widths(
        final_conductivities, edge_widths.mean()
    )

    # Arrows
    incidence_matrix = output_data['incidence_matrix']
    av_final_flows = output_data['av_final_flows']

    directed_edges = _get_directed_edges(
        incidence_matrix, av_final_flows, edges
    )
    arrow_positions, arrow_directions = _get_arrows(nodes, directed_edges)

    fig = _plot(
        im_path, model_widths, nodes, edges, is_added, border_node_coords,
        arrow_positions, arrow_directions
    )

    output_dir = structure.Directories.figures_flows
    output_file = filename_maker.get_fig_filename(output_dir)
    fig.savefig(output_file, dpi=300)


if __name__ == '__main__':
    main()
