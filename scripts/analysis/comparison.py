import matplotlib.pyplot as plt
from skimage.io import imread

from src import (dataframes, network_utils, param_parser, paths, plotting_utils,
                 structure)


def _plot_subfig(ax, im, edges, edge_widths, is_added,
                 nodes, border_node_coords, title):
    # Scales input and model widths in plots
    scale = 0.08
    alpha = 0.5
    border_nodes_s = 0.1
    border_nodes_c = 'k'

    # Show image        
    ax.imshow(im)

    # Edges
    ax = plotting_utils.plot_edge_widths(
        nodes, edges, edge_widths, is_added, scale, alpha, ax
    )
 
    # Nodes
    ax.scatter(
        nodes[:, 0], nodes[:, 1], s=0.1, color='g',
        edgecolor='None'
    )

    # Border nodes
    if len(border_node_coords) != 0:
        ax.scatter(
            border_node_coords[:,0], border_node_coords[:,1], s=border_nodes_s,
            edgecolor='None', color=border_nodes_c, zorder=10
        )

    ax.set_title(title)


def _plot(im_path, model_widths, nodes, edges, edge_widths, is_added,
          border_node_coords):

    fig, axs = plt.subplots(1, 2, figsize=(6,8), sharex=True, sharey=True)

    im = imread(im_path)

    titles = ['Input', 'Model']
    plot_widths = [edge_widths, model_widths]

    for i in range(2):
        _plot_subfig(
            axs[i], im, edges, plot_widths[i], is_added,
            nodes, border_node_coords, titles[i]
        )

    return fig


def _get_border_node_coords(output_data, nodes):
    border_nodes = output_data['border_nodes']
    border_node_coords = nodes[border_nodes]

    return border_node_coords


def main():
    plot_params = param_parser.get_params()

    im_name = plot_params['im_name']
    im_path = structure.Directories.images_originals
    im_path = im_path / (im_name + '.tiff')

    filename_maker = paths.FilenameMaker(plot_params)
    filename = filename_maker.pkl_filename
    output_data = plotting_utils.get_output_data(filename)

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

    fig = _plot(
        im_path, model_widths, nodes, edges, edge_widths, is_added,
        border_node_coords,
    )

    output_dir = structure.Directories.figures_comparison
    figname = filename_maker.get_fig_filename(output_dir)
    fig.savefig(figname, dpi=1000)


if __name__ == '__main__':
    main()
