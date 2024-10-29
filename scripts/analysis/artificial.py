from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from src import network_utils, param_parser, paths, plotting_utils, structure


def _plot(plot_params, output_data):
    final_conductivities = output_data['final_conductivities']
    incidence_matrix = output_data['incidence_matrix']
    nodes = output_data['nodes']
    edges = output_data['edges']

    fig, ax = plt.subplots(figsize=(8, 10))

    # Shape: (n_edges, n_nodes)
    incidence_matrix = incidence_matrix.T

    n_nodes = len(nodes)

    edge_widths_mean = plot_params['edge_width']
    edge_widths = network_utils.get_model_widths(
        final_conductivities, edge_widths_mean
    )

    coord_list = plotting_utils.get_coord_list(nodes, edges)

    # Plot edges
    scale = 10
    edge_color = 'r'
    alpha = 1.0
    line_segments = LineCollection(
        coord_list, linewidths=scale * edge_widths, color=edge_color,
        alpha=alpha
    )

    ax.add_collection(line_segments)

    # Plot nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], s=5.0, c='k')

    method = plot_params['method']
    nx, ny = plot_params['nx'], plot_params['ny']
    gamma = plot_params['gamma']
    sink_fluctuation = plot_params['sink_fluctuation']

    fig.suptitle(
        (f'No. of nodes: {n_nodes}, '
         + r'$n_x$' + f' = {nx}, '
         + r'$n_y$' + f' = {ny}, '
         + f'{method}' + '\n'
         + f'gamma = {gamma}, '
         + f'sink_fluctuation = {sink_fluctuation}'),
        size=16
    )

    fig.tight_layout()

    return fig


def main():
    plot_params = param_parser.get_params()

    filename_maker = paths.FilenameMaker(plot_params)
    filename = filename_maker.pkl_filename
    output_data = plotting_utils.get_output_data(filename)

    fig = _plot(plot_params, output_data)

    output_dir = structure.Directories.figures_artificial
    figname = filename_maker.get_fig_filename(output_dir)
    fig.savefig(figname, dpi=300)


if __name__ == '__main__':
    main()
