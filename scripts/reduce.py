from src import (
    dataframes, network_utils, param_parser, pruning, species, structure
)


_PRUNE_RADII = {'species_1': 8, # Valid
               'species_2': 8, # Valid
               'species_3': 8,
               'species_4': 8,
               'species_5': 6, # Valid
               'species_6': 8,
               'species_7': 8}


def _get_prune_radius(im_name):
    species_ = species.get_species_number(im_name)
    prune_radius = _PRUNE_RADII[species_]
    return prune_radius


def main():
    dirs = structure.Directories()
    params = param_parser.get_params()
    im_name = params['im_name']

    input_fname = dirs.extracted / im_name
    dfs = dataframes.DFs(input_fname, n_edges_added=0)
    nodes = dfs.get_nodes()
    degrees = dfs.get_degrees()

    edges_dict = dfs.get_edges_dict()
    edges = edges_dict['edges']
    edge_lengths = edges_dict['edge_lengths']
    edge_widths = edges_dict['edge_widths']
    edge_areas = edges_dict['edge_areas']

    print(f'Graph has {len(nodes)} nodes and {len(edges)} edges.\n')

    print(f'Pruning...')
    prune_radius = _get_prune_radius(im_name)
    (nodes, degrees, edges, edge_lengths,
     edge_widths, edge_areas) = pruning.get_pruned(
         prune_radius, nodes, degrees, edges, edge_lengths, edge_widths,
         edge_areas
    )

    (nodes, degrees, edges, edge_lengths, edge_widths,
     edge_areas) = network_utils.clean(
        nodes, degrees, edges, edge_lengths, edge_widths, edge_areas
    )

    # Save
    print(f'Saving {len(nodes)} nodes and {len(edges)} edges.\n')
    output_dir = dirs.reduced / im_name
    dataframes.save_dfs(
        nodes, degrees, edges, edge_lengths, edge_widths, edge_areas,
        output_dir, n_edges_added=0
    )


if __name__ == '__main__':
    main()
