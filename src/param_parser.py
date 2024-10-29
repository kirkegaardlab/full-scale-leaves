import argparse


DEFAULT_PARAMS = {
    'im_name': 'artificial',
    'sink_fluctuation': 0.0,
    'gamma': 0.5,
    'error_threshold': 0.001,
    'method': 'scaling',
    'full_system': False,
    'n_edges_added': 0,
    'source_pos': 'Top',
    'nx': 31,
    'ny': 25,
    'edge_length': 1.0,
    'edge_width': 0.1,
    'optimize': False,
    'use_gpu': False
}


def _parse_cl_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--im',
                        dest='im_name',
                        type=str,
                        help=('The filename of the image on which to run the '
                              'model.\nIf no argument is provided, the model '
                              'will run on an artificial leaf.'))

    parser.add_argument('--sf',
                        dest='sink_fluctuation',
                        type=float,
                        help=('Sink fluctuation parameter.'))

    parser.add_argument('--g',
                        dest='gamma',
                        type=float,
                        help=('Gamma parameter.'))

    # Define the custom function for type conversion
    def _float_or_str(value):
        if value == 'inf':
            return str(value)
        else:
            return float(value)

    parser.add_argument('--e',
                        dest='error_threshold',
                        type=_float_or_str,
                        help=('Error threshold for the iterative solver.\n'
                              + 'float: the error threshold.\n'
                              + 'inf: run one iteration.'))

    parser.add_argument('--m',
                        dest='method',
                        type=str,
                        choices=['scaling',
                                 'evolution',
                                 'evolution_with_growth'],
                        help=('The method used to update the connectivities.'))

    parser.add_argument('--full',
                        dest='full_system',
                        action='store_true',
                        help=('Whether to use subsystems or the full system.\n'
                              + 'Using subsystems has significant memory '
                              + 'advantages for larger leaves. It also usually '
                              + 'leads to speedup.'))

    parser.add_argument('--nadded',
                        dest='n_edges_added',
                        type=int,
                        help=('Number of edges added to a leaf during '
                              + 'preprocessing.'))

    parser.add_argument('--pos',
                        dest='source_pos',
                        type=str,
                        choices=['Top', 'Bottom', 'Left', 'Right'],
                        help=('The position of the source in the image.'))

    parser.add_argument('--nx',
                        dest='nx',
                        type=int,
                        help=('Number of nodes along the leaf stem.'))

    parser.add_argument('--ny',
                        dest='ny',
                        type=int,
                        help=('Half the number of rows of nodes perpendicular '
                              + 'to the leaf stem.'))

    parser.add_argument('--l',
                        dest='edge_length',
                        type=float,
                        help=('Length of an artificial leaf edge.'))

    parser.add_argument('--w',
                        dest='edge_width',
                        type=float,
                        help=('Width of an artificial leaf edge.'))

    parser.add_argument('--opt',
                        dest='optimize',
                        action='store_true',
                        help=('Run the optimizer.'))

    parser.add_argument('--gpu',
                        dest='use_gpu',
                        action='store_true',
                        help=('Use GPU for computations.'))

    # To dictionary
    params = vars(parser.parse_args())
    return params


def _merge_default_and_cl_params(command_line_params):
    all_params = DEFAULT_PARAMS.copy()

    for cl_param_name, cl_param_val in command_line_params.items():
        if cl_param_val is not None:
            all_params[cl_param_name] = cl_param_val

    return all_params


def get_params():
    command_line_params = _parse_cl_params()
    all_params = _merge_default_and_cl_params(command_line_params)

    return all_params
