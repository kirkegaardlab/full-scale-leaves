import logging
import pickle

import jax

from src import integration, paths, structure, system


def _setup_jax(use_gpu):
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_debug_nans', True)

    if not use_gpu:
        jax.config.update('jax_platform_name', 'cpu')


def _make_output_dir(im_name):
    output_dir = structure.Directories.model_output / im_name
    structure.Directories.make_dir(output_dir)


def _save_data(output, params):
    filename_maker = paths.FilenameMaker(params)
    filename = filename_maker.pkl_filename

    _make_output_dir(params['im_name'])
    with open(filename, 'wb') as f:
        pickle.dump(output, f)


def run(params):
    logging.info(params)

    _setup_jax(params['use_gpu'])

    init_system = system.get(params)
    output = integration.run(init_system, params)

    _save_data(output, params)
