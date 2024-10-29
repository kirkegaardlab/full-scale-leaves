import pickle

import jax
import numpy as np
from scipy.optimize import minimize

from src import integration, paths, species, system


_INIT_GUESSES_1D = {'species_1': np.array([0.3]), # based on optimization 
                    'species_2': np.array([0.1]), # based on optimization
                    'species_5': np.array([0.2])} # based on optimization

_INIT_GUESSES_2D = {'species_1': np.array([0.04, 0.78]), # for added edges
                    #'species_1': np.array([0.025, 0.911]), # mean of all vals
                    'species_2': np.array([0.075, 0.300]), # mean of all vals
                    'species_5': np.array([0.0163, 0.8532]), # first val
                    'species_7': np.array([0.05, 0.9])}


def _setup_jax(use_gpu):
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_debug_nans', True)

    if not use_gpu:
        jax.config.update('jax_platform_name', 'cpu')


def _check_im_name(im_name):
    if im_name == 'artificial':
        raise ValueError(f'Invalid image name: {im_name}')


def _get_init_guess(species_, dim):
    if dim == '1d':
        init_guess = _INIT_GUESSES_1D[species_]
    elif dim == '2d':
        init_guess = _INIT_GUESSES_2D[species_]
    return init_guess


def _get_init_simplex(init_guess):
    offset = 0.05
    init_simplex = np.zeros((3, 2))
    init_simplex[0] = init_guess + np.array([-offset, -offset])
    init_simplex[1] = init_guess + np.array([0, offset])
    init_simplex[2] = init_guess + np.array([offset, -offset])
    return init_simplex


def _get_optimizer_func(dim):
    if dim == '1d':
        def optimizer_func(x, params):
            sink_fluctuation = x[0]

            print(f'sink_fluctuation = {sink_fluctuation}')

            # Insert optimization params into params dict
            params['sink_fluctuation'] = sink_fluctuation

            init_system = system.get(params)
            output = integration.run(init_system, params, optimize=True)

            loss = output['loss']
            print(f'Loss: {loss:3f}')

            return loss

    elif dim == '2d':
        def optimizer_func(x, params):
            sink_fluctuation, gamma = x

            print(f'sink_fluctuation = {sink_fluctuation}, gamma = {gamma}\n')

            # Insert optimization params into params dict
            params['sink_fluctuation'] = sink_fluctuation
            params['gamma'] = gamma

            init_system = system.get(params)
            output = integration.run(init_system, params, optimize=True)

            loss = output['loss']
            print(f'Loss: {loss:3f}')

            return loss

    return optimizer_func


def _get_options(dim, init_guess):
    if dim == '1d':
        options = {'disp': True,
                   'maxfev': 200,
                   'return_all': True,
                   'fatol': 0.1}
    elif dim == '2d':
        init_simplex = _get_init_simplex(init_guess)
        options = {'disp': True,
                   'maxfev': 200,
                   'return_all': True,
                   'initial_simplex': init_simplex,
                   'fatol': 0.1}

    return options


def _get_bounds(dim):
    if dim == '1d':
        bounds = [(0.0, 1.0)]
    elif dim == '2d':
        bounds = [(0.0, 1.0), (0.0, 1.0)]

    return bounds


def _save_data(output, filename):
    with open(filename, 'wb') as f:
        pickle.dump(output, f)


def run(params, dim):
    _setup_jax(params['use_gpu'])

    _check_im_name(params['im_name'])

    print(f'Optimizing for {dim}')

    im_name = params['im_name']
    species_ = species.get_species_number(im_name)

    optimizer_func = _get_optimizer_func(dim)
    init_guess = _get_init_guess(species_, dim)
    options = _get_options(dim, init_guess)
    bounds = _get_bounds(dim)

    print(f'Optimizing: {im_name} ({species_})')
    output = minimize(optimizer_func,
                      x0=init_guess,
                      method='Nelder-Mead',
                      args=params,
                      options=options,
                      bounds=bounds)

    filename_maker = paths.FilenameMaker(params)
    filename = filename_maker.get_optimized_filename(dim)
    _save_data(output, filename)
