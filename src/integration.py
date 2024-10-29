from abc import ABC, abstractmethod
import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
import tqdm

from src import areoles, network_utils, paths, subsystem


class _Transformations:
    def __init__(self, n_nodes, incidence_matrix):
        self._n_nodes = n_nodes

        self.Tt = self._get_Tt()
        self.Tt_I = self.Tt @ incidence_matrix
        self.It_T = (sparse.csr_matrix.transpose(self.Tt_I)).tocsr()

    def _get_Tt(self):
        Tt = sparse.lil_matrix((self._n_nodes-1, self._n_nodes))

        Tt[:,0] = 1
        for j in range(self._n_nodes-1):
            Tt[j, j+1] = -1

        Tt = Tt.tocsr()
        return Tt


class _Checkpointer:
    def __init__(self, params):
        self._previous_vals = np.inf
        self.value = None

        filename_maker = paths.FilenameMaker(params)
        self._filename = filename_maker.get_checkpoints_filename()

        with open(self._filename, 'w') as file:
            file.write('')

    def check(self, updated_vals, error_threshold):
        abs_diffs = np.abs(self._previous_vals - updated_vals)
        mean_abs_diff = abs_diffs.mean()

        with open(self._filename, 'a') as file:
            file.write(f'{mean_abs_diff}\n')

        # Set previous value to new value
        self._previous_vals = copy.deepcopy(updated_vals)
        self.value = mean_abs_diff

        # Stop after one iteration
        if error_threshold == 'inf':
            return True
        # Assume error_threshold is float value
        else:
            return (mean_abs_diff < error_threshold)


class _FlowMethodsABC(ABC):
    n_iterations = 1000

    @abstractmethod
    def get_new_conductivities(self):
        pass


class _Scaling(_FlowMethodsABC):
    name = 'scaling'

    def get_new_conductivities(self, total_flows_squared, conductivities,
                               gamma):

        conductivities = total_flows_squared**(1 / (1 + gamma))

        return conductivities


class _Evolution(_FlowMethodsABC):
    name = 'evolution'

    def get_new_conductivities(self, total_flows_squared, conductivities,
                               gamma):
        c0 = 0.1
        dt = 5 / gamma**2

        d_conductivities = (
            dt * conductivities
            * (total_flows_squared / conductivities**(1 + gamma) - gamma * c0)
        )

        conductivities += d_conductivities
        conductivities = jnp.where(conductivities < 1e-7, 1e-7, conductivities)

        return conductivities


class _EvolutionWithGrowth(_FlowMethodsABC):
    name = 'evolution_with_growth'
    current_step = 0

    def get_new_conductivities(self, total_flows_squared, conductivities,
                               gamma):
        c0 = 0.1
        dt = 0.5 / gamma**2

        d_conductivities = dt * (
            conductivities * (
                total_flows_squared / conductivities**(1 + gamma) - gamma * c0
            )
            + 0.1 * jnp.exp(-20 * self.current_step / 1500)
        )

        conductivities += d_conductivities
        conductivities = jnp.where(conductivities < 1e-7, 1e-7, conductivities)

        self.current_step += 1

        return conductivities


def _get_flow_method_classes():
    flow_methods = dict()
    for flow_method_class in _FlowMethodsABC.__subclasses__():
        method_name = flow_method_class.name
        flow_methods[method_name] = flow_method_class

    return flow_methods


def _get_flow_method(method):
    # All method classes
    flow_method_classes = _get_flow_method_classes()

    # Specific class
    FlowMethod = flow_method_classes[method]
    flow_method = FlowMethod()

    return flow_method


def _get_initial_conductivities(edge_widths):
    # Values based on edge widths from image
    conductivities = (0.5 * edge_widths)**4
    return conductivities


def _get_sparse_k(eff_conductivities, n_edges):
    sparse_k = sparse.lil_matrix((n_edges, n_edges))

    for i in range(n_edges):
        sparse_k[i,i] = eff_conductivities[i]
 
    sparse_k = sparse_k.tocsr()
    return sparse_k


def _get_k_It_T(eff_conductivities, It_T, n_edges):
    eff_conductivities = _get_sparse_k(eff_conductivities, n_edges)
    k_It_T = eff_conductivities @ It_T
    return k_It_T


def _get_A_transformed(Tt_I, k_It_T):
    A_transformed = Tt_I @ k_It_T
    A_transformed = A_transformed.toarray()
    return A_transformed


class _Cache:
    def __init__(self, func):
        self._func = func
        self._cache = {}

    def __call__(self, conductivities, *args, cache_B):
        # Jax arrays are not hashable
        hash_cs = np.array(conductivities)

        hash_val = hash(
            (hash_cs.sum(), hash_cs.std(), hash_cs.shape)
        )
 
        if hash_val in self._cache:
            print('Using cached B')
            return self._cache[hash_val]
        else:
            B = self._func(hash_cs, *args)
            if cache_B:
                print('Caching B')
                self._cache[hash_val] = B
            return B


@_Cache
def _get_B_full(conductivities, edge_lengths, n_edges, transformations,
                cache_B=True):
    Tt = transformations.Tt
    It_T = transformations.It_T
    Tt_I = transformations.Tt_I

    eff_conductivities = conductivities / edge_lengths
    
    k_It_T = _get_k_It_T(eff_conductivities, It_T, n_edges)

    A_transformed = _get_A_transformed(Tt_I, k_It_T)
    A_transformed_inv = jnp.linalg.inv(A_transformed)

    # (n_edges, n_nodes)
    B_full = k_It_T @ A_transformed_inv @ Tt

    return B_full


def _get_B_sub(conductivities, edge_lengths, I_sub, I_sub_t):
    eff_conductivities = conductivities / edge_lengths

    # Get A_inv
    KI_t = eff_conductivities[:,None] * I_sub_t
    I_K_It = I_sub @ KI_t
    A_pinv = jnp.linalg.pinv(I_K_It, hermitian=True)

    # Get B
    B_sub = KI_t @ A_pinv

    return B_sub


@partial(jax.jit, static_argnums=[1])
def _get_all_flows(B, source_sinks):
    areas = source_sinks.areas
    source_idx = source_sinks.source_idx
    sink_inds = source_sinks.sink_inds
    x = source_sinks.sink_fluctuation
    s_av = source_sinks.average_sink

    cs = source_sinks.cs

    # Scale sinks by areas
    # In-place by JAX
    B = B * areas

    B_source = B[:,source_idx] / areas[source_idx]
    B_sinks = B[:,sink_inds]
    B_sum = B_sinks.sum(axis=1)

    # Replaces matrix multiplication
    aBs = B_source[:,None] / cs - s_av * B_sum[:,None] - x * B_sinks

    # In-place by JAX
    aBs = aBs * cs

    return aBs


@partial(jax.jit, static_argnums=[1])
def _get_total_flows_squared(B, source_sinks):
    aBs = _get_all_flows(B, source_sinks)

    # Sum taken instead of average for higher numerical stability
    total_flows_squared = (aBs**2).sum(axis=1)

    return total_flows_squared


def _update_full(conductivities, edge_lengths, transformations, n_edges,
                 source_sinks, flow_method, gamma):

    B = _get_B_full(
        conductivities, edge_lengths, n_edges, transformations, cache_B=False
    )

    total_flows_squared = _get_total_flows_squared(B, source_sinks)

    conductivities = flow_method.get_new_conductivities(
        total_flows_squared, conductivities, gamma
    )

    return conductivities, B


@partial(jax.jit, donate_argnums=0, static_argnums=[5, 6])
def _update_sub(conductivities, edge_lengths, I_sub, I_sub_t, Z,
                source_sinks, flow_method, gamma):

    # B_sub.shape = (n_sub_edges, n_sub_nodes)
    B_sub = _get_B_sub(conductivities, edge_lengths, I_sub, I_sub_t)
    # Z.shape = (n_sub_nodes, n_nodes)
    # B_sub_Z.shape = (n_sub_edges, n_nodes)
    B_sub_Z = B_sub @ Z

    total_flows_squared = _get_total_flows_squared(B_sub_Z, source_sinks)

    conductivities = flow_method.get_new_conductivities(
        total_flows_squared, conductivities, gamma
    )

    return conductivities, B_sub_Z


def _iterate(conductivities, params, flow_method, source_sinks, error_threshold,
             update_func):
    checkpointer = _Checkpointer(params)

    tqdm_range = tqdm.tqdm(range(flow_method.n_iterations))
    for i in tqdm_range:
        conductivities, B = update_func(conductivities)

        # Stop if converged
        converged = checkpointer.check(conductivities, error_threshold)
        if converged:
            break

        tqdm_range.set_description(
            f'{i+1} / {flow_method.n_iterations}, '
            + f'Mean abs. error = {checkpointer.value: 0.5f}'
        )

    all_flows = _get_all_flows(B, source_sinks)

    return conductivities, all_flows


def _integrate(init_system, params):
    source_sinks = areoles.SourceSinks(
        init_system.nodes, init_system.n_nodes, init_system.source_idx,
        params['sink_fluctuation']
    )

    flow_method = _get_flow_method(params['method'])

    initial_conductivities = _get_initial_conductivities(
        init_system.edge_widths
    )
    incidence_matrix = init_system.incidence_matrix

    transformations = _Transformations(init_system.n_nodes, incidence_matrix)

    error_threshold = params['error_threshold']

    # Full system
    if params['full_system']:
        update_func = partial(_update_full,
                              edge_lengths=init_system.edge_lengths,
                              transformations=transformations,
                              n_edges=init_system.n_edges,
                              source_sinks=source_sinks,
                              flow_method=flow_method,
                              gamma=params['gamma'])
 
        final_conductivities, all_flows = _iterate(
            initial_conductivities, params, flow_method, source_sinks,
            error_threshold, update_func
        )

        av_final_flows = all_flows.mean(axis=1)

        all_border_nodes = []
        all_sub_edges = np.arange(init_system.n_edges)

    # Subsystem
    else:
        all_border_nodes = []
        all_sub_edges = []
        all_flows = np.zeros((init_system.n_edges, init_system.n_nodes - 1))

        # Calculated once for full system
        B = _get_B_full(
            initial_conductivities, init_system.edge_lengths,
            init_system.n_edges, transformations, cache_B=True
        )

        n_clusters = 10
        clusters = subsystem.get_clusters(init_system.nodes, n_clusters)

        final_conductivities = np.zeros(init_system.n_edges)

        for i, cluster in enumerate(clusters):
            print(f'Get subsystem {i+1} / {n_clusters}')

            submatrices = subsystem.get_submatrices(
                init_system.nodes, init_system.edges, incidence_matrix,
                cluster, B
            )

            Z = submatrices['Z']
            I_sub = submatrices['I_sub']
            I_sub_t = submatrices['I_sub_t']
            border_nodes = submatrices['border_nodes']
            sub_edges = submatrices['sub_edges']

            edge_lengths = init_system.edge_lengths[sub_edges]

            all_border_nodes.append(border_nodes)
            all_sub_edges.append(sub_edges)

            update_func = partial(_update_sub,
                                  edge_lengths=edge_lengths,
                                  I_sub=I_sub,
                                  I_sub_t=I_sub_t,
                                  Z=Z,
                                  source_sinks=source_sinks,
                                  flow_method=flow_method,
                                  gamma=params['gamma'])
        
            # Initial values
            conductivities = initial_conductivities[sub_edges]

            conductivities, all_flows_sub = _iterate(
                conductivities, params, flow_method, source_sinks,
                error_threshold, update_func
            )

            all_flows[sub_edges] = all_flows_sub
            final_conductivities[sub_edges] = conductivities
            
        all_border_nodes = np.concatenate(all_border_nodes)
        all_sub_edges = np.concatenate(all_sub_edges)
        av_final_flows = all_flows.mean(axis=1)

    integration_output = {'final_conductivities': final_conductivities,
                          'all_border_nodes': all_border_nodes,
                          'sub_edges': all_sub_edges,
                          'av_final_flows': av_final_flows,
                          'all_flows': all_flows}

    return integration_output


def _get_sub_edge_mask(sub_edges, n_data_widths):
    mask = np.zeros(n_data_widths, dtype=np.bool_)
    mask[sub_edges] = True
    return mask


def _get_stats_widths(sub_edges, n_edges, data_widths, model_widths):
    is_sub_edge = _get_sub_edge_mask(sub_edges, n_edges)

    # Exclude edges not within clusters, i.e. between borders
    stats_data_widths = data_widths[is_sub_edge]
    stats_model_widths = model_widths[is_sub_edge]

    return stats_data_widths, stats_model_widths


def _get_loss_with_zeros(data_widths, model_widths):
    # Prevent division by 0
    denominator = np.where(np.isclose(data_widths, 0), 1, data_widths)

    chi_squared = ((data_widths - model_widths)**2 / denominator).sum()

    # Normalize
    loss = chi_squared / len(data_widths)

    return loss


def run(init_system, params, optimize=False):
    integration_output = _integrate(init_system, params)

    final_conductivities = integration_output['final_conductivities']
    border_nodes = integration_output['all_border_nodes']
    sub_edges = integration_output['sub_edges']
    av_final_flows = integration_output['av_final_flows']
    all_flows = integration_output['all_flows']

    final_conductivities = np.array(final_conductivities)
    av_final_flows = np.array(av_final_flows)

    # Set widths of added edges to 0, for loss calculation
    data_widths = copy.deepcopy(init_system.edge_widths)
    data_widths[init_system.is_added] = 0

    model_widths = network_utils.get_model_widths(
        final_conductivities, data_widths.mean()
    )

    stats_data_widths, stats_model_widths = _get_stats_widths(
        sub_edges, init_system.n_edges, data_widths, model_widths
    )

    loss = _get_loss_with_zeros(stats_data_widths, stats_model_widths)

    output = {'final_conductivities': final_conductivities,
              'data_widths': init_system.edge_widths,
              'is_added': init_system.is_added,
              'sub_edges': sub_edges,
              'loss': loss}

    # Additional output if not optimizing
    if not optimize:
        output['av_final_flows'] = av_final_flows
        output['all_flows'] = all_flows
        output['incidence_matrix'] = init_system.incidence_matrix
        output['nodes'] = init_system.nodes
        output['degrees'] = init_system.degrees
        output['border_nodes'] = border_nodes
        output['edges'] = init_system.edges

    return output
