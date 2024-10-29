import mahotas as mh
import numba as nb
import numpy as np
import h5py
from scipy.ndimage import distance_transform_edt
from skimage.measure import label
from skimage.morphology import medial_axis, isotropic_closing
from skimage.filters import gaussian
from skimage.transform import rescale

from src import dataframes, network_utils, param_parser, structure


_UPSCALE = 5 # Scaling in each dimension


def _get_label_probs(probabilities_dir, im_name):
    input_dir = probabilities_dir / (im_name + '_Probabilities.h5')
    f = h5py.File(input_dir, 'r') # HDF5 file
    label_probs = f['exported_data']

    # Array shape = (5202, 3465, 2)
    label_probs = np.array(label_probs)

    return label_probs


def _get_mask(label_probs, closing_radius):
    # Mask contains values of 1.0 where there are
    # pixels corresponding to leaf veins0 (and nodes)
    # Assumes veins have fewer pixels than background
    i = np.argmin(label_probs.sum(axis=(0, 1)))

    mask = 1.0 * (label_probs[..., i] > 0.5)

    ## Keep the one largest connected element
    # labels is a labeled array of continuous elements
    labels = label(mask, background=0, connectivity=1)

    label_id, count = np.unique(labels, return_counts=True)

    # Remove background
    label_id, count = label_id[label_id > 0], count[label_id > 0]

    # ID of the continuous element
    label_id = label_id[np.argmax(count)]

    mask = (labels == label_id)
    mask = 1.0 * isotropic_closing(mask, radius=closing_radius)

    zero_surroundings = False
    if zero_surroundings:
        closing_radius = 45
        smoothing_std = 10
        cutoff = 0.85  # between 0.5 and 1.0

        mask_shape = isotropic_closing(mask, radius=closing_radius)
        mask_shape = gaussian(1.0 * mask_shape, sigma=smoothing_std) > cutoff
        mask = gaussian(1.0 * mask, sigma=1.0) > 0.5
        mask *= mask_shape

    return mask


def _get_skel_distances(mask):
    # Turn into pixel thin skeleton (centreline)
    seed = 0
    skel, distances = medial_axis(mask, return_distance=True, rng=seed)
    skel = 1 * skel

    return skel, distances


# Look for endpoints and branchpoints
_SKEL_ENDPOINTS = [np.array([[0, 0, 0], [0, 1, 0], [2, 1, 2]]),
                  np.array([[0, 0, 0], [0, 1, 2], [0, 2, 1]]),
                  np.array([[0, 0, 2], [0, 1, 1], [0, 0, 2]]),
                  np.array([[0, 2, 1], [0, 1, 2], [0, 0, 0]]),
                  np.array([[2, 1, 2], [0, 1, 0], [0, 0, 0]]),
                  np.array([[1, 2, 0], [2, 1, 0], [0, 0, 0]]),
                  np.array([[2, 0, 0], [1, 1, 0], [2, 0, 0]]),
                  np.array([[0, 0, 0], [2, 1, 0], [1, 2, 0]])]


def _find_skel_endpoints(skel):
    ep = np.array(
        [mh.morph.hitmiss(skel, skel_endpoints) 
          for skel_endpoints in _SKEL_ENDPOINTS]
    )

    return ep.sum(axis=0) > 0.5


_SKEL_BRANCHPOINTS = [np.array([[2, 2, 1], [1, 1, 2], [2, 2, 1]]),
                     np.array([[1, 2, 2], [2, 1, 1], [1, 2, 2]]),
                     np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]]),
                     np.array([[2, 1, 2], [1, 1, 2], [2, 2, 1]]),
                     np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]]),
                     np.array([[2, 2, 1], [1, 1, 2], [2, 1, 2]])]


def _get_branch_ep(skel, skel_branchpoint):
    return (mh.morph.hitmiss(skel, skel_branchpoint)
            + mh.morph.hitmiss(skel, skel_branchpoint[::-1, :])
            + mh.morph.hitmiss(skel, skel_branchpoint[:, ::-1])
            + mh.morph.hitmiss(skel, skel_branchpoint.T))


def _find_branch_endpoints(skel):
    ep = np.array([_get_branch_ep(skel, skel_branchpoint)
          for skel_branchpoint in _SKEL_BRANCHPOINTS])
    
    return ep.sum(axis=0) > 0.5


@nb.njit
def _prune(nodes_mask):
    directions = [np.array([-1, -1]),
                  np.array([1, -1]),
                  np.array([-1, 1]),
                  np.array([1, 1]),
                  np.array([0, -1]),
                  np.array([0, 1]),
                  np.array([-1, 0]),
                  np.array([1, 0])]

    for i in range(nodes_mask.shape[0]):
        for j in range(nodes_mask.shape[1]):
            p = np.array([i, j])
            if nodes_mask[p[0], p[1]] == 1:
                for d in directions:
                    n = p + d
                    if not (0 <= n[0] < nodes_mask.shape[0]
                            and 0 <= n[1] < nodes_mask.shape[1]):
                        continue

                    if nodes_mask[n[0], n[1]] == 1:
                        nodes_mask[n[0], n[1]] = 0

    return nodes_mask


def _get_nodes_mask(skel):
    a = _find_skel_endpoints(skel)
    b = _find_branch_endpoints(skel)

    nodes_mask = 1 * ((a + b) > 0.5)

    print('Pruning...')
    nodes_mask = _prune(nodes_mask)

    return nodes_mask


@nb.njit
def _make_graph(typed_skel, distances):
    directions = [np.array([-1, -1]),
                  np.array([1, -1]),
                  np.array([-1, 1]),
                  np.array([1, 1]),
                  np.array([0, -1]),
                  np.array([0, 1]),
                  np.array([-1, 0]),
                  np.array([1, 0])]

    typed_skel = typed_skel.copy()
    edges = []
    edge_lengths = []
    edge_widths = []
    nodes = np.argwhere(typed_skel == 2)
    nodes_idxs = np.argsort(nodes[:, 0])
    nodes = nodes[nodes_idxs, :]

    zeroed = np.zeros(
        (typed_skel.shape[0] * typed_skel.shape[1], 3), dtype=nb.int64
    )

    print('   Calculating edges...')
    for i in range(len(nodes)):
        zeroed_m = 0

        pos = [nodes[i]]
        lengths = [0]
        widths = [[distances[_UPSCALE * nodes[i][0], _UPSCALE * nodes[i][1]]]]

        while len(pos) > 0:
            new_pos = []
            new_lengths = []
            new_widths = []

            for pi, p in enumerate(pos):
                if typed_skel[p[0], p[1]] == 0:
                    continue
                zeroed[zeroed_m, 0] = p[0]
                zeroed[zeroed_m, 1] = p[1]
                zeroed[zeroed_m, 2] = typed_skel[p[0], p[1]]
                zeroed_m += 1

                typed_skel[p[0], p[1]] = 0

                for d in directions:
                    n = p + d
                    v = typed_skel[n[0], n[1]]
                    if v == 2:
                        edges.append((i, n))
                        edge_lengths.append(lengths[pi] + 1)
                        # edge_widths.append(sum(widths[pi]) / len(widths[pi]))

                        q = np.asarray(widths[pi])
                        edge_widths.append(np.percentile(q, 10))

                        # edge_widths.append(min(widths[pi]))
                        break
                else:
                    for d in directions:
                        n = p + d
                        if not (0 <= n[0] < typed_skel.shape[0]
                                and 0 <= n[1] < typed_skel.shape[1]):
                            continue
                        v = typed_skel[n[0], n[1]]
                        if v == 1:
                            new_pos.append(n)
                            add = 1
                            if abs(d[0]) == 1 and abs(d[1]) == 1:
                                add = np.sqrt(2)
                            new_lengths.append(lengths[pi] + add)
                            new_widths.append(
                                widths[pi]
                                + [distances[_UPSCALE * n[0], _UPSCALE * n[1]]]
                            )

            pos.clear()
            pos.extend(new_pos)

            lengths.clear()
            lengths.extend(new_lengths)

            widths.clear()
            widths.extend(new_widths)

        for i in range(zeroed_m):
            typed_skel[zeroed[i, 0], zeroed[i, 1]] = zeroed[i, 2]

    print('   Finding indexes...')
    final_edges = []
    for i in range(len(edges)):
        j1 = np.searchsorted(nodes[:, 0], edges[i][1][0])
        j = j1
        for k in range(j1, nodes.shape[0]):
            if nodes[k, 1] == edges[i][1][1]:
                j = k
                break

        final_edges.append((edges[i][0], j))

    return nodes, final_edges, edge_lengths, edge_widths


def _remove_duplicates(edges, edge_lengths, edge_widths, edge_areas):
    edges, unique_inds = np.unique(edges, axis=0, return_index=True)

    edge_lengths = edge_lengths[unique_inds]
    edge_widths = edge_widths[unique_inds]
    edge_areas = edge_areas[unique_inds]

    return edges, edge_lengths, edge_widths, edge_areas


def _get_nodes_and_edges(skel, distances):
    nodes_mask = _get_nodes_mask(skel)

    typed_skel = skel + nodes_mask
    nodes, edges, edge_lengths, edge_widths = _make_graph(typed_skel, distances)

    # Transform
    edges = np.vstack(edges)
    edge_lengths = np.array(edge_lengths)
    edge_widths = np.array(edge_widths)

    # Remove flips
    mask = edges[:, 0] < edges[:, 1]
    edges[mask, 0], edges[mask, 1] = edges[mask, 1], edges[mask, 0]

    # Calculate areas
    edge_areas = network_utils.get_edge_areas(edge_widths)

    edges, edge_lengths, edge_widths, edge_areas = _remove_duplicates(
        edges, edge_lengths, edge_widths, edge_areas
    )

    return nodes, edges, edge_lengths, edge_widths, edge_areas


def main():
    dirs = structure.Directories()
    probabilities_dir = dirs.probabilities
    params = param_parser.get_params()
    im_name = params['im_name']

    print('Image transformations...')
    label_probs = _get_label_probs(probabilities_dir, im_name)
    closing_radius = 2
    mask = _get_mask(label_probs, closing_radius)
    skel, _ = _get_skel_distances(mask)

    upscaled_label_probs = rescale(label_probs, _UPSCALE, channel_axis=2)
    upscaled_closing_radius = _UPSCALE * closing_radius
    upscaled_mask = _get_mask(upscaled_label_probs, upscaled_closing_radius)
    scaled_distances = distance_transform_edt(upscaled_mask) / _UPSCALE

    print('Building graph...')
    (nodes, edges, edge_lengths,
     edge_widths, edge_areas) = _get_nodes_and_edges(skel, scaled_distances)

    degrees = network_utils.set_degrees(nodes, edges)

    print('Saving...\n')
    output_dir = dirs.extracted / im_name
    dataframes.save_dfs(
        nodes, degrees, edges, edge_lengths, edge_widths, edge_areas,
        output_dir, n_edges_added=0
    )


if __name__ == '__main__':
    main()
