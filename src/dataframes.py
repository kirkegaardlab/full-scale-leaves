import numpy as np
import pandas as pd

from src import structure


def _extend_fname(fname, n_edges_added):
    fname += f'_n_edges_added={n_edges_added}.feather'
    return fname


class DFs:
    def __init__(self, path_to_file, n_edges_added):
        fname = _extend_fname('nodes', n_edges_added)
        fname = path_to_file / fname
        self._node_df = pd.read_feather(fname)

        fname = _extend_fname('edges', n_edges_added)
        fname = path_to_file / fname
        self._edge_df = pd.read_feather(fname)

    def get_nodes(self):
        nodes = self._node_df[['nodes_x', 'nodes_y']].to_numpy()
        return nodes

    def get_degrees(self):
        degrees = self._node_df['degree'].to_numpy()
        return degrees

    def get_edges_dict(self):
        edges = self._edge_df[['edge_start', 'edge_end']].to_numpy()
        edge_lengths = self._edge_df['length'].to_numpy()
        edge_widths = self._edge_df['width'].to_numpy()
        edge_areas = self._edge_df['area'].to_numpy()
        is_added = self._edge_df['is_added'].to_numpy()

        edges_dict = {'edges': edges,
                      'edge_lengths': edge_lengths,
                      'edge_widths': edge_widths,
                      'edge_areas': edge_areas,
                      'is_added': is_added}

        return edges_dict


def _make_nodes_df(nodes, degrees):
    df = pd.DataFrame({'nodes_x': nodes[:,0],
                       'nodes_y': nodes[:,1],
                       'degree': degrees})
    return df


def _get_added_mask(n_data_edges, n_edges_added):
    added_mask = np.zeros(n_data_edges, dtype=np.bool_)
    # If n_edges_added is 0, it sets all entries to True
    if n_edges_added > 0:
        added_mask[-n_edges_added:] = True
    return added_mask


def _make_edges_df(edges, edge_lengths, edge_widths, edge_areas, n_edges_added):
    is_added = _get_added_mask(len(edges), n_edges_added)

    df = pd.DataFrame({'edge_start': edges[:,0],
                       'edge_end': edges[:,1],
                       'length': edge_lengths,
                       'width': edge_widths,
                       'area': edge_areas,
                       'is_added': is_added})

    return df


def save_dfs(nodes, degrees, edges, edge_lengths, edge_widths, edge_areas,
             output_dir, n_edges_added):

    structure.Directories.make_dir(output_dir)

    df = _make_nodes_df(nodes, degrees)

    fname = _extend_fname('nodes', n_edges_added)
    fname = output_dir / fname

    df.to_feather(fname)

    df = _make_edges_df(
        edges, edge_lengths, edge_widths, edge_areas, n_edges_added
    )
    fname = _extend_fname('edges', n_edges_added)
    fname = output_dir / fname

    df.to_feather(fname)
