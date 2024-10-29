import alphashape
import matplotlib.pyplot as plt
from matplotlib import patches 
import numpy as np
from scipy import spatial
import shapely


def _get_uniform_points(nodes):
    n = 10000

    x_interval, y_interval = _get_bounds(nodes)
    x_points = np.random.uniform(*x_interval, size=(n,1))
    y_points = np.random.uniform(*y_interval, size=(n,1))
    points = np.concatenate((x_points, y_points), axis=1)

    return points


def _get_nodes_outside(polygon, nodes):
    uniform_points = _get_uniform_points(nodes)

    inside_points = [
        point for point in uniform_points
        if not shapely.geometry.Point(point).within(polygon)
    ]
    inside_points = np.vstack(inside_points)

    return inside_points


def _get_bounds(nodes):
    min_coords = nodes.min(axis=0)
    max_coords = nodes.max(axis=0)
    scale = 0.1
    offsets = scale * (max_coords - min_coords)
    xmin, ymin = min_coords - offsets
    xmax, ymax = max_coords + offsets
    x_interval, y_interval = (xmin, xmax), (ymin, ymax)
    
    return x_interval, y_interval


def _get_alpha_polygon(nodes, alpha):
    alpha_shape = alphashape.alphashape(nodes, alpha)

    # Keep only the largest polygon
    if isinstance(alpha_shape, shapely.MultiPolygon):
        pp = None
        mm = 0
        for i, p in enumerate(alpha_shape.geoms):
            if p.area > mm:
                mm = p.area
                pp = p
        alpha_shape = pp

    alpha_polygon = shapely.geometry.Polygon(alpha_shape.exterior)

    return alpha_polygon


def _get_alpha_patch(alpha_polygon):
    poly_coords = np.array(alpha_polygon.exterior.coords)

    alpha_patch = patches.Polygon(
        poly_coords, alpha=0.2, facecolor='blue', label='Alpha Shape'
    )

    return alpha_patch


def _get_areoles(nodes, outside_nodes):
    all_nodes = np.concatenate((nodes, outside_nodes), axis=0)
    
    vor = spatial.Voronoi(all_nodes)

    # Map each input point to its Voronoi region
    nodes_to_areoles = {}
    for i, node in enumerate(nodes):
        region_index = vor.point_region[i]
        region_vertices = vor.regions[region_index]
        # Exclude "vertices outside Voronoi diagram"
        valid_vertices = [v for v in region_vertices if v != -1]
        # Numpy array
        areole = vor.vertices[valid_vertices]
        nodes_to_areoles[tuple(node)] = areole

    return nodes_to_areoles


def _plot(nodes, nodes_to_areoles, alpha_polygon, outside_nodes):
    fig, ax = plt.subplots(figsize=(12,19))

    # Plot the Voronoi regions using matplotlib patches
    for i, region_polygon in enumerate(nodes_to_areoles.values()):
        polygon = patches.Polygon(region_polygon, edgecolor='black', alpha=0.3)
        ax.add_patch(polygon)

    alpha_patch = _get_alpha_patch(alpha_polygon)
    ax.add_patch(alpha_patch)

    # Plot nodes
    ax.scatter(nodes[:,0], nodes[:,1], s=1.0, color='g', edgecolors=None)
    ax.scatter(outside_nodes[:,0], outside_nodes[:,1], s=0.5, c='r')

    plt.show()


def _get_areas(nodes, n_nodes, source_idx, sink_indices):
    # Shapely polygon
    # Alpha shape for leaf nodes
    alpha = 0.008
    alpha_polygon = _get_alpha_polygon(nodes, alpha)
    
    # Random nodes surrounding leaf to make Voronoi regions
    outside_nodes = _get_nodes_outside(alpha_polygon, nodes)
    # Maps leaf nodes to Voronoi regions/areoles
    nodes_to_areoles = _get_areoles(nodes, outside_nodes)

    areas = np.empty(n_nodes)

    for i, region_polygon in enumerate(nodes_to_areoles.values()):
        shapely_polygon = shapely.geometry.Polygon(region_polygon)
        area = shapely_polygon.area
        areas[i] = area

    # areas = np.ones(n_nodes)
    # xx, yy = 915, 2215
    # idx = np.argmin((nodes[:, 0] - xx)**2 + (nodes[:, 1] - yy)**2)
    # print(nodes[idx])
    # areas[idx] = 1000 * np.max(areas)

    # _plot(nodes, nodes_to_areoles, alpha_polygon, outside_nodes)

    return areas


class SourceSinks:
    def __init__(self, nodes, n_nodes, source_idx, sink_fluctuation):
        self._n_nodes = n_nodes
        self.source_idx = source_idx
        self.sink_inds = self._get_sink_inds()
        self.areas = _get_areas(nodes, n_nodes, source_idx, self.sink_inds)
        self._sink_areas = self.areas[self.sink_inds]

        # Absolute values of sink fluctuation and average sink.
        # The values get a negative sign in the source_sinks vector.
        self.sink_fluctuation = sink_fluctuation
        self.average_sink = (1 - self.sink_fluctuation) / (n_nodes - 1)
        self.cs = 1 / (
            self.average_sink * self._sink_areas.sum()
            + self.sink_fluctuation * self._sink_areas
        )

    def _get_sink_inds(self):
        sink_indices = np.arange(self._n_nodes)
        sink_indices = sink_indices[sink_indices != self.source_idx]
        return sink_indices
