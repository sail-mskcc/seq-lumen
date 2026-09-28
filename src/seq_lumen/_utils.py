from __future__ import annotations

import numpy as np
import scipy as sp
from geopandas import GeoDataFrame, points_from_xy
from shapely import GeometryType
from shapely.io import from_ragged_array
from sklearn.neighbors import radius_neighbors_graph


def gen_bins(x: np.ndarray, pixel_size: float) -> np.ndarray:
    """Evenly spaced points from `min(x)` to `max(x)`, roughly `pixel_size` apart (in the units of `x`).

    `linspace` includes both ends, so the true spacing is `extent / (n - 1)`, slightly larger than `pixel_size`.
    """
    return np.linspace(np.min(x), np.max(x), int((np.max(x) - np.min(x)) / pixel_size))


def pad_and_shift(x: np.ndarray) -> np.ndarray:
    """Turn evenly spaced bin centers into `len(x) + 1` bin edges (shift by half a step, append one more edge)."""
    x_diffed = x - np.diff(x)[0] / 2
    return np.append(x_diffed, np.max(x_diffed) + np.diff(x_diffed)[0])


def propagate_points(anchor: np.ndarray, candidate: np.ndarray, max_distance: float = 20) -> np.ndarray:
    """Rows of `candidate` that lie within `max_distance` (centroid units) of any row of `anchor`.

    Parameters
    ----------
    anchor
        `(n, 2)` points to grow from.
    candidate
        `(m, 2)` points that may be reached.
    max_distance
        Search radius, in centroid units (µm).

    Returns
    -------
    The `(k, 2)` subset of `candidate` that was reached.
    """
    joined_points = np.concatenate((candidate, anchor))
    point_label = np.array([0] * candidate.shape[0] + [1] * anchor.shape[0])
    neighbor_graph = radius_neighbors_graph(joined_points, radius=max_distance, include_self=True)
    anchor_neighbor = np.ravel(neighbor_graph[:, point_label == 1].sum(axis=1))[point_label == 0] > 0
    rescue_points = candidate[anchor_neighbor, :]
    return rescue_points


def attach_component_size(points: np.ndarray, max_distance: float = 20) -> np.ndarray:
    """Append the size of each point's connected component as a 3rd column.

    Parameters
    ----------
    points
        `(n, 2)` points.
    max_distance
        Points closer than this (centroid units, µm) are connected.

    Returns
    -------
    `(n, 3)` array: x, y, number of points in the component.
    """
    neighbor_graph = radius_neighbors_graph(points, radius=max_distance)
    connected_components = sp.sparse.csgraph.connected_components(neighbor_graph)[1]
    component_size = np.bincount(connected_components)[connected_components]
    return np.concatenate((points, component_size[:, None]), axis=1)


def filter_lesions_by_size(points: np.ndarray, max_distance: float = 20, min_cells_per_lesion: int = 5) -> np.ndarray:
    """Keep only points in connected components with at least `min_cells_per_lesion` points.

    With epithelial centroids this is the paper's "lesion": cells within 20 µm of each other, lesion size = component size.

    Parameters
    ----------
    points
        `(n, 2)` points.
    max_distance
        Points closer than this (centroid units, µm) are connected.
    min_cells_per_lesion
        Minimum component size to keep.

    Returns
    -------
    The `(k, 2)` subset of `points` in large-enough components.
    """
    neighbor_graph = radius_neighbors_graph(points, radius=max_distance)
    connected_components = sp.sparse.csgraph.connected_components(neighbor_graph)[1]
    lesion_size = np.bincount(connected_components)[connected_components]
    return points[lesion_size >= min_cells_per_lesion, :].copy()


def circles_to_polygons(gdf: GeoDataFrame) -> GeoDataFrame:
    """Buffer each point in `gdf` by its `radius` column into a polygon (port of spatialdata's `circles_to_polygons`).

    Each circle becomes a 64-gon (`quad_segs=16` per quarter) inscribed in the true circle,
    so it undershoots the radius by at most `radius * (1 - cos(pi / 64))` ≈ 0.07 µm at 60 µm.

    Parameters
    ----------
    gdf
        Point geometries with a `radius` column (µm).

    Returns
    -------
    A copy of `gdf` with polygon geometries.
    """
    buffered_df = gdf.copy()
    # quad_segs is the default from spatialdata
    buffered_df["geometry"] = buffered_df.apply(lambda row: row.geometry.buffer(row["radius"], quad_segs=16), axis=1)

    # Ensure the GeoDataFrame recognizes the updated geometry column
    buffered_df = buffered_df.set_geometry("geometry")

    return buffered_df


def get_circles(centers_x: np.ndarray, centers_y: np.ndarray, outer_radius: float) -> GeoDataFrame:
    """Circles of radius `outer_radius` around each center, as polygons.

    Parameters
    ----------
    centers_x, centers_y
        `(n,)` center coordinates (µm).
    outer_radius
        Circle radius (µm).

    Returns
    -------
    `n` polygons, indexed `0..n-1` in the order of the centers.
    """
    data = np.column_stack((centers_x, centers_y))
    geometry_type = GeometryType(0)
    geometries = from_ragged_array(geometry_type=geometry_type, coords=data)
    dilated_circles_outer = GeoDataFrame({"geometry": geometries, "radius": outer_radius})
    circles = circles_to_polygons(dilated_circles_outer)
    return circles


def count_points_within_circles(centers_x: np.ndarray, centers_y: np.ndarray, circles: GeoDataFrame) -> np.ndarray:
    """Number of points inside (or on the boundary of) each circle.

    Parameters
    ----------
    centers_x, centers_y
        `(m,)` coordinates of the points to count (despite the name, these are not the circle centers).
    circles
        `n` polygons indexed `0..n-1`, as returned by :func:`get_circles`.

    Returns
    -------
    `(n,)` integer counts, aligned with `circles`.
    """
    shapely_points = GeoDataFrame(geometry=points_from_xy(centers_x, centers_y))
    joined = circles.sjoin(shapely_points)
    point_count = joined.index.value_counts()
    result = np.array([0] * circles.shape[0])
    result[point_count.index.to_numpy()] = point_count.to_numpy()
    return result
