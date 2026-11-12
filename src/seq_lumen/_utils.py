from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy as sp
from geopandas import GeoDataFrame
from scipy.spatial import KDTree
from shapely import GeometryType, points
from shapely.io import from_ragged_array
from sklearn.neighbors import radius_neighbors_graph

if TYPE_CHECKING:
    from anndata import AnnData
    from anndata.acc import AdRef, MultiAcc


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


def get_circles(centers: np.ndarray, outer_radius: float) -> GeoDataFrame:
    """Circles of radius `outer_radius` around each center, as polygons.

    Parameters
    ----------
    centers
        `(n, 2)` center coordinates (µm).
    outer_radius
        Circle radius (µm).

    Returns
    -------
    `n` polygons, indexed `0..n-1` in the order of the centers.
    """
    geometries = from_ragged_array(geometry_type=GeometryType.POINT, coords=centers)
    dilated_circles_outer = GeoDataFrame({"geometry": geometries, "radius": outer_radius})
    circles = circles_to_polygons(dilated_circles_outer)
    return circles


def assign_to_nearest_lumen(
    xy: np.ndarray, lumen: np.ndarray, connect_distance: float, max_distance: float
) -> np.ndarray:
    """Which lumen each point borders: the connected component of its nearest lumen point.

    Parameters
    ----------
    xy
        `(n, 2)` points to assign (µm).
    lumen
        `(m, 2)` lumen points on the density grid (µm).
    connect_distance
        Lumen points closer than this are in the same lumen.
    max_distance
        Points with no lumen point within this distance are unassigned.

    Returns
    -------
    `(n,)` lumen ids `0..k-1`, `-1` where unassigned.
    """
    if len(lumen) == 0:
        return np.full(len(xy), -1)
    component = sp.sparse.csgraph.connected_components(radius_neighbors_graph(lumen, radius=connect_distance))[1]
    dist, idx = KDTree(lumen).query(xy, distance_upper_bound=max_distance)
    # Unmatched queries get `idx == len(lumen)`, so clip before indexing and mask them out.
    return np.where(np.isfinite(dist), component[np.minimum(idx, len(lumen) - 1)], -1)


def count_points_within_circles(xy: np.ndarray, circles: GeoDataFrame) -> np.ndarray:
    """Number of points inside (or on the boundary of) each circle.

    Parameters
    ----------
    xy
        `(m, 2)` coordinates of the points to count.
    circles
        `n` polygons indexed `0..n-1`, as returned by :func:`get_circles`.

    Returns
    -------
    `(n,)` integer counts, aligned with `circles`.
    """
    shapely_points = GeoDataFrame(geometry=points(xy))
    joined = circles.sjoin(shapely_points)
    point_count = joined.index.value_counts()
    result = np.array([0] * circles.shape[0])
    result[point_count.index.to_numpy()] = point_count.to_numpy()
    return result


def centroids(adata: AnnData, centroids_acc: MultiAcc) -> np.ndarray:
    """Look up and validate the `(n_obs, 2)` centroids."""
    centroids = adata[centroids_acc]
    if not isinstance(centroids, np.ndarray) or len(centroids.shape) != 2 or centroids.shape[1] != 2:
        raise ValueError(f"{centroids_acc} must point at a n_obs x 2 array of points")
    return centroids


def centroids_and_border(adata: AnnData, centroids_acc: MultiAcc, border_acc: AdRef) -> tuple[np.ndarray, np.ndarray]:
    """Look up and validate the `(n_obs, 2)` centroids and the boolean `n_obs` border mask."""
    if not (border_ids := adata[border_acc]).dtype == "bool":
        raise ValueError("border must be marked with a boolean mask")
    return centroids(adata, centroids_acc), border_ids


def samples(adata: AnnData, sample_acc: AdRef) -> pd.Categorical:
    """Look up and validate the categorical `n_obs` sample labels; unused categories are dropped."""
    if not isinstance(labels := adata[sample_acc], pd.Categorical):
        raise TypeError(f"{sample_acc} must point at a categorical column, e.g. `adata.obs[col].astype('category')`")
    if labels.isna().any():
        raise ValueError(f"{sample_acc} has missing values; every cell needs a sample")
    return labels.remove_unused_categories()


def gap_quantile(xy: np.ndarray, quantile: float, rng: np.random.Generator) -> float:
    """`quantile` of the distance from random in-tissue locations to the nearest centroid."""
    # In tissue: >= 10 cells within 30 µm, so that large lumens and the outside don't count.
    # Fixed tissue test and sample size; expose them if sections are much sparser or tinier.
    locations = rng.uniform(xy.min(axis=0), xy.max(axis=0), size=(200_000, 2))
    dist, _ = KDTree(xy).query(locations, k=10, distance_upper_bound=30)
    return float(np.quantile(dist[np.isfinite(dist[:, -1]), 0], quantile))
