from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import scipy as sp
from anndata.acc import AdRef, MetaAcc
from geopandas import GeoDataFrame, points_from_xy
from shapely import GeometryType
from shapely.io import from_ragged_array
from sklearn.neighbors import radius_neighbors_graph
from tqdm import tqdm

if TYPE_CHECKING:
    from anndata import AnnData
    from anndata.acc import MultiAcc


def _gen_bins(x: np.ndarray, pixel_size: float) -> np.ndarray:
    """Evenly spaced points from `min(x)` to `max(x)`, roughly `pixel_size` apart (in the units of `x`).

    `linspace` includes both ends, so the true spacing is `extent / (n - 1)`, slightly larger than `pixel_size`.
    """
    return np.linspace(np.min(x), np.max(x), int((np.max(x) - np.min(x)) / pixel_size))


def _pad_and_shift(x: np.ndarray) -> np.ndarray:
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


@dataclass(frozen=True)
class SeqLumen:
    """Lumen detection from cell centroids (luminal-area steps 1-3 of the paper's STAR Methods).

    Every distance is in the units of the centroid coordinates. The defaults (and the paper) assume **µm**.

    Algorithm
    ---------
    1. **Empty space.** All centroids are binned onto the *density grid*, a mesh with spacing
       :attr:`interpoint_distance` (5 µm/px by default). The per-pixel cell counts are smoothed with a
       :attr:`blurring_bandwdith`-px box filter, and pixels whose smoothed count is below :attr:`min_density`
       become *empty points*. Each empty point stands for `interpoint_distance**2` µm² of area.
    2. **Enclosed lumen.** The border (epithelial) centroids are drawn onto the *border raster* at 1 µm/px.
       The raster is dilated :attr:`n_iterations` times with a 10 px (10 µm) elliptical kernel to close gaps
       between neighbouring cells, holes fully enclosed by the dilated border cells are filled, and the result is
       eroded `n_iterations + 1` times to shrink it back. Empty points inside the filled mask are lumen.
    3. **Open lumen (rescue).** Lumen that isn't fully enclosed (e.g. a torn gland) is grown out from the
       border. Anchors are border cells in lesions (groups of border cells linked within
       :attr:`anchor_max_distance`) of at least :attr:`min_cells_per_lesion` cells. The remaining empty points
       within :attr:`candidate_max_distance` of an anchor are candidates. Candidates within
       :attr:`rescue_max_distance` of an anchor seed the rescued lumen, which then grows 5 times by one
       density-grid step into neighbouring candidates.

    Divergences
    -----------
    Marked in the source with ``# Divergence from notebook:`` or ``# Divergence from paper:``.

    From the original notebook (bugs fixed here, so results differ slightly from the published ones):

    - The y axis of the density grid was sized with the x extent, so its spacing wasn't :attr:`interpoint_distance`.
    - Rescued lumen was counted up to 6 times, because each growth iteration contains the previous ones and all were
      concatenated. This inflated both lumen area and rescued component sizes.
    - Flood fill was seeded at pixel (0, 0) without padding, so it silently failed if dilated border cells covered
      that corner.
    - Coordinates < 1 (and empty points made negative by the half-step shift) wrapped around when cast to `uint32`
      instead of landing on the image edge; an empty rescue set crashed.

    From the paper's text (the code here, like Jose's notebook, does the following instead):

    - Step 1 smooths with a 3x3 box filter, not a Gaussian, and empty points are shifted half a grid step down-left
      (to bin corners).
    - Step 2 rasterizes at 1 µm/px, not on the 5 µm density grid, so the 10 px kernel is 10 µm, not ~50 µm;
      it dilates :attr:`n_iterations` (2) times and erodes 3 times, not once each.
    - Step 3 seeds with 15 µm (:attr:`rescue_max_distance`), not 20 µm, and adds steps the paper doesn't describe:
      only lesions of >= :attr:`min_cells_per_lesion` cells are anchors, growth is capped at
      :attr:`candidate_max_distance` and 5 iterations.
    """

    interpoint_distance: float = 5
    """Spacing of the density grid in µm, i.e. its µm/px. Paper: "5 µm px⁻¹".

    Used in step 1; one grid step is also how far the rescued lumen grows per iteration in step 3.
    """

    blurring_bandwdith: int = 3
    """Side of the box filter that smooths cell counts, in density-grid pixels (3 px x 5 µm/px = 15 µm).

    The paper describes a Gaussian with a 3 px bandwidth, but the code uses a 3x3 box filter (`cv2.blur`).
    Used in step 1.
    """

    min_density: float = 0.01
    """Smoothed cell count per density-grid pixel (unitless) below which the pixel is empty. Paper: 0.01.

    With a 3x3 box filter, one centroid anywhere in the window gives 1/9,
    so in practice this means "no centroid within ±1 grid pixel (±5 µm)". Used in step 1.
    """

    candidate_max_distance: float = 70
    """Maximum distance (µm) from an anchor for an empty point to be a rescue candidate; caps how far lumen can grow.

    Used in step 3.
    """

    rescue_max_distance: float = 15
    """Distance (µm) from an anchor within which candidates seed the rescued lumen. Paper text: 20 µm.

    Used in step 3.
    """

    min_cells_per_lesion: int = 4
    """Minimum number of border cells in a lesion for its cells to be anchors.

    Used in step 3.
    """

    n_iterations: int = 2
    """Number of dilations of the epithelial raster with the 10 px (10 µm) elliptical kernel; erosion uses one more.

    Each iteration grows the mask by ~5 µm (~10 µm total by default). Paper text: 10 px disk, 1 iteration.
    Used in step 2.
    """

    anchor_max_distance: float = 20
    """Distance (µm) below which two border cells belong to the same lesion. Paper: 20 µm.

    Used in step 3.
    """

    lumen_radius: float = 60
    """Radius (µm) of the circle around each cell in which lumen points are counted. Paper: 60 µm (Figure 2H).

    Used in :meth:`count_lumen`.
    """

    def _empty_regions(self, xy: np.ndarray) -> np.ndarray:
        """Step 1: density-grid points with no nearby cells.

        Parameters
        ----------
        xy
            `(n_obs, 2)` cell centroids.

        Returns
        -------
        `(n, 2)` coordinates of empty density-grid points.
        """
        x, y = xy.T

        # Create bins of size `interpoint_distance` based on the x and y dimensions
        # Divergence from notebook: `yy` there was sized with `x`'s extent, so its spacing wasn't `interpoint_distance`.
        xx, yy = (_gen_bins(e, self.interpoint_distance) for e in [x, y])

        # Generate 2d grid from the 1d bins
        xx_base, yy_base = np.meshgrid(xx, yy)

        # Get cell density in each point along each bin
        h = np.histogram2d(x, y, tuple(_pad_and_shift(e) for e in [xx, yy]))

        # Blur cellular density estimate and find x,y coordinates that have very few cells (empty regions)
        # Divergence from paper: a box filter (`cv2.blur`), not a Gaussian.
        empty_bins = np.ravel(cv2.blur(h[0].T, (self.blurring_bandwdith, self.blurring_bandwdith)) < self.min_density)
        xval_linear, yval_linear = (np.ravel(base)[empty_bins] for base in (xx_base, yy_base))

        # These are the empty points
        return np.transpose(np.vstack((xval_linear, yval_linear)))

    def _get_rasterized_holes(self, rasterized: np.ndarray) -> np.ndarray:
        """Step 2: dilate the epithelial raster, fill enclosed holes, erode back.

        Parameters
        ----------
        rasterized
            `(h, w)` binary image at 1 µm/px, 1 where an epithelial centroid falls.

        Returns
        -------
        `(h + 2, w + 2)` uint8 mask (the floodFill mask), 1 = epithelium or enclosed hole.
        Note the 1-px offset: `output[i + 1, j + 1]` corresponds to `rasterized[i, j]`.
        """
        h, w = rasterized.shape
        # Divergence from paper: 10 px on the 1 µm/px raster is 10 µm (paper: 10 px on the 5 µm grid),
        # and it dilates `n_iterations` times and erodes `n_iterations + 1` times (paper: 1 each).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilation = cv2.dilate(rasterized, kernel, iterations=self.n_iterations)
        del rasterized
        # Flood the background (non-epithelium) connected to the (0, 0) corner.
        # The fill is recorded in `mask` (returned at index 2): 1 = outside, 0 = epithelium or enclosed hole.
        # Divergence from notebook: the image is padded with 1 px of background so the (0, 0) seed is outside the tissue even if dilated
        # epithelium reaches the corner; slicing the mask back keeps its 1-px offset relative to `rasterized`.
        padded = np.pad(np.logical_not(dilation).astype(np.uint8), 1, constant_values=1)
        del dilation
        filled = cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)
        del padded
        final = np.logical_not(filled[2][1:-1, 1:-1]).astype(np.uint8)
        del filled
        return cv2.erode(final, kernel, iterations=self.n_iterations + 1)

    def _rescue_points(
        self, lumen: np.ndarray, remaining_empty: np.ndarray, border_centroids: np.ndarray
    ) -> np.ndarray:
        """Step 3: grow lumen outward from epithelial anchor cells into empty points that aren't enclosed.

        Parameters
        ----------
        lumen
            `(n, 2)` enclosed lumen points from step 2.
        remaining_empty
            `(m, 2)` empty points not enclosed.
        border_centroids
            `(k, 2)` epithelial centroids.

        Returns
        -------
        `(n + r, 3)` lumen points with their component size (0 for enclosed lumen).
        """
        # Anchors: epithelial cells in lesions of >= `min_cells_per_lesion` cells.
        # Note on the paper: the lesion-size filter, `candidate_max_distance` and the 5-iteration cap aren't described.
        anchor_points = filter_lesions_by_size(
            border_centroids, max_distance=self.anchor_max_distance, min_cells_per_lesion=self.min_cells_per_lesion
        )
        # Enclosed (step 2) lumen gets component size 0 in the 3rd column.
        lumen = np.concatenate((lumen, np.array([0] * lumen.shape[0])[:, None]), axis=1)

        candidate_points = propagate_points(anchor_points, remaining_empty, max_distance=self.candidate_max_distance)
        # candidate_points = remaining_empty
        # Divergence from paper: seeds within `rescue_max_distance` = 15 µm (paper: 20 µm). This is what is done in Jose's notebook.
        rescued = propagate_points(anchor_points, candidate_points, max_distance=self.rescue_max_distance)
        # Grow one density-grid step per iteration: `interpoint_distance + 1` reaches the 4 direct neighbours
        # (5 µm) but not diagonals (~7.1 µm). Max reach ≈ rescue_max_distance + 5 * interpoint_distance = 40 µm.
        # Divergence from notebook: each iteration's result contains the previous one (rescued points are candidates
        # too), so only the last is kept; the notebook concatenated all of them, counting points up to 6 times.
        # It also had no guard for an empty rescue set.
        num_propagate = 5
        if len(rescued) > 0:
            for _ in tqdm(range(num_propagate)):
                rescued = propagate_points(rescued, candidate_points, max_distance=self.interpoint_distance + 1)
            lumen = np.concatenate((lumen, attach_component_size(rescued, max_distance=self.interpoint_distance + 1)))

        return lumen

    def lumen_coords(self, adata: AnnData, *, centroids_acc: MultiAcc, border_acc: AdRef) -> np.ndarray:
        """Find luminal points.

        Parameters
        ----------
        adata
            One tissue section.
        centroids_acc
            Accessor for an `(n_obs, 2)` array of cell centroids (µm for the default parameters, non-negative).
        border_acc
            Accessor for a boolean `n_obs` mask of the cells lining lumens (epithelial cells).

        Returns
        -------
        `(n, 3)` lumen points: x, y on the density grid (each point = `interpoint_distance**2` µm²) and the size
        of the rescued connected component it belongs to (0 for points inside a closed hole from step 2).
        """
        centroids = adata[centroids_acc]
        if not (border_ids := adata[border_acc]).dtype == "bool":
            raise ValueError("border must be marked with a boolean mask")
        if not isinstance(centroids, np.ndarray) or len(centroids.shape) != 2 or centroids.shape[1] != 2:
            raise ValueError(f"{centroids_acc} must point at a n_obs x 2 array of points")
        empty_region_coords = self._empty_regions(centroids)
        empty_region_coords = empty_region_coords - self.interpoint_distance / 2
        border_centroids = centroids[border_ids, :].copy()

        # Step 2. Initialize image on the basis of epithelial coordinates: 1 px = 1 centroid unit (1 µm).
        # Divergence from paper: the paper builds claims this is on the 5 µm density grid, but the notebook uses this procedure.
        # The `- 1` here is undone by the 1-px offset of the floodFill mask, so the lookup below uses no `- 1`.
        # Divergence from notebook: indices are clipped so coordinates < 1 land on the edge instead of wrapping around
        # (the notebook cast to `uint32`). The `uint8` dtype (notebook: float64) only saves memory.
        rasterized = np.zeros([int(np.max(centroids[:, 0])), int(np.max(centroids[:, 1]))], dtype=np.uint8)
        border_px = np.clip(np.floor(border_centroids).astype(np.int64) - 1, 0, np.array(rasterized.shape) - 1)
        rasterized[border_px[:, 0], border_px[:, 1]] = 1

        # Dilate border-centroid mask and fill holes
        rasterized_holes = self._get_rasterized_holes(rasterized)
        # Divergence from notebook: the half-step shift above can make empty points negative, so clip here too.
        empty_px = np.clip(np.floor(empty_region_coords).astype(np.int64), 0, np.array(rasterized_holes.shape) - 1)
        masked_value = rasterized_holes[empty_px[:, 0], empty_px[:, 1]]
        del rasterized_holes

        # Luminal points are empty points that lie within filled epithelium
        lumen = empty_region_coords[masked_value == 1, :]
        remaining_empty = empty_region_coords[masked_value == 0, :].copy()
        del masked_value
        del empty_region_coords
        # Step 3. Rescue empty points that are also in lumen, but were not captured because cysts were not completely filled
        return self._rescue_points(lumen, remaining_empty, border_centroids)

    def count_lumen(
        self, adata: AnnData, lumen: np.ndarray, *, centroids_acc: MultiAcc, lumen_count_acc: AdRef
    ) -> None:
        """Count lumen points within :attr:`lumen_radius` of each cell, i.e. the paper's "lumen area".

        The count is in density-grid pixels; multiply by `interpoint_distance**2` for µm².
        Rescued and enclosed lumen points both count.

        Parameters
        ----------
        adata
            The tissue section `lumen` was computed from.
        lumen
            `(n, 2+)` lumen points, as returned by :meth:`lumen_coords` (only x, y are used).
        centroids_acc
            Accessor for the `(n_obs, 2)` array of cell centroids (µm) to center the circles on.
        lumen_count_acc
            `obs` column accessor, e.g. `A.obs["lumen_count"]`, to write the `(n_obs,)` counts to.
        """
        centroids = adata[centroids_acc]
        circles = get_circles(centroids[:, 0], centroids[:, 1], self.lumen_radius)
        lumen_count = count_points_within_circles(lumen[:, 0], lumen[:, 1], circles)
        match lumen_count_acc:
            case AdRef(acc=MetaAcc(dim="obs"), idx=str(col)):
                adata.obs[col] = lumen_count
            case _:
                raise TypeError(
                    f"`lumen_count_acc` must be an `obs` column accessor like `A.obs['col']`, not {lumen_count_acc!r}"
                )
