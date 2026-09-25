from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.neighbors import radius_neighbors_graph
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

    from anndata import AnnData
    from anndata.acc import AdRef, MultiAcc


def ismember(a: Iterable[Hashable], b: Iterable[Hashable]) -> list[np.ndarray]:
    """MATLAB-style `ismember`.

    Returns
    -------
    `[index, found]`: for each element of `a`, its first index in `b` (`None` if absent), and a boolean mask of presence.
    """
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    result = np.array([bind.get(itm, None) for itm in a])  # None can be replaced by any other "not in b" value
    result = [result, np.array([x is not None for x in result])]
    return result


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
    lesion_size = pd.Series(connected_components).value_counts()
    points = np.concatenate((points, lesion_size.loc[connected_components].to_numpy()[:, None]), axis=1)
    return points


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
    lesion_size = pd.Series(connected_components).value_counts()
    valid_lesions = lesion_size.index[lesion_size >= min_cells_per_lesion].to_numpy()
    result = points[ismember(connected_components, valid_lesions)[1], :].copy()
    return result


@dataclass(frozen=True)
class SeqLumen:
    """Lumen detection from cell centroids (luminal-area steps 1-3 of the paper's STAR Methods).

    Every distance is in the units of the centroid coordinates. The defaults (and the paper) assume **µm**,
    as in Xenium's `cells.parquet` / `obsm["spatial"]`; the 0.2125 µm Xenium image pixel never enters here.
    If your centroids are in image pixels, multiply them by 0.2125 first (or scale every distance up by 1 / 0.2125 ≈ 4.71).

    Two grids are used:

    - the *density grid* (step 1): spacing :attr:`interpoint_distance`, i.e. 5 µm/px by default.
      Empty/lumen points returned by :meth:`resolve` live on this grid,
      so each one stands for `interpoint_distance**2` = 25 µm² of area.
    - the *epithelial raster* (step 2): 1 centroid unit per pixel, i.e. 1 µm/px.
    """

    interpoint_distance: float = 5
    """Step 1. Spacing of the density grid, in µm (= µm/px of that grid). Paper: "5 µm px⁻¹"."""

    blurring_bandwdith: int = 3
    """Step 1. Side of the smoothing kernel, in *density-grid pixels* (3 px x 5 µm/px = 15 µm).

    The paper describes a Gaussian with 3 px bandwidth; the code is a 3x3 box filter (`cv2.blur`).
    """

    min_density: float = 0.01
    """Step 1. Smoothed cells-per-grid-pixel below which a pixel is "empty" (unitless). Paper: 0.01.

    With a 3x3 box filter one centroid anywhere in the window gives 1/9,
    so this effectively means "no centroid within ±1 grid pixel (±5 µm)".
    """

    candidate_max_distance: float = 70
    """Step 3. Empty points farther than this (µm) from every anchor are never considered for rescue."""

    rescue_max_distance: float = 15
    """Step 3. Empty points within this distance (µm) of an anchor seed the rescued lumen. Paper text: 20 µm."""

    min_cells_per_lesion: int = 4
    """Step 3. Epithelial lesions with fewer cells than this are not used as anchors. Original notebook: 4."""

    n_iterations: int = 2
    """Step 2. Dilation iterations of the 10x10 px elliptical kernel on the 1 µm/px raster; erosion uses `n_iterations + 1`.

    Each iteration grows the mask by ~5 µm (~10 µm total by default). Paper text: 10 px disk, 1 iteration.
    """

    anchor_max_distance: float = 20
    """Step 3 / lesion definition. Epithelial cells closer than this (µm) are in the same lesion. Paper: 20 µm."""

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
        # Previously, the `yy` (in Jose's original impl.) divided by `x`'s extent in `_gen_bins`, which is likely a bug.
        xx, yy = (_gen_bins(e, self.interpoint_distance) for e in [x, y])

        # Generate 2d grid from the 1d bins
        xx_base, yy_base = np.meshgrid(xx, yy)

        # Get cell density in each point along each bin
        h = np.histogram2d(x, y, tuple(_pad_and_shift(e) for e in [xx, yy]))

        # Blur cellular density estimate and find x,y coordinates that have very few cells (empty regions)
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
        mask = np.zeros((h + 2, w + 2), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilation = cv2.dilate(rasterized, kernel, iterations=self.n_iterations)
        del rasterized
        # Flood the background (non-epithelium) connected to the (0, 0) corner.
        # The fill is recorded in `mask` (returned at index 2): 1 = outside, 0 = epithelium or enclosed hole.
        # Assumes the corner is not covered by (dilated) epithelium.
        filled = cv2.floodFill(np.logical_not(dilation).astype(np.uint8), mask, (0, 0), 0)
        del dilation
        final = np.logical_not(filled[2]).astype(np.uint8)
        del filled
        return cv2.erode(final, kernel, iterations=self.n_iterations + 1)

    def resolve(
        self, adata: AnnData, *, centroids_acc: MultiAcc, border_acc: AdRef
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        `(centroids, border_centroids, lumen)`, where `lumen` is `(n, 3)`: x, y on the density grid
        (each point = `interpoint_distance**2` µm²) and the size of the rescued connected component
        it belongs to (0 for points inside a closed hole from step 2).
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
        # The `- 1` here is undone by the 1-px offset of the floodFill mask, so the lookup below uses no `- 1`.
        rasterized = np.zeros([int(np.max(centroids[:, 0])), int(np.max(centroids[:, 1]))])
        rasterized[
            border_centroids[:, 0].astype(np.uint32) - 1,
            border_centroids[:, 1].astype(np.uint32) - 1,
        ] = 1

        # Dilate border-centroid mask and fill holes
        rasterized_holes = self._get_rasterized_holes(rasterized)
        masked_value = rasterized_holes[
            empty_region_coords[:, 0].astype(np.uint32), empty_region_coords[:, 1].astype(np.uint32)
        ]
        del rasterized_holes

        # Luminal points are empty points that lie within filled epithelium
        lumen = empty_region_coords[masked_value == 1, :]
        remaining_empty = empty_region_coords[masked_value == 0, :].copy()
        del masked_value
        del empty_region_coords
        # Step 3. Rescue empty points that are also in lumen, but were not captured because cysts were not completely filled
        return (centroids, border_centroids, self._rescue_points(lumen, remaining_empty, border_centroids))

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
        anchor_points = filter_lesions_by_size(
            border_centroids, max_distance=self.anchor_max_distance, min_cells_per_lesion=self.min_cells_per_lesion
        )
        # Enclosed (step 2) lumen gets component size 0 in the 3rd column.
        lumen = np.concatenate((lumen, np.array([0] * lumen.shape[0])[:, None]), axis=1)

        candidate_points = propagate_points(anchor_points, remaining_empty, max_distance=self.candidate_max_distance)
        # candidate_points = remaining_empty
        rescue_points = propagate_points(anchor_points, candidate_points, max_distance=self.rescue_max_distance)
        # Grow one density-grid step per iteration: `interpoint_distance + 1` reaches the 4 direct neighbours
        # (5 µm) but not diagonals (~7.1 µm). Max reach ≈ rescue_max_distance + 5 * interpoint_distance = 40 µm.
        num_propagate = 5
        total_rescued = [rescue_points.copy()]
        if len(candidate_points) > 0:
            for _ in tqdm(range(num_propagate)):
                rescue_points = propagate_points(
                    total_rescued[-1],
                    candidate_points,
                    max_distance=self.interpoint_distance + 1,
                )
                total_rescued.append(rescue_points.copy())
        if any(len(rp) > 0 for rp in total_rescued):
            total_rescued = np.concatenate(total_rescued)

            # final_rescued = filter_lesions_by_size(total_rescued, max_distance = self.interpoint_distance + 1, min_cells_per_lesion = 500)
            final_rescued = attach_component_size(total_rescued, max_distance=self.interpoint_distance + 1)
            lumen = np.concatenate((lumen, final_rescued))

        return lumen
