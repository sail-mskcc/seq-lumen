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
    from anndata import AnnData
    from anndata.acc import AdRef, MultiAcc


def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    result = np.array([bind.get(itm, None) for itm in a])  # None can be replaced by any other "not in b" value
    result = [result, np.array([x is not None for x in result])]
    return result


def _gen_bins(x: np.ndarray, pixel_size: int) -> np.ndarray:
    """Makes a grid from the lowest reachable x to highest, making each returned pixel `pixel_size` of the inputs"""
    return np.linspace(np.min(x), np.max(x), int((np.max(x) - np.min(x)) / pixel_size))


def _pad_and_shift(x: np.ndarray) -> np.ndarray:
    """Shift the linspac-ed array by its difference size and append to the end a final point one more evenly-spaced size away."""
    x_diffed = x - np.diff(x)[0] / 2
    return np.append(x_diffed, np.max(x_diffed) + np.diff(x_diffed)[0])


def propagate_points(anchor, candidate, max_distance=20):
    joined_points = np.concatenate((candidate, anchor))
    point_label = np.array([0] * candidate.shape[0] + [1] * anchor.shape[0])
    neighbor_graph = radius_neighbors_graph(joined_points, radius=max_distance, include_self=True)
    anchor_neighbor = np.ravel(neighbor_graph[:, point_label == 1].sum(axis=1))[point_label == 0] > 0
    rescue_points = candidate[anchor_neighbor, :]
    return rescue_points


def attach_component_size(points, max_distance=20):
    neighbor_graph = radius_neighbors_graph(points, radius=max_distance)
    connected_components = sp.sparse.csgraph.connected_components(neighbor_graph)[1]
    lesion_size = pd.Series(connected_components).value_counts()
    points = np.concatenate((points, lesion_size.loc[connected_components].to_numpy()[:, None]), axis=1)
    return points


def filter_lesions_by_size(points, max_distance=20, min_cells_per_lesion=5):
    neighbor_graph = radius_neighbors_graph(points, radius=max_distance)
    connected_components = sp.sparse.csgraph.connected_components(neighbor_graph)[1]
    lesion_size = pd.Series(connected_components).value_counts()
    valid_lesions = lesion_size.index[lesion_size >= min_cells_per_lesion].to_numpy()
    result = points[ismember(connected_components, valid_lesions)[1], :].copy()
    return result


@dataclass(frozen=True)
class SeqLumen:
    interpoint_distance: int = 5
    blurring_bandwdith: int = 3
    min_density: float = 0.01
    candidate_max_distance: int = 70
    rescue_max_distance: int = 15
    min_cells_per_lesion: int = 5
    n_iterations: int = 1
    anchor_max_distance: int = 20

    def _empty_regions(
        self,
        xy: np.ndarray,
    ) -> np.ndarray:
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

    def _get_rasterized_holes(self, rasterized: np.ndarray):
        h, w = rasterized.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilation = cv2.dilate(rasterized, kernel, iterations=self.n_iterations)
        del rasterized
        filled = cv2.floodFill(np.logical_not(dilation).astype(np.uint8), mask, (0, 0), 0)
        del dilation
        final = np.logical_not(filled[2]).astype(np.uint8)
        del filled
        return cv2.erode(final, kernel, iterations=self.n_iterations + 1)

    def resolve(self, adata: AnnData, *, centroids_acc: MultiAcc, border_acc: AdRef):

        centroids = adata[centroids_acc]
        if not (border_ids := adata[border_acc]).dtype == "bool":
            raise ValueError("border must be marked with a boolean mask")
        if not isinstance(centroids, np.ndarray) or len(centroids.shape) != 2 or centroids.shape[1] != 2:
            raise ValueError(f"{centroids_acc} must point at a n_obs x 2 array of points")
        empty_region_coords = self._empty_regions(centroids)

        border_centroids = centroids[border_ids, :].copy()

        # Initialize image on the basis of epithelial coordinates
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
        return (centroids, border_centroids, self._rescue_points(lumen, remaining_empty, border_centroids))

        # Rescue empty points that are also in lumen, but were not captured because cysts were not completely filled

    def _rescue_points(self, lumen: np.ndarray, remaining_empty: np.ndarray, border_centroids: np.ndarray):
        anchor_points = filter_lesions_by_size(
            border_centroids, max_distance=self.anchor_max_distance, min_cells_per_lesion=self.min_cells_per_lesion
        )
        lumen = np.concatenate((lumen, np.array([0] * lumen.shape[0])[:, None]), axis=1)

        candidate_points = propagate_points(anchor_points, remaining_empty, max_distance=self.candidate_max_distance)
        # candidate_points = remaining_empty
        rescue_points = propagate_points(anchor_points, candidate_points, max_distance=self.rescue_max_distance)
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
