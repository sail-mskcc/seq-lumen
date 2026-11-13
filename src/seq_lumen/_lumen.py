from __future__ import annotations

import warnings
from dataclasses import InitVar, dataclass
from typing import TYPE_CHECKING, Unpack, overload

import cv2
import numpy as np
import pandas as pd
import scipy as sp
from anndata.acc import AdRef, MetaAcc
from tqdm import tqdm

from ._utils import (
    assign_to_nearest_lumen,
    centroids,
    centroids_and_border,
    count_points_within_circles,
    filter_lesions_by_size,
    gap_quantile,
    gen_bins,
    get_circles,
    pad_and_shift,
    propagate_points,
    samples,
)
from .acc import assignment_acc, centroids_acc, count_acc

if TYPE_CHECKING:
    from anndata import AnnData
    from anndata.acc import MultiAcc

    from .types import AutoSettings


@dataclass(frozen=True)
class Quantifier:
    """Lumen detection from cell centroids (luminal-area steps 1-3 of the paper's STAR Methods).

    Every distance is in the units of the centroid coordinates. The defaults (and the paper) assume **µm**.

    Algorithm
    ---------
    1. **Empty space.** All centroids are binned onto the *density grid*, a mesh with spacing
       :attr:`interpoint_distance` (5 µm/px by default). The per-pixel cell counts are smoothed with a
       :attr:`blurring_bandwidth`-px box filter, and pixels whose smoothed count is below :attr:`min_density`
       become *empty points*. With :attr:`exclude_outside`, those connected to the grid's edge are dropped.
       Each empty point stands for `interpoint_distance**2` µm² of area.
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

    Defaults
    --------
    The field defaults are those of Jose Reyes's notebook (the code behind :cite:`Reyes2026`):
    :attr:`interpoint_distance` 5, :attr:`blurring_bandwidth` 3, :attr:`min_density` 0.01,
    :attr:`candidate_max_distance` 70, :attr:`rescue_max_distance` 15, :attr:`min_cells_per_lesion` 4,
    :attr:`n_iterations` 2, :attr:`anchor_max_distance` 20, :attr:`lumen_radius` 60, plus the hard-coded 10 px kernel
    and 5 rescue iterations. :attr:`exclude_outside` (`False`) is not in the notebook. They assume µm and
    cell spacing like the paper's data; :meth:`from_adata` derives the spacing-dependent ones from the data instead.

    Divergences
    -----------
    Marked in the source with ``# Divergence from notebook:`` or ``# Divergence from paper:``.

    From the original notebook/code (bugs fixed here, so results differ slightly from the published ones):

    - The y axis of the density grid was sized with the x extent, so its spacing wasn't :attr:`interpoint_distance`.
    - Rescued lumen were counted up to 6 times, because each growth iteration contains the previous ones and all were
      concatenated. This inflated both lumen area and rescued component sizes.
    - Flood fill was seeded at pixel (0, 0) without padding, so it silently failed if dilated border cells covered
      that corner.
    - Coordinates < 1 (and empty points made negative by the half-step shift) wrapped around when cast to `uint32`
      instead of landing on the image edge (crashing, as `uint32`); an empty rescue set crashed.

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
    """Spacing of the density grid in µm, i.e. its µm/px (paper uses "5 µm px⁻¹").

    Used in step 1; one grid step is also how far the rescued lumen grows per iteration in step 3.
    """

    blurring_bandwidth: int = 3
    """Side of the box filter that smooths cell counts, in density-grid pixels (3 px x 5 µm/px = 15 µm).

    The paper describes a Gaussian with a 3 px bandwidth, but the code uses a 3x3 box filter (`cv2.blur`).
    Used in step 1.
    """

    min_density: float = 0.01
    """Smoothed cell count per density-grid pixel (unitless) below which the pixel is empty (paper uses 0.01).

    With a 3x3 box filter, one centroid anywhere in the window gives 1/9,
    so in practice this means "no centroid within ±1 grid pixel (±5 µm)". Used in step 1.
    """

    candidate_max_distance: float = 70
    """Maximum distance (µm) from an anchor for an empty point to be a rescue candidate; caps how far lumen can grow.

    Used in step 3.
    """

    rescue_max_distance: float = 15
    """Distance (µm) from an anchor within which candidates seed the rescued lumen (paper text says 20 µm).

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
    """Distance (µm) below which two border cells belong to the same lesion (paper uses 20 µm).

    Used in step 3.
    """

    lumen_radius: float = 60
    """Radius (µm) of the circle around each cell in which lumen points are counted (paper uses 60 µm, Figure 2H).

    Used in :meth:`count`.
    """

    exclude_outside: bool = False
    """Drop empty points connected to the density grid's edge, i.e. the background outside the tissue.

    Without this, step 3 grows lumen from epithelium on the tissue edge out into the background.
    Lumen opening onto the section edge is dropped too. Not in the paper or the notebook; on in :meth:`from_adata`.
    Used in step 1.
    """

    _from_adata: InitVar[bool] = False

    def __post_init__(self, _from_adata: bool) -> None:
        if not _from_adata:
            warnings.warn(
                "Constructing `Quantifier` directly uses fixed parameters that assume µm and cell spacing like the "
                "paper's; `Quantifier.from_adata` derives them from your data and is more robust.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def from_adata(
        cls,
        adata: AnnData,
        *,
        centroids_acc: MultiAcc = centroids_acc,
        sample_acc: AdRef | None = None,
        quantile: float = 0.95,
        rtol: float = 0.2,
        seed: int = 0,
        **kwargs: Unpack[AutoSettings],
    ) -> Quantifier:
        """Derive the density-grid spacing from cell spacing: one setting for all possible samples.

        A density-grid point is empty when its `blurring_bandwidth * interpoint_distance`-wide window has no
        centroid, i.e. roughly when the gap around it is at least half that. So the grid is sized to put that
        half-width at the `quantile` of gaps (distance to the nearest centroid) *inside tissue*, which makes
        ordinary intercellular gaps rarely count as empty:

        - :attr:`interpoint_distance` = `2 * Q(gap) / blurring_bandwidth`
        - :attr:`rescue_max_distance` = `2 * interpoint_distance`: about one grid row beyond the closest an empty
          point can get to a cell.
        - :attr:`exclude_outside` = `True`.

        Each section is estimated on its own (coordinates of different sections overlap) and the median is used,
        so that every section is measured with the same parameters and lumen area stays comparable across them.
        Everything else keeps its default: parameters derived from epithelial spacing were dominated by isolated
        epithelial cells and unstable across sections.

        Calibrate once and reuse the result: adding sections can shift the median.

        Parameters
        ----------
        adata
            One or more tissue sections.
        centroids_acc
            Accessor for an `(n_obs, 2)` array of cell centroids (µm) (default :data:`~seq_lumen.acc.centroids_acc`).
        sample_acc
            Categorical `obs` column accessor, e.g. `A.obs["sample_id"]`, labelling each cell's section.
            If `None`, `adata` is a single section.
        quantile
            Quantile of the in-tissue gap distribution the emptiness window must exceed.
        rtol
            Warn about sections whose own `interpoint_distance` is more than this fraction away from the median.
        seed
            Seed for sampling the gap distribution.
        kwargs
            Any other fields, overriding the derived ones (`blurring_bandwidth` is used in the derivation).
        """
        xy = centroids(adata, centroids_acc)
        labels = pd.Categorical(np.zeros(len(xy), dtype=int)) if sample_acc is None else samples(adata, sample_acc)
        blur = kwargs.get("blurring_bandwidth", cls.blurring_bandwidth)
        rng = np.random.default_rng(seed)
        per_sample = {s: 2 * gap_quantile(xy[labels == s], quantile, rng) / blur for s in labels.categories}

        interpoint_distance = float(np.median(list(per_sample.values())))
        if off := {s: round(d, 1) for s, d in per_sample.items() if abs(d / interpoint_distance - 1) > rtol}:
            warnings.warn(
                f"interpoint_distance is {interpoint_distance:.1f} (median), but these sections differ by more "
                f"than {rtol:.0%}: {off}. Check them before comparing lumen across sections.",
                UserWarning,
                stacklevel=2,
            )
        derived: AutoSettings = {
            "interpoint_distance": interpoint_distance,
            "rescue_max_distance": 2 * interpoint_distance,
            "exclude_outside": True,
        }
        return cls(**(derived | kwargs), _from_adata=True)

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
        xx, yy = (gen_bins(e, self.interpoint_distance) for e in [x, y])

        # Generate 2d grid from the 1d bins
        xx_base, yy_base = np.meshgrid(xx, yy)

        # Get cell density in each point along each bin
        h = np.histogram2d(x, y, tuple(pad_and_shift(e) for e in [xx, yy]))

        # Blur cellular density estimate and find x,y coordinates that have very few cells (empty regions)
        # Divergence from paper: a box filter (`cv2.blur`), not a Gaussian.
        empty = cv2.blur(h[0].T, (self.blurring_bandwidth, self.blurring_bandwidth)) < self.min_density
        if self.exclude_outside:
            components, _ = sp.ndimage.label(empty)
            outside = np.unique(np.concatenate([components[[0, -1]].ravel(), components[:, [0, -1]].ravel()]))
            empty &= ~np.isin(components, outside[outside > 0])
        empty_bins = np.ravel(empty)
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
        # Divergence from paper: 10 px on the 1 µm/px raster is 10 µm (paper uses 10 px on the 5 µm grid),
        # and it dilates `n_iterations` times and erodes `n_iterations + 1` times (paper uses 1 each).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilation = cv2.dilate(rasterized, kernel, iterations=self.n_iterations)
        del rasterized
        # Flood the background (non-epithelium) connected to the (0, 0) corner.
        # The fill is recorded in `mask` (returned at index 2): 1 = outside, 0 = epithelium or enclosed hole.
        # Divergence from notebook: the image is padded with 1 px of background so the (0, 0) seed is outside the tissue even if dilated
        # epithelium reaches the corner; slicing the mask back keeps its 1-px offset relative to `rasterized`.
        background = np.logical_not(dilation).astype(np.uint8)
        del dilation
        padded = np.pad(background, 1, constant_values=1)
        mask = cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)[2][1:-1, 1:-1]
        del background
        final = np.logical_not(mask).astype(np.uint8)
        del mask
        return cv2.erode(final, kernel, iterations=self.n_iterations + 1)

    def _rescue_points(
        self,
        lumen: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        remaining_empty: np.ndarray,
        border_centroids: np.ndarray,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
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
        `(n + r, 2)` enclosed and rescued lumen points.
        """
        # Anchors: epithelial cells in lesions of >= `min_cells_per_lesion` cells.
        # Note on the paper: the lesion-size filter, `candidate_max_distance` and the 5-iteration cap aren't described.
        anchor_points = filter_lesions_by_size(
            border_centroids, max_distance=self.anchor_max_distance, min_cells_per_lesion=self.min_cells_per_lesion
        )
        candidate_points = propagate_points(anchor_points, remaining_empty, max_distance=self.candidate_max_distance)
        # candidate_points = remaining_empty
        # Divergence from paper: seeds within `rescue_max_distance` = 15 µm (paper uses 20 µm). This is what is done in Jose's notebook.
        rescued = propagate_points(anchor_points, candidate_points, max_distance=self.rescue_max_distance)
        # Grow one density-grid step per iteration: `interpoint_distance + 1` reaches the 4 direct neighbours
        # (5 µm) but not diagonals (~7.1 µm). Max reach ≈ rescue_max_distance + 5 * interpoint_distance = 40 µm.
        # Divergence from notebook: each iteration's result contains the previous one (rescued points are candidates
        # too), so only the last is kept; the notebook concatenated all of them, counting points up to 6 times.
        # It also had no guard for an empty rescue set.
        num_propagate = 5
        if len(rescued) > 0:
            for _ in tqdm(range(num_propagate)):
                rescued = propagate_points(rescued, candidate_points, self.interpoint_distance + 1)
            lumen = np.concatenate([lumen, rescued])

        return lumen

    @overload
    def find(
        self, adata: AnnData, *, centroids_acc: MultiAcc = centroids_acc, border_acc: AdRef, sample_acc: None = None
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]: ...
    @overload
    def find(
        self, adata: AnnData, *, centroids_acc: MultiAcc = centroids_acc, border_acc: AdRef, sample_acc: AdRef
    ) -> dict[str, np.ndarray[tuple[int, int], np.dtype[np.floating]]]: ...
    def find(
        self,
        adata: AnnData,
        *,
        centroids_acc: MultiAcc = centroids_acc,
        border_acc: AdRef,
        sample_acc: AdRef | None = None,
    ) -> (
        np.ndarray[tuple[int, int], np.dtype[np.floating]]
        | dict[str, np.ndarray[tuple[int, int], np.dtype[np.floating]]]
    ):
        """Find luminal points.

        Parameters
        ----------
        adata
            One tissue section, or several if `sample_acc` is given.
        centroids_acc
            Accessor for an `(n_obs, 2)` array of cell centroids (µm for the default parameters, non-negative) (default :data:`~seq_lumen.acc.centroids_acc`).
        border_acc
            Accessor for a boolean `n_obs` mask of the cells lining lumens (epithelial cells).
        sample_acc
            Categorical `obs` column accessor, e.g. `A.obs["sample_id"]`, labelling each cell's section.
            Each section is processed on its own. If `None`, `adata` is a single section.

        Returns
        -------
        `(n, 2)` lumen points: x, y on the density grid (each point = `interpoint_distance**2` µm²).
        With `sample_acc`, a dict from each sample to its lumen points.
        """
        if sample_acc is not None:
            labels = samples(adata, sample_acc)
            return {
                s: self.find(adata[labels == s], centroids_acc=centroids_acc, border_acc=border_acc)
                for s in labels.categories
            }
        centroids, border_ids = centroids_and_border(adata, centroids_acc, border_acc)
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

    @overload
    def count(
        self,
        adata: AnnData,
        lumen: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        *,
        centroids_acc: MultiAcc = centroids_acc,
        count_acc: AdRef,
        sample_acc: None = None,
    ) -> None: ...
    @overload
    def count(
        self,
        adata: AnnData,
        lumen: dict[str, np.ndarray[tuple[int, int], np.dtype[np.floating]]],
        *,
        centroids_acc: MultiAcc = centroids_acc,
        count_acc: AdRef,
        sample_acc: AdRef,
    ) -> None: ...
    def count(
        self,
        adata: AnnData,
        lumen: np.ndarray[tuple[int, int], np.dtype[np.floating]]
        | dict[str, np.ndarray[tuple[int, int], np.dtype[np.floating]]],
        *,
        centroids_acc: MultiAcc = centroids_acc,
        count_acc: AdRef = count_acc,
        sample_acc: AdRef | None = None,
    ) -> None:
        """Count lumen points within :attr:`lumen_radius` of each cell, i.e. the paper's "lumen area".

        The count is in density-grid pixels; multiply by `interpoint_distance**2` for µm².
        Rescued and enclosed lumen points both count.

        Parameters
        ----------
        adata
            The tissue section(s) `lumen` was computed from.
        lumen
            `(n, 2)` lumen points, as returned by :meth:`find`.
            With `sample_acc`, a dict from each sample to its lumen points.
        centroids_acc
            Accessor for the `(n_obs, 2)` array of cell centroids (µm) to center the circles on (default :data:`~seq_lumen.acc.centroids_acc`).
        count_acc
            `obs` column accessor, e.g. `A.obs["lumen_count"]`, to write the `(n_obs,)` counts to.
        sample_acc
            Categorical `obs` column accessor, e.g. `A.obs["sample_id"]`, labelling each cell's section.
            Each cell only counts its own section's lumen. If `None`, `adata` is a single section.
        """
        match count_acc:
            case AdRef(acc=MetaAcc(dim="obs"), idx=str(col)):
                pass
            case _:
                raise TypeError(f"`count_acc` must be an `obs` column accessor like `A.obs['col']`, not {count_acc!r}")
        xy = adata[centroids_acc]
        if sample_acc is None:
            if isinstance(lumen, dict):
                raise TypeError("`lumen` is per-sample, so pass the `sample_acc` it was found with")
            adata.obs[col] = self._count(xy, lumen)
            return
        if not isinstance(lumen, dict):
            raise TypeError("with `sample_acc`, `lumen` must map each sample to its lumen, as from `find`")
        labels = samples(adata, sample_acc)
        lumen_count = np.zeros(adata.n_obs, dtype=int)
        for s in labels.categories:
            mask = labels == s
            lumen_count[mask] = self._count(xy[mask], lumen[s])
        adata.obs[col] = lumen_count

    def _count(self, xy: np.ndarray, lumen: np.ndarray) -> np.ndarray:
        """Lumen points within :attr:`lumen_radius` of each of the `(n, 2)` `xy`."""
        circles = get_circles(xy, self.lumen_radius)
        return count_points_within_circles(lumen, circles)

    @overload
    def assign(
        self,
        adata: AnnData,
        lumen: np.ndarray[tuple[int, int], np.dtype[np.floating]],
        *,
        centroids_acc: MultiAcc = centroids_acc,
        assignment_acc: AdRef,
        sample_acc: None = None,
    ) -> None: ...
    @overload
    def assign(
        self,
        adata: AnnData,
        lumen: dict[str, np.ndarray[tuple[int, int], np.dtype[np.floating]]],
        *,
        centroids_acc: MultiAcc = centroids_acc,
        assignment_acc: AdRef,
        sample_acc: AdRef,
    ) -> None: ...
    def assign(
        self,
        adata: AnnData,
        lumen: np.ndarray[tuple[int, int], np.dtype[np.floating]]
        | dict[str, np.ndarray[tuple[int, int], np.dtype[np.floating]]],
        *,
        centroids_acc: MultiAcc = centroids_acc,
        assignment_acc: AdRef = assignment_acc,
        sample_acc: AdRef | None = None,
    ) -> None:
        """Assign each cell to the lumen nearest it, within :attr:`lumen_radius`.

        A lumen is a connected component of lumen points (8-connected on the density grid).
        Points within 1.5 grid steps are linked: diagonal neighbours (√2 steps) join, points 2 steps apart don't.
        This assumes the grid spacing is ~:attr:`interpoint_distance` (`gen_bins` stretches it slightly),
        which holds for real sections.

        This function is not present in :cite:`Reyes2026`.

        Parameters
        ----------
        adata
            The tissue section(s) `lumen` was computed from.
        lumen
            `(n, 2)` lumen points, as returned by :meth:`find`.
            With `sample_acc`, a dict from each sample to its lumen points.
        centroids_acc
            Accessor for the `(n_obs, 2)` array of cell centroids (µm) (default :data:`~seq_lumen.acc.centroids_acc`).
        assignment_acc
            `obs` column accessor, e.g. `A.obs["lumen_id"]`, to write the `(n_obs,)` lumen ids to.
            Ids are `0..k-1` per section (pair them with the sample column), `-1` for cells with no lumen nearby.
        sample_acc
            Categorical `obs` column accessor, e.g. `A.obs["sample_id"]`, labelling each cell's section.
            Each cell is only assigned to its own section's lumens. If `None`, `adata` is a single section.
        """
        match assignment_acc:
            case AdRef(acc=MetaAcc(dim="obs"), idx=str(col)):
                pass
            case _:
                raise TypeError(
                    f"`assignment_acc` must be an `obs` column accessor like `A.obs['col']`, not {assignment_acc!r}"
                )
        xy = adata[centroids_acc]
        if sample_acc is None:
            if isinstance(lumen, dict):
                raise TypeError("`lumen` is per-sample, so pass the `sample_acc` it was found with")
            adata.obs[col] = self._assign(xy, lumen)
            return
        if not isinstance(lumen, dict):
            raise TypeError("with `sample_acc`, `lumen` must map each sample to its lumen, as from `find`")
        labels = samples(adata, sample_acc)
        lumen_id = np.full(adata.n_obs, -1)
        for s in labels.categories:
            mask = labels == s
            lumen_id[mask] = self._assign(xy[mask], lumen[s])
        adata.obs[col] = lumen_id

    def _assign(self, xy: np.ndarray, lumen: np.ndarray) -> np.ndarray:
        """Lumen id of each of the `(n, 2)` `xy`, `-1` if none within :attr:`lumen_radius`."""
        # 1.5 grid steps links diagonal neighbours (√2 steps) but not points 2 steps apart; assumes the
        # grid spacing is ~`interpoint_distance` (`gen_bins` stretches it slightly), which holds for real sections.
        return assign_to_nearest_lumen(xy, lumen, 1.5 * self.interpoint_distance, self.lumen_radius)

    @overload
    def __call__(
        self,
        adata: AnnData,
        *,
        centroids_acc: MultiAcc = centroids_acc,
        border_acc: AdRef,
        count_acc: AdRef = count_acc,
        assignment_acc: AdRef = assignment_acc,
        sample_acc: None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]: ...
    @overload
    def __call__(
        self,
        adata: AnnData,
        *,
        centroids_acc: MultiAcc = centroids_acc,
        border_acc: AdRef,
        count_acc: AdRef = count_acc,
        assignment_acc: AdRef = assignment_acc,
        sample_acc: AdRef,
    ) -> dict[str, np.ndarray[tuple[int, int], np.dtype[np.floating]]]: ...
    def __call__(
        self,
        adata: AnnData,
        *,
        centroids_acc: MultiAcc = centroids_acc,
        border_acc: AdRef,
        count_acc: AdRef = count_acc,
        assignment_acc: AdRef = assignment_acc,
        sample_acc: AdRef | None = None,
    ) -> (
        np.ndarray[tuple[int, int], np.dtype[np.floating]]
        | dict[str, np.ndarray[tuple[int, int], np.dtype[np.floating]]]
    ):
        """A small wrapper for :meth:`~Quantifier.find` followed by :meth:`~Quantifier.count` and :meth:`~Quantifier.assign`.

        See their docstrings for more information.

        Parameters
        ----------
        adata
            One tissue section, or several if `sample_acc` is given.
        centroids_acc
            Accessor for the `(n_obs, 2)` array of cell centroids (µm) (default :data:`~seq_lumen.acc.centroids_acc`).
        border_acc
            Accessor for a boolean `n_obs` mask of the cells lining lumens (epithelial cells).
        count_acc
            `obs` column accessor to write the `(n_obs,)` counts to (default :data:`~seq_lumen.acc.count_acc`).
        assignment_acc
            `obs` column accessor to write the `(n_obs,)` lumen ids to (default :data:`~seq_lumen.acc.assignment_acc`).
        sample_acc
            Categorical `obs` column accessor, e.g. `A.obs["sample_id"]`, labelling each cell's section.
            If `None`, `adata` is a single section.

        Returns
        -------
        `(n, 2)` lumen points: x, y on the density grid (each point = `interpoint_distance**2` µm²).
        With `sample_acc`, a dict from each sample to its lumen points.
        """
        lumen = self.find(adata, centroids_acc=centroids_acc, border_acc=border_acc, sample_acc=sample_acc)
        # mypy can't tell that `lumen` is a dict exactly when `sample_acc` is given; `count`/`assign` check at runtime.
        self.count(
            adata,
            lumen,  # type: ignore[arg-type]
            centroids_acc=centroids_acc,
            count_acc=count_acc,
            sample_acc=sample_acc,  # type: ignore[arg-type]
        )
        self.assign(
            adata,
            lumen,  # type: ignore[arg-type]
            centroids_acc=centroids_acc,
            assignment_acc=assignment_acc,
            sample_acc=sample_acc,  # type: ignore[arg-type]
        )
        return lumen
