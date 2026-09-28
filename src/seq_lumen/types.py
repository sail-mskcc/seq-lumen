from __future__ import annotations

from typing import TypedDict


class AutoSettings(TypedDict, total=False):
    """Overrides accepted by :meth:`~seq_lumen.Quantifier.from_adata`; each maps to a :class:`~seq_lumen.Quantifier` field."""

    #: See :attr:`~seq_lumen.Quantifier.interpoint_distance`.
    interpoint_distance: float
    #: See :attr:`~seq_lumen.Quantifier.blurring_bandwidth`.
    blurring_bandwidth: int
    #: See :attr:`~seq_lumen.Quantifier.min_density`.
    min_density: float
    #: See :attr:`~seq_lumen.Quantifier.candidate_max_distance`.
    candidate_max_distance: float
    #: See :attr:`~seq_lumen.Quantifier.rescue_max_distance`.
    rescue_max_distance: float
    #: See :attr:`~seq_lumen.Quantifier.min_cells_per_lesion`.
    min_cells_per_lesion: int
    #: See :attr:`~seq_lumen.Quantifier.n_iterations`.
    n_iterations: int
    #: See :attr:`~seq_lumen.Quantifier.anchor_max_distance`.
    anchor_max_distance: float
    #: See :attr:`~seq_lumen.Quantifier.lumen_radius`.
    lumen_radius: float


__all__ = ["AutoSettings"]
