from __future__ import annotations

from anndata.acc import A, AdRef, MultiAcc

centroids_acc: MultiAcc = A.obsm["spatial"]
"""Where :class:`~seq_lumen.Quantifier` reads cell centroids from by default."""

count_acc: AdRef = A.obs["lumen_count"]
"""Where :meth:`~seq_lumen.Quantifier.__call__` writes lumen counts by default."""

assignment_acc: AdRef = A.obs["lumen_id"]
"""Where :meth:`~seq_lumen.Quantifier.__call__` writes lumen ids by default."""

__all__ = ["assignment_acc", "centroids_acc", "count_acc"]
