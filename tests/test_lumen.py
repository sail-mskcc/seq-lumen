import anndata as ad
import numpy as np
import pytest
from anndata.acc import A

from seq_lumen import SeqLumen


@pytest.mark.parametrize("closed", [True, False])
def test_gland(closed: bool):
    """A ring of epithelium in stroma: its inside is lumen (enclosed, or rescued if the ring is open), counted once."""
    # Stroma on a 10 µm grid starting at 0 (so tissue touches the raster corner), with a hole of radius 60 µm
    g = np.stack(np.meshgrid(np.arange(0, 400, 10.0), np.arange(0, 400, 10.0)), -1).reshape(-1, 2)
    stroma = g[np.linalg.norm(g - 200, axis=1) >= 60]
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    if not closed:
        theta = theta[theta > 0.8]  # ~40 µm gap, wider than the dilation can bridge
    ring = 200 + 50 * np.stack([np.cos(theta), np.sin(theta)], -1)
    adata = ad.AnnData(obs={"epi": np.r_[np.zeros(len(stroma), bool), np.ones(len(ring), bool)]})
    adata.obsm["spatial"] = np.r_[stroma, ring]

    lumen = SeqLumen().lumen_coords(adata, centroids_acc=A.obsm["spatial"], border_acc=A.obs["epi"])

    assert len(np.unique(lumen[:, :2], axis=0)) == len(lumen)
    found = lumen[(lumen[:, 2] == 0) == closed, :2]
    assert len(found) > 0
    assert np.all(np.linalg.norm(found - 200, axis=1) < 60)
