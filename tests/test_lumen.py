import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata.acc import A
from scipy.spatial import KDTree

from seq_lumen import Quantifier


def gland(closed: bool) -> ad.AnnData:
    """A ring of epithelium (radius 50 µm, centered at 200, 200) in stroma."""
    # Stroma on a 10 µm grid starting at 0 (so tissue touches the raster corner), with a hole of radius 60 µm
    g = np.stack(np.meshgrid(np.arange(0, 400, 10.0), np.arange(0, 400, 10.0)), -1).reshape(-1, 2)
    stroma = g[np.linalg.norm(g - 200, axis=1) >= 60]
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    if not closed:
        theta = theta[theta > 0.8]  # ~40 µm gap, wider than the dilation can bridge
    ring = 200 + 50 * np.stack([np.cos(theta), np.sin(theta)], -1)
    adata = ad.AnnData(obs={"epi": np.r_[np.zeros(len(stroma), bool), np.ones(len(ring), bool)]})
    adata.obsm["spatial"] = np.r_[stroma, ring]
    return adata


@pytest.mark.parametrize("closed", [True, False])
def test_gland(closed: bool):
    """A ring of epithelium in stroma: its inside is lumen (enclosed, or rescued if the ring is open), counted once."""
    adata = gland(closed)
    quantifier = Quantifier()
    lumen = quantifier.find(adata, centroids_acc=A.obsm["spatial"], border_acc=A.obs["epi"])

    assert lumen.shape[1] == 2
    assert len(np.unique(lumen, axis=0)) == len(lumen)
    assert len(lumen) > 0
    assert np.all(np.linalg.norm(lumen - 200, axis=1) < 60)

    # Polygon circles are inscribed 64-gons: counts lie between exact counts at the inner and outer radius
    quantifier.count(adata, lumen, centroids_acc=A.obsm["spatial"], lumen_count_acc=A.obs["lumen_count"])
    tree, xy, r = KDTree(lumen), np.asarray(adata.obsm["spatial"]), quantifier.lumen_radius
    count = adata.obs["lumen_count"].to_numpy()
    assert np.all(tree.query_ball_point(xy, r * np.cos(np.pi / 64), return_length=True) <= count)
    assert np.all(count <= tree.query_ball_point(xy, r, return_length=True))
    assert count[adata.obs["epi"].to_numpy()].min() > 0


def test_from():
    """The grid scales with cell spacing; a cohort gets the median, and outlying sections warn."""
    rng = np.random.default_rng(0)
    xy = np.stack(np.meshgrid(np.arange(0, 1000, 10.0), np.arange(0, 1000, 10.0)), -1).reshape(-1, 2)
    xy += rng.normal(0, 2, xy.shape)
    # Three sections on the same 10 µm lattice (coordinates overlap), one at twice the spacing
    adata = ad.AnnData(obs={"sample": pd.Categorical(np.repeat(["a", "b", "c", "d"], len(xy)))})
    adata.obsm["spatial"] = np.r_[xy, xy, xy, 2 * xy]

    single = Quantifier.from_adata(adata[: len(xy)], centroids_acc=A.obsm["spatial"])
    # Gaps in a jittered 10 µm lattice are ~5-7 µm, so the grid lands near the paper's 5 µm
    assert 3 < single.interpoint_distance < 6
    assert single.rescue_max_distance == 2 * single.interpoint_distance

    with pytest.warns(UserWarning, match="'d'"):
        cohort = Quantifier.from_adata(adata, centroids_acc=A.obsm["spatial"], sample_acc=A.obs["sample"])
    assert np.isclose(cohort.interpoint_distance, single.interpoint_distance, rtol=0.05)
    assert Quantifier.from_adata(adata[: len(xy)], centroids_acc=A.obsm["spatial"], n_iterations=5).n_iterations == 5


def test_samples():
    """With `sample_acc`, each section (coordinates overlap) gives the same result as on its own."""
    sections = {"closed": gland(closed=True), "open": gland(closed=False)}
    adata = ad.concat(sections, label="sample", index_unique="-")
    kw = {"centroids_acc": A.obsm["spatial"], "border_acc": A.obs["epi"], "lumen_count_acc": A.obs["lumen_count"]}
    quantifier = Quantifier()

    lumen = quantifier.find_and_count(adata, **kw, sample_acc=A.obs["sample"])
    assert lumen.keys() == sections.keys()
    for name, section in sections.items():
        expected = quantifier.find_and_count(section, **kw)
        np.testing.assert_array_equal(lumen[name], expected)
        in_section = (adata.obs["sample"] == name).to_numpy()
        np.testing.assert_array_equal(adata.obs["lumen_count"].to_numpy()[in_section], section.obs["lumen_count"])

    adata.obs["sample"] = adata.obs["sample"].astype(str)
    with pytest.raises(TypeError, match="categorical"):
        quantifier.find(adata, centroids_acc=A.obsm["spatial"], border_acc=A.obs["epi"], sample_acc=A.obs["sample"])
