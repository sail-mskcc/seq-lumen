import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata.acc import A
from scipy.spatial import KDTree

from seq_lumen import Quantifier, assignment_acc, count_acc


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
    quantifier.count(adata, lumen, centroids_acc=A.obsm["spatial"], count_acc=A.obs["lumen_count"])
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
    kw = {"centroids_acc": A.obsm["spatial"], "border_acc": A.obs["epi"]}
    quantifier = Quantifier()

    lumen = quantifier.run(adata, **kw, sample_acc=A.obs["sample"])
    assert lumen.keys() == sections.keys()
    for name, section in sections.items():
        expected = quantifier.run(section, **kw)
        np.testing.assert_array_equal(lumen[name], expected)
        in_section = (adata.obs["sample"] == name).to_numpy()
        for acc in (count_acc, assignment_acc):
            np.testing.assert_array_equal(adata[acc][in_section], section[acc])

    adata.obs["sample"] = adata.obs["sample"].astype(str)
    with pytest.raises(TypeError, match="categorical"):
        quantifier.find(adata, centroids_acc=A.obsm["spatial"], border_acc=A.obs["epi"], sample_acc=A.obs["sample"])


def test_assign():
    """Two glands side by side: each ring's cells mostly go to its own lumen, far stroma to none.

    Not all: rescue also finds small pockets between ring and stroma, which are lumens of their own.
    """
    left, right = gland(closed=True), gland(closed=True)
    right.obsm["spatial"] = right.obsm["spatial"] + [400, 0]
    adata = ad.concat([left, right], index_unique="-")
    quantifier = Quantifier()
    lumen = quantifier.find(adata, centroids_acc=A.obsm["spatial"], border_acc=A.obs["epi"])
    quantifier.assign(adata, lumen, centroids_acc=A.obsm["spatial"], assignment_acc=A.obs["lumen_id"])

    ids = adata.obs["lumen_id"].to_numpy()
    epi, x = adata.obs["epi"].to_numpy(), adata.obsm["spatial"][:, 0]
    left, right = (np.bincount(ids[epi & side]).argmax() for side in (x < 400, x >= 400))
    assert left != right
    # Lumen lies within 60 µm of a center, so cells beyond that + `lumen_radius` have none
    xy = adata.obsm["spatial"]
    far = np.minimum(*(np.linalg.norm(xy - c, axis=1) for c in ([200, 200], [600, 200]))) > 60 + quantifier.lumen_radius
    assert np.all(ids[far] == -1)


def test_no_lumen_outside_tissue():
    """Epithelium on the edge of an L-shaped section: the empty corner of its bounding box is only lumen by default."""
    g = np.stack(np.meshgrid(np.arange(0, 400, 10.0), np.arange(0, 400, 10.0)), -1).reshape(-1, 2)
    stroma = g[(g[:, 0] < 200) | (g[:, 1] < 200)]
    edge = np.r_[[[200.0, y] for y in range(200, 400, 10)], [[x, 200.0] for x in range(200, 400, 10)]]
    adata = ad.AnnData(obs={"epi": np.r_[np.zeros(len(stroma), bool), np.ones(len(edge), bool)]})
    adata.obsm["spatial"] = np.r_[stroma, edge]
    find = {"centroids_acc": A.obsm["spatial"], "border_acc": A.obs["epi"]}
    assert len(Quantifier().find(adata, **find)) > 0
    assert len(Quantifier(exclude_outside=True).find(adata, **find)) == 0
