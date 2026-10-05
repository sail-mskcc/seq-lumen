# seq_lumen

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/ilan-gold/seq_lumen/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/seq_lumen/badge/

A small package for lumen/luminal-structure quantification only from centroids. Based on {cite}`Reyes2026`'s method.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

```python
import seq_lumen
from anndata.acc import A

# Set up relevant accessors for each relevant field.
centroids_acc = A.obsm["X_spatial"]
border_acc = A.obs["is_epithelial"]
# Only if needed
sample_acc = A.obs["sample_id"]
# Resolve the parameters for quantification for all applicable samples, if present.
sl_quantifier = seq_lumen.Quantifier.from_adata(adata, centroids_acc=centroids_acc, sample_acc=sample_acc)
# Each value of `lumen` is a `(n_lumen_points, 2)` array of luminal centroids and the key is the sample id.
# If no `sample_acc` is passed in, just an ndarray is returned.
lumen: dict[str, np.ndarray] = sl_quantifier.run(
    adata, centroids_acc=centroids_acc, border_acc=border_acc, sample_acc=sample_acc
)
# Per-cell lumen scores and lumen ids are written to these default accessors (pass `count_acc`/`assignment_acc` to override)
assert seq_lumen.count_acc in adata
assert seq_lumen.assignment_acc in adata
```

## Divergences from the Original

There are a combination of fixes/divergences from the original code/paper, some bugs, some things being misreported. These are indicated within the code and in the docstring of {class}`seq_lumen.Quantifier`

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].
Choose from the options below to install seq_lumen:

<!--
1. Add the latest release of `seq_lumen` from [PyPI][] to your `uv` project:

   ```bash
   uv add seq_lumen
   ```

1. Install the latest release into a [standard virtual environment][venv]:

   ```bash
   (after activating your venv)
   pip install seq_lumen
   ```

-->

1. Install the latest development version:

   ```bash
   pip install git+https://github.com/ilan-gold/seq_lumen.git  # (or `uv add`)
   ```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

Reyes, J., Del Priore, I., Chaikovsky, A. C., Pasnuri, N., Elhossiny, A. M., Park, J., Weiler, P., Krause, T., Moorman, A., Snopkowski, C., Takizawa, M., Burdziak, C., Ratnayeke, N., Masilionis, I., Ho, Y.-J., Chaligné, R., Romesser, P. B., Filliol, A., Nawy, T., … Lowe, S. W. (2026). Oncogenic and tumor-suppressive forces converge on a progenitor niche at the benign-to-malignant transition. Cell, 189(10), 2875–2897.e53. https://doi.org/10.1016/j.cell.2026.03.032

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/ilan-gold/seq_lumen/issues
[tests]: https://github.com/ilan-gold/seq_lumen/actions/workflows/test.yaml
[documentation]: https://seq_lumen.readthedocs.io
[changelog]: https://seq_lumen.readthedocs.io/page/changelog.html
[api documentation]: https://seq_lumen.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/seq_lumen
[venv]: https://docs.python.org/3/tutorial/venv.html
