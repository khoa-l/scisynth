# scisynth

Synthetic scientific data and synthetic observation pipelines.

```python
from scisynth.core import Domain
from scisynth.latent import GaussianField
from scisynth.observer import Observer
from scisynth.samplers import GridSampler
from scisynth.stages import GaussianNoise

domain = Domain.from_extents([(0, 10), (0, 10)])
latent = GaussianField(domain, length_scale=1.5).realize(seed=1)
obs = Observer(GridSampler((64, 64)), [GaussianNoise(0.3)]).run(latent, seed=42)
```

The names are also available directly, as `from scisynth import GridSampler`,
but the subpackages are the preferred path.

## Install

```bash
pip install scisynth                 # numpy + scipy
pip install "scisynth[sklearn]"      # optional: sklearn, polars, xarray
pip install "scisynth[viz]"          # optional: plotting (plotly)
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv build
```

API docs are made with [pdoc](https://pdoc.dev) from the docstrings. They are
generated. There is one page for each subpackage (`core`, `latent`, `samplers`,
`stages`, ...), listing the names it exports:

```bash
uv run python scripts/build_docs.py
open docs/api/index.html
```

Add `--internals` to document every module instead, nested by folder.

The `Docs` workflow (`.github/workflows/docs.yml`) builds the same site on every
push and pull request, and publishes it to GitHub Pages from `main`.

Docs live in `docs/`; runnable examples in `examples/`. The usage tour is a
marimo notebook: `uv run marimo edit examples/example_notebook.py`.
