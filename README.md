# petra

From the LISA global fit to catalogs of single sources.

A LISA global fit produces posterior samples whose source labels are arbitrary:
"source 0" in one sample need not describe the same object as "source 0" in the
next, and the number of resolved sources can change between samples. `petra`
relabels a trans-dimensional chain so each catalog slot consistently describes
one source.

## Installation

```console
uv add petra-catalogs
```

or:

```console
pip install petra-catalogs
```

Install `petra-catalogs[plots]` for the optional plotting helpers.

The copula-flow method uses `coppuccino`, which supplies its JAX and FlowJAX
training dependencies. Petra has no separate FlowJAX catalog method.

## Chain convention

A `PosteriorChain` stores an array with shape
`(n_samples, n_sources, n_params_per_source)`. It also accepts a flat
`(n_samples, n_sources * n_params_per_source)` array and reshapes it.

An all-`NaN` source row means that source is absent from that posterior sample.
Partial `NaN` rows are invalid. The returned catalog's `prob_in_model` gives the
fraction of samples in which each source is present.

Parameters must be real and finite when a source is present. Raw readers pad
absent sources with `NaN`; a different `fill_value` is rejected because it would
be interpreted as an observed source.

## Quick start

```python
import numpy as np
import petra

rng = np.random.default_rng(42)
n_samples, n_sources, n_params = 200, 3, 2
means = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 1.5]])

chain = rng.normal(
    loc=means,
    scale=0.3,
    size=(n_samples, n_sources, n_params),
)
chain[:50, 2, :] = np.nan
for sample in chain:
    rng.shuffle(sample, axis=0)

posterior = petra.PosteriorChain(
    chain,
    num_sources=n_sources,
    num_params_per_source=n_params,
    trans_dimensional=True,
)
catalog = petra.make_catalog_mv_normal(
    posterior,
    max_num_sources=n_sources,
    num_iterations=20,
    progress=False,
)

print(catalog.shape)                 # (200, 3, 2)
print(catalog.prob_in_model)         # [1.   1.   0.75], up to label order
```

`PosteriorChain.to_feather()` and `PosteriorChain.read_feather()` preserve the
chain and its catalog metadata. `petra.load_samples()` loads the supported raw
text-chain layouts and UCBMCMC output directories.

## Catalog methods

All catalog entry points take a `PosteriorChain` and `max_num_sources`, then
return a relabeled `PosteriorChain`.

| Entry point | Source model | Suggested use |
| --- | --- | --- |
| `make_catalog_mv_normal` | Multivariate Gaussian | Fast default for separated, roughly Gaussian sources. |
| `make_catalog_bayesian_gaussian` | Normal-inverse-Wishart posterior predictive | Regularized Gaussian fit for sparse slots or near-degenerate covariances. |
| `make_catalog_copula_flows` | Copula transform plus normalizing flow, fitted by `coppuccino` | Non-Gaussian dependence with well-behaved marginals. |

The Gaussian methods are inexpensive. Each flow iteration trains a model for
every populated source slot, so use smaller iteration and training budgets when
exploring a dataset.

## Settings objects

Two frozen dataclasses group related options and validate them when they are
constructed:

| Object | Configures | Used by |
| --- | --- | --- |
| `petra.CopulaFlowFit` | Copula-flow construction and training | `make_catalog_copula_flows` |
| `petra.Initialization` | Optional univariate and multivariate Gaussian initialization | `make_catalog_copula_flows` and `make_catalog_bayesian_gaussian` |

```python
from petra import CopulaFlowFit, Initialization

catalog = petra.make_catalog_copula_flows(
    posterior,
    max_num_sources=n_sources,
    num_iterations=2,
    initialization=Initialization(num_iterations=20),
    flow_fit=CopulaFlowFit(
        knots=8, flow_layers=2, max_epochs=20, max_patience=5,
    ),
    progress=False,
)
```

`initialization_param_index=None` skips univariate initialization.
`Initialization(with_mv_normal=False)` skips multivariate initialization. These
switches are independent.

## Changes in 1.1

- Returned `prob_in_model` values are recomputed from the final relabeled chain
  without clipping. The `eps` argument only protects logarithms inside the cost
  matrix.
- Catalog entry points keep the lowest-cost labeling encountered during their
  relabeling loop, including in checkpoints used to resume a run.
- Flow assignment scores use source-exclusion probabilities for absent slots.
- Gaussian covariance regularization respects each parameter's scale, including
  parameters constant within a source. Changing physical units no longer causes
  narrow source populations to collapse together because of an absolute ridge.
- Raw and Feather readers reject malformed row widths before reshaping, and
  Feather serialization supports empty chains. Text layout detection supports
  inline comments.
- Iteration and training counts must be positive integers, learning rates must
  be finite and positive, and probability clipping requires `0 <= eps <= 0.5`.
- Duplicate-value processing rejects non-finite input and uses representable
  increments, avoiding hangs on `NaN` and unresolved duplicates at large scales.
- Ambiguous one-parameter text chains require an explicit reader:
  `petra.samples_io.load_samples_fixed_num_sources()` or
  `petra.samples_io.load_samples_product_space()`. Product-space source counts
  are validated before reading source data.
- `PosteriorChain.cost_dict` is always a dictionary.
- The experimental MCMC flow sampler, source-merging API, and standalone flow
  bijections were removed. This is a breaking API change. Their source and
  retired notebooks are preserved in
  `archive/retired-experiments-2026-09-11.tar.gz` for reference; archived code
  is unsupported and is not installed with the package.
- The standardized-flow catalog method and its `FlowFit` settings were removed.
  Use `make_catalog_copula_flows` with `CopulaFlowFit` for flow-based relabeling;
  all retained flow training runs through `coppuccino`.

## Examples

- `examples/simple_flows_example.py` is a self-checking copula-flow
  clustering example.
- `example_notebooks/toy_problem.ipynb` introduces Gaussian relabeling.
- `example_notebooks/toy_problem_flows.ipynb` runs a small trained copula-flow example.
- `example_notebooks/lisa_example.ipynb` relabels the included LISA sample data.

## Development

```console
uv sync --locked --extra plots
uv run pytest --cov-fail-under=98
uv run mypy
uv run flake8 --select=F,E9 petra/ tests/ examples/
uv run bash .github/scripts/lint_notebooks.sh
uv run python .github/scripts/check_notebook_imports.py
uv run sphinx-build -W -b html docs docs/_build/html
```

CI tests Python 3.11–3.14 on Linux and macOS, verifies the interpreter selected
for each job, and builds distributions on pull requests. It also executes the
Python example and all three maintained notebooks. Notebook execution writes
only temporary copies, leaving the checkout unchanged.
