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
fraction of samples in which each source is present. A NumPy masked array is
also accepted; its masked entries are read as `NaN`, that is, as absent.

The chain is always stored as native-byte-order `float64`. Float16, float32,
longdouble, integer and byte-swapped input is converted on construction, so
every catalog is fitted in double precision and every chain can be written to
Feather.
Only `float64` input is stored without a copy.

The `trans_dimensional` flag is informational metadata. It is stored and saved
with the chain, but no part of petra reads it: presence is decided by `NaN`
rows alone, so the flag changes neither the relabeling nor the catalog.

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
text-chain layouts and UCBMCMC output directories. A UCBMCMC directory splits
one chain by number of sources, so `burn` and `thin` are applied to each
`dimension_chain.dat.<n>` file separately. A per-file burn reweights the
posterior on the number of sources, so load UCBMCMC output with `burn=0`. The
loaded chain is ordered by dimension rather than by iteration, so do not apply a
burn after loading either.

## Catalog methods

All catalog entry points take a `PosteriorChain` and `max_num_sources`, then
return a relabeled `PosteriorChain`.

| Entry point | Source model | Suggested use |
| --- | --- | --- |
| `make_catalog_mv_normal` | Multivariate Gaussian | Fast default for separated, roughly Gaussian sources. |
| `make_catalog_copula_flows` | Copula transform plus normalizing flow, fitted by `coppuccino` | Non-Gaussian dependence with well-behaved marginals. |

The Gaussian method is inexpensive. Each flow iteration trains a model for
every populated source slot, so use smaller iteration and training budgets when
exploring a dataset.

Relabeling is a local search, and a local search can stop in a labeling that
keeps two sources in one label, or that leaves empty the slots
`max_num_sources` adds. `make_catalog_mv_normal` searches past such a stop
with split moves: it looks for a label whose rows fall into two
well-separated clusters, moves the smaller cluster into the sparsest labels in
turn, and keeps a move only if the relabeling that follows lowers the cost. A
move that raises the number of labels in use must lower it further, by the
price the Bayesian information criterion puts on each label it adds. The count
is the net increase, so a move that fills one empty label and empties another
pays nothing, and a cheaper labeling that does not pay that price is turned
down: the catalog is the cheapest labeling the search accepted.
The cost alone would also cut one source with a two-mode posterior in two, so
a label is split only if the chain shows its two clusters together often
enough. A sample shows them together when the label holds a row of one cluster
and another label holds a row like the other. There must be at least a tenth
as many such samples as two independent sources of those sizes would share.
A label whose clusters never appear together is kept whole, because from the
chain alone it cannot be told from one source with two modes, such as the
`psi`/`phi0` pair or the reflected sky position of a galactic binary. Pass
`split_exclusive=True` to split such labels too, when sources that are never
in the same sample should each get a label.
These iterations count against `num_iterations`, up to 10 for each proposal,
and `split_moves=False` gives the plain search. On a chain with many labels
that hold two clusters of rows the search can need several times the
iterations of the plain one. The full LISA example in `example_notebooks`
(2000 samples, 20 slots of 8 parameters) takes the plain search 61 iterations
and about 6 s on an Apple M3 Pro, and the search with split moves 426
iterations and about 34 s; the default `num_iterations=200` stops it partway,
after about 16 s. The search with split moves is over only when a pass
accepts nothing, so whenever the budget ends it first -- before any stall,
inside a pass, or after an accepted move the search has not yet followed to
a stall -- `make_catalog_mv_normal`
says so with a `UserWarning`: the catalog is the best found so far; pass a
larger `num_iterations`, resume a checkpointed run with one, or pass
`split_moves=False` for the plain search, whose budget is a plain ceiling and
which stays silent. The notebook cell of the LISA example, with
`num_iterations=3`, warns for this reason. Sources that
overlap too much to show up as two clusters need another start instead:
`num_starts=k` runs the whole pipeline from `k` shuffles of the chain and
returns the catalog with the lowest `cost_dict[max_num_sources]`. A slot
`max_num_sources` adds that still ends empty is reported with a `UserWarning`.
The other two methods do not take these settings and search as before, so
they can still stop with two sources in one label, or with the slots
`max_num_sources` adds left empty, and they do not warn about it.

## Flow and initialization settings

The copula-flow entry point takes its settings as keyword arguments, each
checked before any work:

| Keywords | Configure | Used by |
| --- | --- | --- |
| `knots`, `flow_layers`, `max_epochs`, `max_patience`, `learning_rate`, `log_prob_floor` | Copula-flow construction and training | `make_catalog_copula_flows` |
| `init_with_mv_normal`, `init_num_iterations` | The optional multivariate Gaussian initialization, and the iteration ceiling of each initialization pass | `make_catalog_copula_flows` |

```python
catalog = petra.make_catalog_copula_flows(
    posterior,
    max_num_sources=n_sources,
    num_iterations=2,
    init_num_iterations=20,
    knots=8,
    flow_layers=2,
    max_epochs=20,
    max_patience=5,
    progress=False,
)
```

`initialization_param_index=None` skips univariate initialization.
`init_with_mv_normal=False` skips multivariate initialization. These switches
are independent.

## Unreleased

These notes compare the development tree with 1.1.

### Removed

- `make_catalog_bayesian_gaussian` and the module `petra.bayesian_gaussian`.
  Its normal-inverse-Wishart prior took its scale from the spread between
  sources, so its predictive was far wider than the posteriors (41x in
  frequency on the bundled LISA example) and it moved rows into worse slots;
  with the prior corrected it reproduced the `make_catalog_mv_normal`
  catalog, so it added nothing.
- The modules `petra.options`, `petra.parametric_fits`,
  `petra.aux_distributions`, `petra.cost_matrix` and `petra.flow_utils`. The
  fits, log-probabilities and the cost matrix now live in `petra.fits`, and
  the flow helpers in `petra.copula_flows`.
- The settings objects `Initialization` and `CopulaFlowFit`, with the
  `initialization=` and `flow_fit=` keywords that took them. Pass their fields
  as keywords instead; see
  [Flow and initialization settings](#flow-and-initialization-settings).
- The keyword aliases `n_phases`, `mv_normal_init` and
  `mv_normal_init_iterations`, which now raise `TypeError`, and
  `PetraDeprecationWarning`.
- The helpers `count_swaps`, `process_array`, `sort_by_number`, `get_logger`
  (use `logging.getLogger("petra")` or `logging.getLogger("petra.<module>")`),
  `fill_missing_indices`, `frequency_bin_width`, `find_uniform_bounds`, and the
  histogram helpers `get_max_count_bin_center`, `tetris_rise_nd`,
  `count_nonnan_samples` and `sort_array_by_sample_count`.
- The relabeling factories and wrappers `create_relabel_samples`,
  `relabel_posterior_chain_one_iteration` (use
  `petra.relabel.relabel_samples_one_iteration`), `create_parametric_fit`,
  `create_compute_cost_matrix` (now the plain function
  `petra.fits.compute_cost_matrix`, bound with
  `functools.partial(compute_cost_matrix, log_prob=...)`) and
  `update_parametric_fit_and_prob_in_model`.

### Renamed and moved

- `uni_normal_fit_single_parameter` is `petra.fits.uni_normal_fit`,
  `mv_normal_aux_distribution` is `petra.fits.mv_normal_log_prob`, and
  `uni_normal_aux_distribution_single_parameter` is
  `petra.fits.uni_normal_log_prob`.
- `find_latest_checkpoint` and `load_checkpoint` moved from `petra.relabel` to
  `petra.checkpoint`. `load_checkpoint` returns a `Checkpoint` instead of a
  `(PosteriorChain, int)` tuple.

### Changed

- Checkpoints use layout version 2. Checkpoints written earlier, and Feather
  files that are not petra checkpoints, are refused with a `ValueError`.
- `run_relabeling_loop` takes `method=` and `settings=`, which a checkpoint
  records and a resume compares. `method` is required when `checkpoint_dir` or
  `resume_from` is given.

## Changes in 1.1

These notes compare 1.1 with 1.0.0, the previous release on PyPI. Changes that
only users of unreleased 1.1 development builds will notice are listed at the
end.

### Added

- Two catalog methods: `make_catalog_bayesian_gaussian`, a
  normal-inverse-Wishart posterior predictive (removed since), and `make_catalog_copula_flows`,
  a copula transform plus a normalizing flow fitted by `coppuccino`. Both are
  exported from `petra`, together with their settings objects
  `petra.Initialization` and `petra.CopulaFlowFit` (removed since; see
  [Unreleased](#unreleased)).
- Checkpoints. Pass `checkpoint_dir` to save the best labeling a run has found,
  and `resume_from` to restart an interrupted run from it; a resumed run keeps
  the lowest-cost labeling across the interruption. Checkpoints record the run
  that wrote them: the method and its settings, `max_num_sources`, `eps` and a
  digest of the input chain. Resuming with a different chain, method, width or
  `eps` raises `ValueError` instead of returning the other run's catalog;
  checkpoints written by unreleased development builds, which carry no such
  record, still resume, with a warning that they cannot be verified. Use one
  `checkpoint_dir` per run: a fresh run refuses a directory that already holds
  checkpoints, and so does a resume that writes anywhere but the directory it
  resumes from, so petra never deletes another run's files. A checkpoint is
  written atomically, only when the best labeling improves, and is named by the
  iteration that found it. Every file is kept by default, about one copy of the
  chain per improving iteration; pass `keep_checkpoints=1` to
  `make_catalog_mv_normal` to keep only the newest, which is all a resume
  needs. Resuming a run that had already stopped, or whose
  `num_iterations` budget is spent (the budget counts checkpointed
  iterations), returns it unchanged with a `UserWarning`. A run with split
  moves whose budget ended the search has not stopped: a resume with a larger
  `num_iterations` repeats the iterations since its newest checkpoint and goes
  on as a longer run would have. `make_catalog_mv_normal`
  does not shuffle, widen or initialize the input chain of a resumed run.
- A `progress` keyword on every entry point. Progress bars are shown by
  default; pass `progress=False` to hide them. Nothing in petra prints any more:
  1.0.0 printed its progress to standard output, and diagnostics now go to the
  `petra` logger.
- `make_catalog_mv_normal` takes `eps` itself (1.0.0 reached it only through
  `relabel_mv_normal`), and `rng_seed`, which it accepts for symmetry with the
  other entry points and does not use.
- Search settings for the multivariate-normal method: `split_moves` and
  `split_exclusive` on `make_catalog_mv_normal` and `relabel_mv_normal`, and
  `num_starts` on `make_catalog_mv_normal`. `num_starts` cannot be combined with
  `checkpoint_dir` or `resume_from`. See [Catalog methods](#catalog-methods).
- Feather files also store `prob_in_model` and `cost_dict`. See the
  `PosteriorChain` entry under [Changed](#changed) for the new file format.
- Type information. petra ships a PEP 561 `py.typed` marker, so type checkers
  read its annotations, including the real signatures of the lazily imported
  entry points. An error message that names a file shows the path itself, also
  when it was passed as a `pathlib.Path`, rather than a `PosixPath(...)`.

### Changed

These change what an unchanged 1.0.0 call does.

- Every argument of `make_catalog_mv_normal` after `max_num_sources` is
  keyword-only, as on the other entry points. A positional 1.0.0 call such as
  `make_catalog_mv_normal(pc, 4, 100, 50, 2, 42)` raises `TypeError`; pass the
  same values by name: `num_iterations=100, init_num_iterations=50,
  initialization_param_index=2, shuffle_seed=42`.
- `initialization_param_index` defaults to `0` instead of `None`, so a call
  that does not pass it now runs the univariate initialization on parameter 0
  before the multivariate relabeling. Pass `initialization_param_index=None` to
  skip it, as 1.0.0 did by default.
- The `eps` default is `1e-6` instead of `1e-2`, and `eps` no longer reaches
  the returned `prob_in_model`. It only keeps the inclusion probabilities used
  inside the cost matrix within `[eps, 1 - eps]`, so that the logarithms there
  stay finite. The returned `prob_in_model` is recomputed from the final
  relabeled chain without clipping. Pass `eps=1e-2` to get 1.0.0's cost matrix;
  the returned probabilities are unclipped either way.
- Entries are shuffled only when `shuffle_seed` is given. 1.0.0 shuffled the
  source entries of every sample on every call, using an unseeded generator
  when `shuffle_seed` was `None`, so the same call could return different
  catalogs. Pass a `shuffle_seed` to shuffle as 1.0.0 did, but reproducibly.
- The relabeling loop returns the lowest-cost labeling it accepts, and
  stops as soon as an iteration fails to lower the assignment cost. 1.0.0
  stopped only when the cost repeated exactly, and otherwise returned the
  labeling of the last iteration.
- `make_catalog_mv_normal` and `relabel_mv_normal` try split moves before they
  stop (see [Catalog methods](#catalog-methods)). A chain whose sources the
  plain search kept merged, or one widened by `max_num_sources`, can therefore
  return a different and cheaper catalog than in 1.0.0. `make_catalog_mv_normal`
  also widens the chain before shuffling it, so with both a `shuffle_seed` and
  a `max_num_sources` above the chain's width it starts from a different
  labeling. Pass `split_moves=False` for the plain search without split
  moves; it does not undo the widening before shuffling.
- Gaussian fits add a small ridge to every covariance, scaled to each
  parameter's own variance. A parameter that is constant within a source no
  longer stops the run with `LinAlgError: Singular matrix`, as it did in 1.0.0,
  and the regularization does not depend on the parameters' physical units.
  A source slot whose rows mostly repeat a few states, as a stuck sampler
  writes them, keeps its own mean but takes the covariance pooled over the
  whole chain, with a `UserWarning`; its own covariance would be nearly flat
  and split that source across two catalog entries. A slot that holds its
  few states in more than half of the samples is a source held fixed, not a
  stuck stretch, and keeps its own narrow fit.
- `load_samples` applies `burn` and `thin` to product-space files too; 1.0.0
  ignored them for that layout.
- `PosteriorChain` rejects a source row that mixes `NaN` with finite values,
  and the raw readers reject a `fill_value` other than `NaN`. 1.0.0 accepted
  both, although only an all-`NaN` row marks an absent source.
- `PosteriorChain` stores every chain as native `float64`, so float32 and
  integer input now produce `float64` catalogs, and masked arrays keep their
  mask as `NaN`. Feather files keep the chain's description in their schema
  metadata as strict JSON rather than in extra columns: a `cost_dict` key or
  value that would not read back unchanged, such as the key `3.7`, a bool or
  array value, or a `NaN` or infinite cost, raises `ValueError` instead of being
  coerced. Files written by 1.0.0 still read, with `prob_in_model=None`; 1.0.0
  cannot read files written by 1.1.
- Installing petra also installs `pyarrow`, `tqdm`, `jax` and `coppuccino`, on
  top of 1.0.0's `numpy`, `scipy` and `pandas`. Only the copula-flow method
  imports `jax`; `import petra` and the Gaussian methods do not.

### Stricter checks

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
- The text and UCBMCMC readers require `burn` to be an integer `>= 0` and
  `thin` an integer `>= 1`, so `None`, bools and negative values raise
  `ValueError`. A `burn` that leaves no samples raises instead of returning an
  empty chain. A product-space file whose kept samples all have zero sources
  raises a `ValueError` that names the file. Feather, `.npy` and gzip files
  raise a `ValueError` naming the right reader. Text is read as UTF-8, with or
  without a byte-order mark, and a file that is not UTF-8 raises a `ValueError`
  that names it.
- The catalog entry points and relabelers check their leading arguments before
  any work. `posterior_chain` must be a `PosteriorChain` (a bare array raises
  `TypeError`) holding at least one sample; `max_num_sources` must be a
  positive integer, not a float such as `3.0` or a bool, of at least the
  chain's width; and `shuffle_seed` must be `None` or a non-negative integer,
  so bools, floats, sequences and `numpy.random.Generator` objects raise
  `TypeError`. `PosteriorChain.get_chain` and `get_valid_chain_entry` check
  `burn` and `thin` as the readers do, and `expand_chain` refuses a width that
  is not an integer or would drop a source.
- UCBMCMC directories: an empty `dimension_chain.dat.<n>` file is skipped with
  a warning instead of making the directory unreadable. A dimension file that
  `burn` empties is named in a warning, and its source slots are kept, so `burn`
  no longer narrows the chain. The directory name is taken literally rather than
  as a glob pattern. A missing directory raises `FileNotFoundError`, and stray
  names such as `dimension_chain.dat.2.bak` are skipped with a warning.

### Changes from unreleased development builds

These matter only to scripts written against unreleased 1.1 development code,
not to users of 1.0.0.

- Flow assignment scores use source-exclusion probabilities for absent slots.
- The covariance ridge is relative to each parameter's variance rather than
  absolute, so changing physical units no longer causes narrow source
  populations to collapse together.
- `PosteriorChain.cost_dict` is always a dictionary.
- The experimental MCMC flow sampler, source-merging API, and standalone flow
  bijections were removed. This is a breaking API change. Their source and
  retired notebooks are preserved in
  `archive/retired-experiments-2026-09-11.tar.gz` for reference; archived code
  is unsupported and is not installed with the package.
- The standardized-flow catalog method and its `FlowFit` settings were removed.
  Use `make_catalog_copula_flows` for flow-based relabeling;
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
uv run pytest --cov-fail-under=90
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
