"""
One fast synthetic end-to-end case per exported ``petra`` entry point.

Every test here drives a public entry point on a small, well-separated,
*deterministically generated* chain and asserts that the true cluster means come
back out **up to a permutation of the source labels** -- which is exactly the
guarantee the package makes and the only one that is invariant under the
relabeling itself.  The match is made with
:func:`scipy.optimize.linear_sum_assignment` on the matrix of distances between
recovered and true means, so a run that merely renames the sources passes while
a run that mixes two clusters together fails.

Everything is sized to keep the whole file fast: the flow-based entry points
evaluate ``flow.log_prob`` once per sample per source label in eager JAX, which
costs roughly half a second per sample, so the chains here are deliberately tiny
(15-24 samples) and the flows are trained for a handful of epochs.  They are
still large enough that the clusters are unambiguous.

Every source of randomness -- the synthetic data, the label shuffling, and the
flow initialization and training -- is seeded.  That is what lets
:func:`test_copula_flows_flat_and_grouped_spellings_are_one_run` compare two
whole runs for bit-equality rather than for approximate agreement.
"""

import os

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

import petra
from petra.posterior_chain import PosteriorChain
from petra.utils import find_prob_in_model

#: Three well-separated 2-D clusters.  The separation (>= 3 units) is many times
#: `CLUSTER_SCALE`, so the labeling problem has an unambiguous answer and a test
#: failure means a real regression rather than an unlucky draw.
TRUE_MEANS = np.array([
    [1.0, 2.0],
    [4.0, 5.0],
    [7.0, 1.5],
])
CLUSTER_SCALE = 0.25


def make_shuffled_chain(n_samples: int, seed: int = 0,
                        scale: float = CLUSTER_SCALE) -> PosteriorChain:
    """
    Build a chain of `n_samples` draws from `TRUE_MEANS` with permuted labels.

    Parameters
    ----------
    n_samples : int
        Number of posterior samples to draw.
    seed : int, default 0
        Seed for both the draws and the per-sample label permutation.
    scale : float, default `CLUSTER_SCALE`
        Standard deviation of each cluster, in every parameter.

    Returns
    -------
    PosteriorChain
        Chain of shape ``(n_samples, 3, 2)`` whose source axis has been
        independently permuted in every sample -- i.e. the label degeneracy the
        ``make_catalog_*`` entry points exist to undo.
    """
    rng = np.random.default_rng(seed)
    n_sources, n_params = TRUE_MEANS.shape
    clean = rng.normal(loc=TRUE_MEANS, scale=scale, size=(n_samples, n_sources, n_params))
    shuffled = np.empty_like(clean)
    for i in range(n_samples):
        shuffled[i] = clean[i, rng.permutation(n_sources), :]
    return PosteriorChain(shuffled, n_sources, n_params)


def recovered_means(chain: np.ndarray) -> np.ndarray:
    """
    Per-slot mean of a chain, ignoring the NaNs of absent sources.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Relabeled chain.

    Returns
    -------
    means : ndarray, shape (n_sources, n_params)
        ``means[i]`` is the mean of the samples sitting in slot `i`.
    """
    return np.stack([np.nanmean(chain[:, i, :], axis=0) for i in range(chain.shape[1])])


def assert_recovers_means(chain: np.ndarray, true_means: np.ndarray = TRUE_MEANS,
                          atol: float = 0.3) -> None:
    """
    Assert that the slot means match `true_means` up to a permutation.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Relabeled chain.
    true_means : ndarray, shape (n_sources, n_params), optional
        Cluster centres the chain was generated from.
    atol : float, default 0.3
        Absolute tolerance per coordinate.  Comfortably above the sampling
        error of the smallest chain used here (0.25 / sqrt(15) ~ 0.065) and far
        below the smallest gap between two clusters (3.0), so it separates
        "recovered" from "mixed" without being flaky.

    Raises
    ------
    AssertionError
        If no permutation of the slots brings every mean within `atol`.
    """
    found = recovered_means(chain)
    distances = np.linalg.norm(found[:, None, :] - true_means[None, :, :], axis=-1)
    slot, truth = linear_sum_assignment(distances)
    assert np.allclose(found[slot], true_means[truth], atol=atol), (
        f"recovered means (best-matched order)\n{found[slot]}\n"
        f"do not match the true means\n{true_means[truth]}\n"
        f"max coordinate error {np.abs(found[slot] - true_means[truth]).max():.3f} > {atol}"
    )


def assert_describes_itself(catalog: PosteriorChain) -> None:
    """
    Assert a catalog's ``prob_in_model`` is derived from the chain it came back with.

    Every chain in this section comes from :func:`make_shuffled_chain`, which
    holds no NaNs at all, so each source is present in every sample and the
    truth is exactly ``1.0`` -- the value the default ``eps=1e-6`` clip would
    report as ``0.999999``.  That is what makes these assertions able to
    distinguish the contract from a near miss, rather than only from ``None``.

    ``tests/test_relabel_loop.py`` pins the same contract one level down, on the
    ``relabel_*`` functions.  This is the ``make_catalog_*`` layer, which wraps
    them in a shuffle, a widening and two initialization passes -- any of which
    could hand back a chain that no longer matches the array travelling with it.

    Parameters
    ----------
    catalog : PosteriorChain
        Chain returned by a ``make_catalog_*`` entry point.

    Raises
    ------
    AssertionError
        If ``prob_in_model`` is not the unclipped occupancy of `catalog`'s own
        samples.
    """
    n_slots = catalog.num_sources
    expected = find_prob_in_model(catalog.get_chain(), n_slots, eps=0)
    assert np.array_equal(catalog.prob_in_model, expected), (
        f"the catalog reports {catalog.prob_in_model!r}, but its chain says {expected!r}"
    )
    assert np.array_equal(catalog.prob_in_model, np.ones(n_slots))
    assert not np.array_equal(catalog.prob_in_model, np.full(n_slots, 1.0 - 1e-6))


# ---------------------------------------------------------------------------
# make_catalog_* entry points
# ---------------------------------------------------------------------------

def test_make_catalog_mv_normal_recovers_clusters():
    """The multivariate-normal relabeler is the baseline every other one builds on."""
    pc = make_shuffled_chain(60, seed=1)
    catalog = petra.make_catalog_mv_normal(
        pc, max_num_sources=3,
        num_iterations=20,
        init_num_iterations=20,
        initialization_param_index=0,
        shuffle_seed=3,
        progress=False,
    )
    assert catalog.chain.shape == (60, 3, 2)
    assert_recovers_means(catalog.chain)
    assert_describes_itself(catalog)


def test_make_catalog_bayesian_gaussian_recovers_clusters():
    """The conjugate-NIW relabeler must reach the same answer as the MV normal one."""
    pc = make_shuffled_chain(60, seed=2)
    catalog = petra.make_catalog_bayesian_gaussian(
        pc, max_num_sources=3,
        num_iterations=5,
        init_num_iterations=20,
        rng_seed=7,
        progress=False,
    )
    assert catalog.chain.shape == (60, 3, 2)
    assert catalog.prob_in_model.shape == (3,)
    assert_recovers_means(catalog.chain)
    assert_describes_itself(catalog)


def test_make_catalog_copula_flows_recovers_clusters():
    """
    The copula-flow relabeler.

    ``coppuccino``'s empirical marginal transform refuses to run on fewer than
    21 samples, so this chain is a little longer than the others.
    """
    pc = make_shuffled_chain(24, seed=5)
    catalog = petra.make_catalog_copula_flows(
        pc, max_num_sources=3,
        num_iterations=1,
        init_num_iterations=20,
        threshold_samples=20,
        max_epochs=20,
        knots=4,
        flow_layers=1,
        max_patience=5,
        rng_seed=7,
        progress=False,
    )
    assert catalog.chain.shape == (24, 3, 2)
    assert_recovers_means(catalog.chain)
    assert_describes_itself(catalog)


# ---------------------------------------------------------------------------
# The grouped-settings API: flat keywords and options objects are one API
# ---------------------------------------------------------------------------

#: Flow settings for the equivalence test, in the spelling
#: :class:`petra.options.CopulaFlowFit` uses for its fields.  *Every* field gets
#: a non-default value: a field that silently failed to be routed would fall
#: back to its default, and a default that happened to match the value under
#: test would hide exactly the bug the test is looking for.
COPULA_FIT_FIELDS = dict(knots=4, flow_layers=1, max_epochs=5, max_patience=2,
                         learning_rate=5e-3, log_prob_floor=-40.0)

#: Keywords shared by every run in the equivalence test.  ``threshold_samples``
#: sits between the two slots' sample counts, so exactly one slot is worth a
#: flow: enough to make the fit settings matter, cheap enough to run three times.
COPULA_SHARED_KWARGS = dict(max_num_sources=2, num_iterations=1,
                            threshold_samples=20, rng_seed=7, progress=False)


def make_one_flow_chain(n_samples: int = 24, n_present: int = 10,
                        seed: int = 5) -> PosteriorChain:
    """
    Build a trans-dimensional chain in which only one slot earns a flow.

    Parameters
    ----------
    n_samples : int, default 24
        Number of posterior samples.  ``coppuccino``'s empirical marginal
        transform refuses to run on fewer than 21.
    n_present : int, default 10
        Samples the second source appears in.  Below ``threshold_samples``, so
        that slot falls back to the uniform prior instead of being fit.
    seed : int, default 5
        Seed for the draws and the per-sample label permutation.

    Returns
    -------
    PosteriorChain
        Chain of shape ``(n_samples, 2, 1)`` with the source axis independently
        permuted in every sample.
    """
    rng = np.random.default_rng(seed)
    chain = np.full((n_samples, 2, 1), np.nan)
    chain[:, 0, 0] = rng.normal(0.0, 0.3, n_samples)
    chain[:n_present, 1, 0] = rng.normal(6.0, 0.3, n_present)
    for i in range(n_samples):
        chain[i] = chain[i, rng.permutation(2), :]
    return PosteriorChain(chain, 2, 1, trans_dimensional=True)


def test_copula_flows_flat_and_grouped_spellings_are_one_run():
    """
    ``knots=4`` and ``flow_fit=CopulaFlowFit(knots=4)`` must be the same call.

    The grouped settings objects collect the flow and initialization keywords on
    :func:`petra.copula_flows.make_catalog_copula_flows`, and the flat spellings
    stay accepted forever.  Two spellings of one setting is only safe while they
    are genuinely one path: if the entry point read a field off the wrong object,
    or forwarded a hardcoded default instead of the resolved one, the flat call
    would keep working and the grouped call would quietly train a different flow.

    The third run is what stops this from being vacuous.  Equality between two
    runs that both ignored their settings is also equality, so one field is
    changed and the answer has to change with it.
    """
    pc = make_one_flow_chain()

    flat = petra.make_catalog_copula_flows(
        pc, init_num_iterations=5, **COPULA_FIT_FIELDS, **COPULA_SHARED_KWARGS)
    grouped = petra.make_catalog_copula_flows(
        pc,
        initialization=petra.Initialization(num_iterations=5),
        flow_fit=petra.CopulaFlowFit(**COPULA_FIT_FIELDS),
        **COPULA_SHARED_KWARGS)

    assert np.array_equal(flat.get_chain(), grouped.get_chain(), equal_nan=True)
    assert np.array_equal(flat.prob_in_model, grouped.prob_in_model)
    assert flat.cost_dict == grouped.cost_dict

    # A flow really was trained, so the settings above had somewhere to land.
    assert flat.cost_dict.keys() == {2}
    assert np.isfinite(flat.cost_dict[2])

    sharper = petra.make_catalog_copula_flows(
        pc,
        initialization=petra.Initialization(num_iterations=5),
        flow_fit=petra.CopulaFlowFit(**{**COPULA_FIT_FIELDS, "knots": 12}),
        **COPULA_SHARED_KWARGS)
    assert sharper.cost_dict[2] != flat.cost_dict[2], (
        "tripling the spline knots changed nothing, so the CopulaFlowFit never "
        "reached the fitter and the equivalence above proves nothing"
    )


def test_copula_flows_initialization_object_is_honoured():
    """
    ``Initialization(with_mv_normal=False)`` must actually skip the pre-relabeling.

    The cheap half of the equivalence check: with ``threshold_samples`` above the
    sample count no flow is trained at all, so this runs in milliseconds and
    isolates the one settings object whose effect is visible without one.  Both
    spellings of "skip it" have to agree with each other and disagree with the
    default, which is what says the object is read rather than defaulted.

    ``initialization_param_index=None`` is on every call because
    ``with_mv_normal`` switches the multivariate pass *only*
    (:func:`petra.initialization.run_initialization_passes`): with the
    univariate pass left on, both runs below would start from the same
    univariately relabeled chain -- for a one-parameter chain that pass is the
    multivariate one -- and the comparison would be measuring the coupling
    between two keywords rather than this object.
    """
    pc = make_one_flow_chain()
    kwargs = dict(max_num_sources=2, num_iterations=1, threshold_samples=10 ** 6,
                  rng_seed=7, initialization_param_index=None, progress=False)

    with_init = petra.make_catalog_copula_flows(
        pc, initialization=petra.Initialization(num_iterations=5), **kwargs)
    grouped_without = petra.make_catalog_copula_flows(
        pc, initialization=petra.Initialization(with_mv_normal=False), **kwargs)
    flat_without = petra.make_catalog_copula_flows(
        pc, init_with_mv_normal=False, **kwargs)

    assert np.array_equal(grouped_without.get_chain(), flat_without.get_chain(),
                          equal_nan=True)
    assert not np.array_equal(with_init.get_chain(), grouped_without.get_chain(),
                              equal_nan=True)


# ---------------------------------------------------------------------------
# The full user journey: load_samples -> make_catalog_* -> to_feather
# ---------------------------------------------------------------------------

def write_ucbmcmc_chains(folder: str, n_samples: int = 40, seed: int = 0) -> None:
    """
    Write a trans-dimensional pair of UCBMCMC chain files under `folder`.

    ``dimension_chain.dat.3`` holds samples with all three clusters present,
    ``dimension_chain.dat.2`` samples with only the first two, and both have
    their source axis shuffled.

    Parameters
    ----------
    folder : str
        Directory to write into; must already exist.
    n_samples : int, default 40
        Number of samples written to each file.
    seed : int, default 0
        Seed for the draws and the shuffling.
    """
    rng = np.random.default_rng(seed)
    n_params = TRUE_MEANS.shape[1]

    for n_sources in (2, 3):
        means = TRUE_MEANS[:n_sources]
        block = rng.normal(loc=means, scale=CLUSTER_SCALE,
                           size=(n_samples, n_sources, n_params))
        for i in range(n_samples):
            block[i] = block[i, rng.permutation(n_sources), :]
        np.savetxt(os.path.join(folder, f"dimension_chain.dat.{n_sources}"),
                   block.reshape(-1, n_params))


def test_load_samples_make_catalog_to_feather_round_trip(tmp_path):
    """
    The advertised journey, start to finish.

    Reads a directory of UCBMCMC chains through the lazily-imported
    ``petra.load_samples``, relabels it, writes the catalog to Feather and reads
    it back, asserting the clusters survive every step.
    """
    write_ucbmcmc_chains(str(tmp_path), n_samples=40, seed=11)

    pc = petra.load_samples(str(tmp_path), num_params_per_source=2)
    assert pc.chain.shape == (80, 3, 2)
    assert pc.trans_dimensional is True
    # The 2-source file contributes a column of NaNs for the missing third slot.
    assert np.isnan(pc.chain).any()

    catalog = petra.make_catalog_mv_normal(
        pc, max_num_sources=3,
        num_iterations=20,
        init_num_iterations=20,
        initialization_param_index=0,
        progress=False,
    )
    assert_recovers_means(catalog.chain)

    path = os.path.join(tmp_path, "catalog.feather")
    catalog.to_feather(path)
    reloaded = PosteriorChain.read_feather(path)

    assert np.array_equal(reloaded.chain, catalog.chain, equal_nan=True)
    assert reloaded.num_sources == 3
    assert reloaded.num_params_per_source == 2
    assert reloaded.trans_dimensional is True
    assert np.array_equal(reloaded.prob_in_model, catalog.prob_in_model, equal_nan=True)
    assert reloaded.cost_dict == catalog.cost_dict
    assert_recovers_means(reloaded.chain)


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_public_surface_contains_only_the_supported_catalog_api():
    assert set(petra.__all__) == {
        "CopulaFlowFit",
        "Initialization",
        "PosteriorChain",
        "load_samples",
        "make_catalog_bayesian_gaussian",
        "make_catalog_copula_flows",
        "make_catalog_mv_normal",
        "__version__",
    }


def test_every_advertised_entry_point_resolves():
    """``petra.__all__`` must not promise a name that ``__getattr__`` cannot supply."""
    for name in petra.__all__:
        assert getattr(petra, name) is not None
    assert set(petra.__all__) <= set(dir(petra))


def test_lazy_getattr_resolves_every_entry_point_to_a_callable():
    """
    ``__getattr__`` must hand back the entry point itself, not its submodule.

    ``__getattr__`` is called directly so the test exercises the lazy resolver
    rather than a value already cached on the package.  The cache it writes is
    undone afterwards so this test cannot change what a later one sees.
    """
    saved = {name: petra.__dict__.get(name) for name in petra._LAZY_ATTRS}
    try:
        for name in petra._LAZY_ATTRS:
            assert callable(petra.__getattr__(name)), f"{name} did not resolve to a callable"
    finally:
        for name, value in saved.items():
            if value is None:
                petra.__dict__.pop(name, None)
            else:
                petra.__dict__[name] = value


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'not_an_entry_point'"):
        petra.not_an_entry_point
