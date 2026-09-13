"""
Structural invariants of petra's relabeling and flows.

These are the properties that hold for *every* input, independently of whether
the answer is any good, and each one of them would have caught a bug found in
the code review this file was written for:

(a) relabeling only ever permutes the source axis of a sample.  A relabeler that
    drops, duplicates or invents a source -- or that loses a NaN, i.e. changes
    which sources are present -- is broken no matter how plausible its output
    looks.
(b) a coppuccino-fitted flow returns finite log-densities and its density integrates to
    one.  A flow that quietly returns ``-inf`` or ``NaN`` turns into a constant
    floor in the cost matrix, which makes the Hungarian assignment arbitrary
    while still reporting convergence.
(c) ``make_catalog_mv_normal``'s keyword contract is exactly what it was before
    the other two entry points grouped their settings into
    :mod:`petra.options`.  This entry point takes no options object, so the
    keyword resolver's new habit of consulting option-object field names must
    collapse to a no-op on it -- in both directions, since one of its pinned
    behaviours is a *rejection*.

Everything stochastic here is seeded.
"""

import inspect

import numpy as np
import pytest

from petra.flow_utils import DEFAULT_LOG_PROB_FLOOR, safe_flow_log_prob
from petra.posterior_chain import PosteriorChain
from petra.copula_flows import make_copula_flows_fit


# ---------------------------------------------------------------------------
# (a) Relabeling is a per-sample permutation
# ---------------------------------------------------------------------------

def canonical_rows(sample: np.ndarray) -> np.ndarray:
    """
    Sort the rows of one sample into a labeling-independent order.

    Parameters
    ----------
    sample : ndarray, shape (n_sources, n_params)
        One posterior sample.  NaN rows (absent sources) are allowed; they sort
        to the end, consistently for both arrays being compared.

    Returns
    -------
    ordered : ndarray, shape (n_sources, n_params)
        The same rows, ordered lexicographically by parameter 0, then 1, and so
        on.  Two samples are permutations of one another exactly when their
        `canonical_rows` are equal.
    """
    sample = np.asarray(sample)
    # np.lexsort takes its *last* key as the primary one, hence the reversal.
    return sample[np.lexsort(sample.T[::-1])]


def assert_is_per_sample_permutation(relabeled: np.ndarray, original: np.ndarray) -> None:
    """
    Assert that `relabeled` reorders the source axis of `original`, nothing more.

    Parameters
    ----------
    relabeled : ndarray, shape (n_samples, n_sources, n_params)
        Output of a relabeler.
    original : ndarray, shape (n_samples, n_sources, n_params)
        Its input.

    Raises
    ------
    AssertionError
        If any sample gained, lost or altered a row, or if the number of NaN
        (absent) sources in a sample changed.
    """
    assert relabeled.shape == original.shape
    for i in range(original.shape[0]):
        # The literal check the invariant is usually stated as: the multiset of
        # values in each column is preserved.
        assert np.array_equal(np.sort(relabeled[i], axis=0),
                              np.sort(original[i], axis=0), equal_nan=True), (
            f"sample {i}: column values changed\n{relabeled[i]}\nvs\n{original[i]}")
        # And the stronger one: whole rows are preserved, not just column values.
        assert np.array_equal(canonical_rows(relabeled[i]),
                              canonical_rows(original[i]), equal_nan=True), (
            f"sample {i}: rows are not a permutation\n{relabeled[i]}\nvs\n{original[i]}")
    # Presence/absence of sources is per sample, so check it per sample.
    assert np.array_equal(np.isnan(relabeled).sum(axis=(1, 2)),
                          np.isnan(original).sum(axis=(1, 2)))


@pytest.fixture
def trans_dimensional_chain():
    """
    A shuffled three-cluster chain in which the third source is often absent.

    Returns
    -------
    PosteriorChain
        Chain of shape ``(80, 3, 2)`` with NaN rows for the absent source and an
        independently permuted source axis in every sample.
    """
    rng = np.random.default_rng(0)
    true_means = np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    chain = rng.normal(loc=true_means, scale=0.3, size=(80, 3, 2))
    chain[rng.random(80) < 0.5, 2, :] = np.nan   # the third source comes and goes
    for i in range(80):
        chain[i] = chain[i, rng.permutation(3), :]
    return PosteriorChain(chain, 3, 2, trans_dimensional=True)


def test_mv_normal_relabeling_is_a_permutation(trans_dimensional_chain):
    from petra.make_catalog import make_catalog_mv_normal

    original = trans_dimensional_chain.chain.copy()
    catalog = make_catalog_mv_normal(
        trans_dimensional_chain, max_num_sources=3,
        num_iterations=10, init_num_iterations=10,
        initialization_param_index=0, progress=False,
    )
    assert_is_per_sample_permutation(catalog.chain, original)


def test_bayesian_gaussian_relabeling_is_a_permutation(trans_dimensional_chain):
    from petra.bayesian_gaussian import make_catalog_bayesian_gaussian

    original = trans_dimensional_chain.chain.copy()
    catalog = make_catalog_bayesian_gaussian(
        trans_dimensional_chain, max_num_sources=3,
        num_iterations=5, init_num_iterations=10, progress=False,
    )
    assert_is_per_sample_permutation(catalog.chain, original)


def test_hungarian_relabeling_step_is_a_permutation(trans_dimensional_chain):
    """The single Hungarian step underneath every EM-style relabeler."""
    from petra.aux_distributions import mv_normal_aux_distribution
    from petra.cost_matrix import create_compute_cost_matrix
    from petra.parametric_fits import create_parametric_fit, mv_normal_fit
    from petra.relabel import relabel_samples_one_iteration
    from petra.utils import find_prob_in_model

    chain = trans_dimensional_chain.get_chain()
    aux_parameters = create_parametric_fit(mv_normal_fit)(chain, max_num_sources=3)
    prob_in_model = find_prob_in_model(chain, max_num_sources=3)
    relabeled, _ = relabel_samples_one_iteration(
        chain, aux_parameters, prob_in_model, 3,
        create_compute_cost_matrix(mv_normal_aux_distribution), progress=False,
    )
    assert_is_per_sample_permutation(relabeled, chain)


# ---------------------------------------------------------------------------
# (b) A coppuccino-fitted flow is a probability density
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_flows():
    """
    One 1-D and one 2-D flow, fitted once and shared by the tests below.

    Returns
    -------
    flow_1d : object
        Flow fitted to 120 draws from ``N(0, 1)``.
    flow_2d : object
        Flow fitted to 200 draws from a correlated 2-D normal.
    """
    rng = np.random.default_rng(0)

    chain_1d = rng.normal(size=(120, 1, 1))
    fit_1d = make_copula_flows_fit(chain_1d, rng_seed=7, threshold_samples=10,
                                   max_epochs=3, flow_layers=1, knots=6, progress=False)
    flow_1d = fit_1d(chain_1d, 1)[0]

    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    chain_2d = rng.multivariate_normal([0.0, 0.0], cov, size=200).reshape(200, 1, 2)
    fit_2d = make_copula_flows_fit(chain_2d, rng_seed=7, threshold_samples=10,
                                   max_epochs=3, flow_layers=1, knots=6, progress=False)
    flow_2d = fit_2d(chain_2d, 1)[0]

    return flow_1d, flow_2d


def test_fitted_flow_log_probs_are_finite(fitted_flows):
    flow_1d, flow_2d = fitted_flows

    grid_1d = np.linspace(-4.0, 4.0, 201)[:, None]
    log_probs = np.asarray(flow_1d.log_prob(grid_1d))
    assert log_probs.shape == (201,)
    assert np.all(np.isfinite(log_probs))

    axis = np.linspace(-3.0, 3.0, 31)
    grid_2d = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    log_probs_2d = np.asarray(flow_2d.log_prob(grid_2d))
    assert np.all(np.isfinite(log_probs_2d))


def test_fitted_1d_flow_density_integrates_to_one(fitted_flows):
    """Coarse 1-D quadrature over a range that holds essentially all the mass."""
    flow_1d, _ = fitted_flows
    grid = np.linspace(-8.0, 8.0, 801)
    density = np.exp(np.asarray(flow_1d.log_prob(grid[:, None])))
    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=0.02)


def test_fitted_2d_flow_density_integrates_to_one(fitted_flows):
    """Coarse 2-D quadrature on a 81 x 81 grid."""
    _, flow_2d = fitted_flows
    axis = np.linspace(-6.0, 6.0, 81)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    density = np.exp(np.asarray(flow_2d.log_prob(grid))).reshape(81, 81)
    cell = (axis[1] - axis[0]) ** 2
    assert density.sum() * cell == pytest.approx(1.0, abs=0.05)


def test_safe_flow_log_prob_floors_the_impossible(fitted_flows):
    """
    Whatever a flow does with an absent source, the cost matrix stays finite.

    This is the guard that keeps a ``-inf`` from one flow from making every
    assignment involving it equally impossible.
    """
    flow_1d, _ = fitted_flows
    assert float(safe_flow_log_prob(flow_1d, np.array([np.nan]))) == DEFAULT_LOG_PROB_FLOOR
    batch = np.array([[0.0], [np.nan], [1.0]])
    log_probs = np.asarray(safe_flow_log_prob(flow_1d, batch))
    assert np.all(np.isfinite(log_probs))
    assert log_probs[1] == DEFAULT_LOG_PROB_FLOOR


# ---------------------------------------------------------------------------
# (c) make_catalog_mv_normal's keyword contract survives the options refactor
# ---------------------------------------------------------------------------
#
# The other two entry points now take their flow and initialization
# hyperparameters as the frozen dataclasses of petra.options, and
# `resolve_deprecated_kwargs` grew a line that adds those objects' field names
# to the keywords it considers accepted.  `make_catalog_mv_normal` deliberately
# takes no options object -- it trains nothing, samples nothing, and cannot take
# an `Initialization` without advertising the one alias it has to reject -- so
# that line has to make no difference to it whatsoever.  It is the entry point
# the resolver change was most likely to break, precisely because its contract
# is the asymmetric one: `mv_normal_init` must still be refused and
# `mv_normal_init_iterations` must still be honoured.

#: The full signature, in order.  Pinned because "leave it alone" is the design
#: decision under test: an `initialization` parameter here would silently make
#: `mv_normal_init` applicable again.
MV_NORMAL_PARAMETERS = [
    "posterior_chain", "max_num_sources", "num_iterations", "init_num_iterations",
    "initialization_param_index", "shuffle_seed", "rng_seed", "checkpoint_dir",
    "resume_from", "progress", "eps", "deprecated",
]


@pytest.fixture
def two_source_chain():
    """
    A 12-sample, two-source, one-parameter chain -- just enough to run on.

    Returns
    -------
    PosteriorChain
        Chain of shape ``(12, 2, 1)`` with two well-separated sources.  These
        tests are about argument handling, not about the answer.
    """
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 1)) + np.array([[0.0], [8.0]])
    return PosteriorChain(chain, 2, 1, trans_dimensional=True)


def test_mv_normal_takes_no_options_object():
    """
    Twelve arguments, none of them an options object, and no field keywords.

    `option_field_keywords` returning empty is what makes the resolver change a
    no-op here: the accepted set it computes collapses to the signature alone.
    """
    from petra.make_catalog import make_catalog_mv_normal
    from petra.utils import option_field_keywords

    parameters = inspect.signature(make_catalog_mv_normal).parameters
    assert list(parameters) == MV_NORMAL_PARAMETERS
    assert option_field_keywords(make_catalog_mv_normal) == {}
    # The catch-all is still `**deprecated`: nothing but aliases reaches it.
    assert parameters["deprecated"].kind is inspect.Parameter.VAR_KEYWORD
    # Both budgets keep their own defaults; neither is routed into the other.
    assert parameters["num_iterations"].default == 200
    assert parameters["init_num_iterations"].default == 200


def test_mv_normal_still_rejects_the_initialization_alias(two_source_chain):
    """
    `mv_normal_init` renames to `init_with_mv_normal`, which this entry point
    does not have and must not acquire -- the multivariate normal *is* its
    method.  The resolver now widens its accepted set with the field names of
    every options object in the signature, so an `initialization` parameter here
    would quietly turn this rejection into an acceptance.
    """
    from petra.make_catalog import make_catalog_mv_normal

    with pytest.raises(TypeError, match="not applicable to this entry point"):
        make_catalog_mv_normal(two_source_chain, 2, num_iterations=1,
                               mv_normal_init=True, progress=False)

    # And the keyword the message would otherwise recommend stays unknown, so
    # the advice and the signature cannot disagree.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        make_catalog_mv_normal(two_source_chain, 2, num_iterations=1,
                               init_with_mv_normal=True, progress=False)


def test_mv_normal_still_routes_the_iteration_alias(two_source_chain, monkeypatch):
    """
    `mv_normal_init_iterations` maps onto a real named parameter here, so it
    stays applicable -- and the value has to arrive at the initialization pass,
    not merely be warned about and dropped.
    """
    import petra.make_catalog as make_catalog_module
    from petra.make_catalog import make_catalog_mv_normal

    recorded = {}

    def record(posterior_chain, **kwargs):
        recorded.update(kwargs)
        return posterior_chain

    monkeypatch.setattr(make_catalog_module, "relabel_univariate_normal", record)
    with pytest.warns(DeprecationWarning, match="mv_normal_init_iterations"):
        make_catalog_mv_normal(two_source_chain, 2, num_iterations=1,
                               mv_normal_init_iterations=7, progress=False)

    assert recorded["num_iterations"] == 7


def test_mv_normal_still_rejects_both_spellings_of_the_initialization_budget(two_source_chain):
    """
    `init_num_iterations` is still passed to the resolver as a *current* value,
    which is what makes the alias and its replacement in one call an error.

    On the other two entry points that key moved out of `current` and into the
    options index, where a different branch raises it.  Here it must not have
    moved, and the DeprecationWarning must still fire before the TypeError.
    """
    from petra.make_catalog import make_catalog_mv_normal

    with pytest.warns(DeprecationWarning, match="mv_normal_init_iterations"):
        with pytest.raises(TypeError, match="not both"):
            make_catalog_mv_normal(two_source_chain, 2, num_iterations=1,
                                   init_num_iterations=3, mv_normal_init_iterations=2,
                                   progress=False)


@pytest.mark.parametrize("keyword", ["num_sources", "prob_in_model", "cost_dict"])
def test_a_posterior_chain_field_is_not_a_keyword_of_mv_normal(keyword, two_source_chain):
    """
    :class:`PosteriorChain` is itself a dataclass and is this function's first
    parameter, so an options-object rule of "the annotation is a dataclass"
    would have made every one of its fields an accepted keyword -- and routed it
    into a silent ``replace()`` of the caller's chain.  The ``FLAT_KEYWORDS``
    marker is what prevents that; this is the regression test for it.
    """
    from petra.make_catalog import make_catalog_mv_normal

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        make_catalog_mv_normal(two_source_chain, 2, num_iterations=1, progress=False,
                               **{keyword: None})
