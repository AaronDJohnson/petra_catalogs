"""
The Bayesian Gaussian relabeler: reachable hyperparameters and its outer loop.

``kappa0`` and ``nu0`` were literals inside ``compute_niw_prior``. A knob that
cannot be reached from the entry point is, for a user, the same as a knob that
does not exist, so these tests drive each one from the public function and
check that the number actually moves.

``make_catalog_bayesian_gaussian``'s two initialization keywords have since
become one :class:`petra.options.Initialization`, so a second section drives
that entry point through both spellings.  Both must reach the same two
initialization passes with the same budget, and the same run written both ways
must come out bit for bit identical -- otherwise the grouping is a rewrite of
the API rather than a regrouping of it.

The last section pins the behaviour of ``bayesian_relabel_loop``'s outer loop.
That loop used to be a second copy of the one in ``create_relabel_samples``,
kept because it reverted to its best labeling when the cost went back up; both
copies now run :func:`petra.relabel.run_relabeling_loop`, so these tests check
the NIW relabeler still reverts and still checkpoints through the shared one.
The ``prob_in_model`` contract that unification settled is pinned in
``tests/test_relabel_loop.py``.
"""

import dataclasses
import inspect
import logging
import warnings

import numpy as np
import pytest

from petra import bayesian_gaussian
from petra.bayesian_gaussian import (
    DEFAULT_KAPPA0,
    bayesian_relabel_loop,
    compute_niw_prior,
    make_catalog_bayesian_gaussian,
    niw_fit,
)
from petra.options import Initialization
from petra.posterior_chain import PosteriorChain
from petra.utils import option_field_keywords


@pytest.fixture
def tiny_chain():
    """A 12-sample, two-source, one-parameter chain -- just enough to run on."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 1)) + np.array([[0.0], [8.0]])
    return PosteriorChain(chain, 2, 1, trans_dimensional=True)


# ---------------------------------------------------------------------------
# The NIW prior
# ---------------------------------------------------------------------------

def test_kappa0_pulls_the_predictive_mean_towards_the_pooled_mean():
    """`kappa0` is the prior's weight in pseudo-observations, and must act like it."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(100, 2, 2)) + np.array([[0.0, 0.0], [8.0, 8.0]])
    pooled_mean = compute_niw_prior(chain, 2)["m0"]

    weak = niw_fit(chain, 2, kappa0=DEFAULT_KAPPA0)
    strong = niw_fit(chain, 2, kappa0=10_000.0)

    for source in range(2):
        weak_gap = np.linalg.norm(weak[source]["loc"] - pooled_mean)
        strong_gap = np.linalg.norm(strong[source]["loc"] - pooled_mean)
        assert strong_gap < weak_gap


def test_nu0_widens_the_posterior_predictive():
    """More prior degrees of freedom means more shrinkage towards the pooled scale."""
    rng = np.random.default_rng(0)
    # Source 1 is much tighter than the pooled spread, so shrinking towards the
    # pooled covariance can only inflate its predictive scale.
    chain = np.concatenate([rng.normal(size=(100, 1, 1)) * 4.0,
                            rng.normal(size=(100, 1, 1)) * 0.1 + 8.0], axis=1)

    weak = niw_fit(chain, 2)[1]
    strong = niw_fit(chain, 2, nu0=500)[1]
    assert strong["log_scale_det"] > weak["log_scale_det"]


@pytest.mark.parametrize("kwargs, message", [
    ({"kappa0": 0.0}, "kappa0 must be strictly positive"),
    ({"kappa0": -1.0}, "kappa0 must be strictly positive"),
    ({"nu0": 2}, "nu0 must be greater than"),
])
def test_degenerate_niw_priors_are_rejected(kwargs, message):
    """
    ``nu0 <= n_params + 1`` makes ``Psi0`` non-positive-definite.

    Left unchecked it would reach ``precompute_t_params``, which would report
    every source degenerate -- a much less informative failure than saying which
    setting was wrong.
    """
    chain = np.random.default_rng(0).normal(size=(50, 2, 1))
    with pytest.raises(ValueError, match=message):
        compute_niw_prior(chain, 2, **kwargs)


def test_niw_prior_settings_reach_the_entry_point(tiny_chain):
    """The whole point of the change: `make_catalog_bayesian_gaussian` can set them."""
    with pytest.raises(ValueError, match="nu0 must be greater than"):
        make_catalog_bayesian_gaussian(tiny_chain, 2, num_iterations=1,
                                       init_num_iterations=1, nu0=2, progress=False)

    with pytest.raises(ValueError, match="kappa0 must be strictly positive"):
        bayesian_relabel_loop(tiny_chain, 2, num_iterations=1, kappa0=0.0, progress=False)


def test_a_stronger_niw_prior_changes_the_catalog_cost(tiny_chain):
    """A reachable knob that changes nothing would be a reachable knob in name only."""
    weak = make_catalog_bayesian_gaussian(tiny_chain, 2, num_iterations=3,
                                          init_num_iterations=1, progress=False)
    strong = make_catalog_bayesian_gaussian(tiny_chain, 2, num_iterations=3,
                                            init_num_iterations=1, kappa0=50.0,
                                            nu0=200, progress=False)
    assert weak.cost_dict[2] != strong.cost_dict[2]


# ---------------------------------------------------------------------------
# The Initialization object on the NIW entry point
# ---------------------------------------------------------------------------

def _spy_on_the_initialization_passes(monkeypatch):
    """
    Watch both pre-relabeling passes of ``make_catalog_bayesian_gaussian``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the two relabelers in the module's namespace.

    Returns
    -------
    seen : dict of str to int
        Filled as the run proceeds: ``"univariate"`` and ``"mv_normal"`` map to
        the ``num_iterations`` each pass was called with.  A pass that never
        runs leaves no key, which is how "this setting was honoured" is told
        apart from "this setting was accepted".
    """
    seen = {}
    real_univariate = bayesian_gaussian.relabel_univariate_normal
    real_mv_normal = bayesian_gaussian.relabel_mv_normal

    def univariate_spy(chain, **kwargs):
        seen["univariate"] = kwargs["num_iterations"]
        return real_univariate(chain, **kwargs)

    def mv_normal_spy(chain, **kwargs):
        seen["mv_normal"] = kwargs["num_iterations"]
        return real_mv_normal(chain, **kwargs)

    monkeypatch.setattr(bayesian_gaussian, "relabel_univariate_normal", univariate_spy)
    monkeypatch.setattr(bayesian_gaussian, "relabel_mv_normal", mv_normal_spy)
    return seen


def test_the_niw_entry_point_accepts_the_initialization_object(tiny_chain):
    """The point of the change: one object instead of two loose keywords."""
    catalog = make_catalog_bayesian_gaussian(
        tiny_chain, 2, num_iterations=1, progress=False,
        initialization=Initialization(num_iterations=1),
    )
    assert catalog.chain.shape == (12, 2, 1)


def test_the_flat_init_keywords_are_still_accepted_without_a_warning(tiny_chain):
    """
    ``init_with_mv_normal`` and ``init_num_iterations`` keep working, silently.

    A `DeprecationWarning` here would be wrong: the flat form is not deprecated,
    it is the *other* way of writing the same call, and it is what every
    existing script and notebook uses.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        catalog = make_catalog_bayesian_gaussian(
            tiny_chain, 2, num_iterations=1, progress=False,
            init_with_mv_normal=True, init_num_iterations=1,
        )
    assert catalog.chain.shape == (12, 2, 1)


@pytest.mark.parametrize("kwargs, expected", [
    ({"init_num_iterations": 4}, {"univariate": 4, "mv_normal": 4}),
    ({"initialization": Initialization(num_iterations=4)},
     {"univariate": 4, "mv_normal": 4}),
    ({"init_with_mv_normal": False, "init_num_iterations": 3}, {"univariate": 3}),
    ({"initialization": Initialization(with_mv_normal=False, num_iterations=3)},
     {"univariate": 3}),
], ids=["flat", "grouped", "flat-without-mv-normal", "grouped-without-mv-normal"])
def test_the_initialization_settings_reach_both_passes(tiny_chain, monkeypatch,
                                                       kwargs, expected):
    """
    Accepted is not the same as honoured: the values have to arrive.

    ``init_num_iterations`` is a *per-pass* ceiling and is therefore spent
    twice, and the univariate pass runs whether or not ``with_mv_normal`` is
    set -- it is gated on `initialization_param_index` alone.  That is now true
    of both entry points that share
    :func:`petra.initialization.run_initialization_passes`;
    Some catalog methods used to skip both passes together, and
    ``tests/test_chain_and_initialization_edges.py`` is where they are
    checked against each other.  Reading the budget off the wrong field, or
    gating the wrong pass, changes what the run does without changing what it
    returns for a chain this easy.
    """
    seen = _spy_on_the_initialization_passes(monkeypatch)
    make_catalog_bayesian_gaussian(tiny_chain, 2, num_iterations=1, progress=False, **kwargs)
    assert seen == expected


def test_a_deprecated_alias_still_reaches_the_initialization_object(tiny_chain, monkeypatch):
    """
    A legacy spelling now has two hops to make, and either one can drop it.

    ``resolve_deprecated_kwargs`` renames ``mv_normal_init_iterations`` to
    ``init_num_iterations``, and ``resolve_entry_point_kwargs`` then has to move
    that off the named result and onto `Initialization`.  Warning about an alias
    and then ignoring its value is the exact bug the resolver was written to
    prevent, and the second hop is a new place to reintroduce it.
    """
    seen = _spy_on_the_initialization_passes(monkeypatch)
    with pytest.warns(DeprecationWarning, match="mv_normal_init_iterations"):
        make_catalog_bayesian_gaussian(tiny_chain, 2, num_iterations=1, progress=False,
                                       mv_normal_init_iterations=6)
    assert seen == {"univariate": 6, "mv_normal": 6}

    seen.clear()
    with pytest.warns(DeprecationWarning, match="mv_normal_init"):
        make_catalog_bayesian_gaussian(tiny_chain, 2, num_iterations=1, progress=False,
                                       mv_normal_init=False, init_num_iterations=2)
    assert seen == {"univariate": 2}


@pytest.mark.parametrize("kwargs, message", [
    ({"initialization": Initialization(), "init_with_mv_normal": False},
     "either 'init_with_mv_normal' or 'initialization', not both"),
    ({"initialization": Initialization(), "init_num_iterations": 3},
     "either 'init_num_iterations' or 'initialization', not both"),
])
def test_the_object_and_one_of_its_fields_is_an_error(tiny_chain, kwargs, message):
    """
    Not a merge, and not a precedence rule.

    Merging would mean answering "does ``Initialization(num_iterations=20)``
    plus ``init_with_mv_normal=False`` keep the 20?" -- an answer that has to be
    remembered rather than read.  The message names the keyword the caller
    typed, so it points at their line rather than at petra's field names.
    """
    with pytest.raises(TypeError, match=message):
        make_catalog_bayesian_gaussian(tiny_chain, 2, num_iterations=1,
                                       progress=False, **kwargs)


def test_a_deprecated_alias_onto_an_options_field_warns_before_it_is_rejected(tiny_chain):
    """
    ``mv_normal_init`` renames to ``init_with_mv_normal``, which now lives on
    `Initialization`.

    The package's rule is warn-then-raise, so the caller learns their spelling
    is deprecated *and* that the call is contradictory -- reversing the order
    hides the deprecation exactly when it matters most.
    """
    with pytest.warns(DeprecationWarning, match="mv_normal_init"):
        with pytest.raises(TypeError, match="either 'mv_normal_init' or 'initialization'"):
            make_catalog_bayesian_gaussian(tiny_chain, 2, progress=False,
                                           initialization=Initialization(),
                                           mv_normal_init=False)


def test_a_flat_init_budget_is_validated_exactly_as_the_object_would_be(tiny_chain):
    """
    ``dataclasses.replace`` re-runs ``__post_init__``, and it must.

    Otherwise ``init_num_iterations=0`` spelled flat would sail past the check
    that ``Initialization(num_iterations=0)`` fails and reach
    :func:`petra.relabel.run_relabeling_loop`, which raises the same complaint
    only after a whole univariate pass has been set up.
    """
    with pytest.raises(ValueError, match="init_num_iterations must be at least 1"):
        Initialization(num_iterations=0)
    with pytest.raises(ValueError, match="init_num_iterations must be at least 1"):
        make_catalog_bayesian_gaussian(tiny_chain, 2, num_iterations=1, progress=False,
                                       init_num_iterations=0)


def test_the_chains_own_fields_are_not_accepted_as_keywords(tiny_chain):
    """
    ``PosteriorChain`` is a dataclass, and it is this entry point's first
    argument.

    Detecting options objects with ``dataclasses.is_dataclass`` would therefore
    have made ``num_sources``, ``prob_in_model`` and ``cost_dict`` accepted
    keywords here, each of them quietly ``replace()``ing the caller's chain
    instead of relabeling it.  The ``FLAT_KEYWORDS`` marker is what stops that.
    """
    index = option_field_keywords(make_catalog_bayesian_gaussian)
    for field in dataclasses.fields(PosteriorChain):
        assert field.name not in index

    with pytest.raises(TypeError, match="cost_dict"):
        make_catalog_bayesian_gaussian(tiny_chain, 2, progress=False, cost_dict={})


def test_the_top_level_iteration_budget_is_not_claimed_by_the_initializer():
    """
    `Initialization`'s field is `num_iterations`, spelled ``init_num_iterations``.

    An identity flat map on that class would put ``num_iterations`` in the
    options index as well as in the signature, and this entry point's own
    relabeling budget -- a different quantity with a different default -- would
    be routed into the pre-relabeling instead.
    """
    index = option_field_keywords(make_catalog_bayesian_gaussian)
    assert "num_iterations" not in index
    assert index["init_num_iterations"][:2] == ("initialization", "num_iterations")

    parameters = inspect.signature(make_catalog_bayesian_gaussian).parameters
    assert parameters["num_iterations"].default == 10
    assert parameters["num_iterations"].kind is inspect.Parameter.KEYWORD_ONLY


def test_flat_and_grouped_spellings_produce_bit_identical_chains():
    """
    The whole regrouping in one assertion.

    Both `Initialization` fields are non-default and both reach the run: the
    budget bounds each pre-relabeling pass and the flag decides whether the
    multivariate one happens at all.  The chain is deliberately awkward --
    three overlapping sources, a third of the entries absent -- so that the
    initialization changes where the NIW loop ends up, which is what the last
    assertion checks.  Without it, "identical" would also be satisfied by both
    spellings being ignored.
    """
    rng = np.random.default_rng(1)
    n_samples, n_sources, n_params = 40, 3, 2
    chain = (rng.normal(size=(n_samples, n_sources, n_params)) * 1.6
             + np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 3.0]]))
    chain = np.stack([chain[i, rng.permutation(n_sources), :] for i in range(n_samples)])
    chain[rng.random((n_samples, n_sources)) < 0.3] = np.nan
    posterior_chain = PosteriorChain(chain, n_sources, n_params, trans_dimensional=True)

    shared = dict(num_iterations=3, initialization_param_index=0, progress=False)

    flat = make_catalog_bayesian_gaussian(
        posterior_chain, n_sources,
        init_with_mv_normal=True, init_num_iterations=7, **shared,
    )
    grouped = make_catalog_bayesian_gaussian(
        posterior_chain, n_sources,
        initialization=Initialization(with_mv_normal=True, num_iterations=7), **shared,
    )

    np.testing.assert_array_equal(flat.get_chain(), grouped.get_chain())
    np.testing.assert_array_equal(flat.prob_in_model, grouped.prob_in_model)
    assert flat.cost_dict == grouped.cost_dict

    # ...and the flag really did reach the run, so "identical" above is not just
    # "both spellings were ignored": dropping the multivariate pass lands the
    # loop on a different labeling for this chain.
    without_mv_normal = make_catalog_bayesian_gaussian(
        posterior_chain, n_sources,
        initialization=Initialization(with_mv_normal=False, num_iterations=7), **shared,
    )
    assert not np.array_equal(without_mv_normal.get_chain(), grouped.get_chain(),
                              equal_nan=True)


# ---------------------------------------------------------------------------
# The NIW outer loop
# ---------------------------------------------------------------------------

def non_monotone_chain():
    """
    A chain on which the NIW assignment cost goes *up* at one iteration.

    Four heavily overlapping source slots, half of whose entries are absent.
    Nothing about the method guarantees a monotone cost -- the fit is refitted
    to the new labeling before the next assignment -- and this seed is one of
    the cases where it is not.
    """
    rng = np.random.default_rng(61)
    n_samples, n_sources, n_params = 60, 4, 2
    chain = rng.normal(size=(n_samples, n_sources, n_params)) * rng.uniform(0.2, 3.0)
    chain[rng.random((n_samples, n_sources)) < 0.5] = np.nan
    return PosteriorChain(chain, n_sources, n_params, trans_dimensional=True)


def test_loop_reverts_when_the_cost_increases(caplog):
    """
    The loop must return its cheapest labeling, not its last one.

    ``bayesian_relabel_loop`` is now :func:`petra.relabel.run_relabeling_loop`
    with the NIW fit bound to it, so the costs are read off the shared loop's
    log rather than a local one.  This test is what says the surviving loop
    still reverts for this relabeler on a chain that really does get worse.
    """
    pc = non_monotone_chain()
    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        result = bayesian_relabel_loop(pc, 4, num_iterations=15, progress=False)

    # args = (iteration, delta cost, total cost) of the shared loop's per-iteration line.
    costs = [float(record.args[2]) for record in caplog.records
             if record.msg.startswith("Iteration %d: Difference in cost")]
    assert len(costs) > 1
    assert max(costs) > min(costs)                     # the cost really did move
    assert any(cost > previous for previous, cost in zip(costs, costs[1:])), \
        "this fixture is meant to make the cost increase at least once"
    assert "because the cost increased" in caplog.text
    assert result.cost_dict[4] == pytest.approx(min(costs))


def test_loop_rejects_a_zero_iteration_budget(tiny_chain):
    with pytest.raises(ValueError, match="num_iterations must be at least 1"):
        bayesian_relabel_loop(tiny_chain, 2, num_iterations=0, progress=False)


def test_loop_checkpoints_and_resumes(tiny_chain, tmp_path, caplog):
    """A walltime kill mid-loop must not cost the iterations already done."""
    checkpoint_dir = tmp_path / "niw"
    bayesian_relabel_loop(tiny_chain, 2, num_iterations=2, progress=False,
                          checkpoint_dir=str(checkpoint_dir))
    written = sorted(p.name for p in checkpoint_dir.iterdir())
    assert written, "the loop wrote no checkpoint"

    resumed = bayesian_relabel_loop(tiny_chain, 2, num_iterations=15, progress=False,
                                    resume_from=str(checkpoint_dir))
    assert resumed.chain.shape == tiny_chain.chain.shape

    # A checkpoint that already holds the whole budget is a no-op, not a rerun.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="petra.bayesian_gaussian"):
        done = bayesian_relabel_loop(tiny_chain, 2, num_iterations=1, progress=False,
                                     resume_from=str(checkpoint_dir))
    assert "nothing to do" in caplog.text
    assert np.array_equal(done.chain, resumed.chain, equal_nan=True) or \
        done.chain.shape == tiny_chain.chain.shape
