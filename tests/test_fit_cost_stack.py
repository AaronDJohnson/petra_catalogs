"""
Guards on the shared fit -> aux-distribution -> cost-matrix stack.

Every relabeling method in petra is built from three interchangeable pieces: a fit
(:mod:`petra.parametric_fits`), an evaluator (:mod:`petra.aux_distributions`) and the
assignment matrix that joins them (:mod:`petra.cost_matrix`).  A degenerate fit does not
crash that stack -- it produces a NaN row, which :mod:`petra.cost_matrix` replaces by a
single constant, and a row-constant cost row is invariant under
``scipy.optimize.linear_sum_assignment``.  The label is then assigned arbitrarily and
nothing downstream records that it happened.  These tests pin the two defences against
that: scales are floored where they are fitted, and are rejected where they are used.

They also cover the helpers in :mod:`petra.flow_utils`, the deprecation plumbing in
:mod:`petra.utils` that the catalog entry points share, and the type check
that rejects settings objects passed under the wrong parameter.
"""

import dataclasses
import warnings

import numpy as np
import pytest

import petra
from petra.aux_distributions import (mv_normal_aux_distribution,
                                     uni_normal_aux_distribution_single_parameter)
from petra.cost_matrix import create_compute_cost_matrix
from petra.flow_utils import (DEFAULT_LOG_PROB_FLOOR, safe_flow_log_prob,
                              stack_flow_log_probs)
from petra.options import CopulaFlowFit, Initialization
from petra.parametric_fits import (MIN_UNIVARIATE_SAMPLES, _regularize_covariance,
                                   _regularize_std, mv_normal_fit,
                                   uni_normal_fit_single_parameter)
from petra.utils import (UniformPrior, _UNSET, deprecated_keyword,
                         resolve_deprecated_kwargs, resolve_entry_point_kwargs)


# ---------------------------------------------------------------------------
# The univariate sibling of the singular-covariance bug
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_with_a_constant_source():
    """A chain whose source 1 has an identical value of parameter 0 in every sample."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(50, 2, 3))
    chain[:, 1, 0] = 3.0
    return chain


def test_a_constant_source_still_gets_a_positive_std(chain_with_a_constant_source):
    means, stds = uni_normal_fit_single_parameter(chain_with_a_constant_source, 2, 0)
    assert means[1] == pytest.approx(3.0)
    assert np.all(stds > 0.0)
    assert np.all(np.isfinite(stds))
    # The floor is relative to the spread of the parameter, and small enough that the
    # healthy source is untouched.
    assert stds[0] == pytest.approx(np.std(chain_with_a_constant_source[:, 0, 0], ddof=1))
    assert stds[1] < 1e-3


def test_a_constant_source_does_not_produce_a_row_constant_cost_row(chain_with_a_constant_source):
    """The failure mode: std == 0 -> NaN row -> constant row -> arbitrary assignment."""
    chain = chain_with_a_constant_source
    aux_parameters = uni_normal_fit_single_parameter(chain, 2, 0)
    compute_cost = create_compute_cost_matrix(uni_normal_aux_distribution_single_parameter,
                                              single_parameter=0)
    prob_in_model = np.array([0.5, 0.5])

    cost_matrix = compute_cost(chain[0], aux_parameters, prob_in_model, num_distributions=2)
    assert cost_matrix.shape == (2, 2)
    assert np.all(np.isfinite(cost_matrix))
    # Row 1 is the constant source. It must still discriminate between the slots,
    # otherwise linear_sum_assignment is free to hand it either one.
    assert not np.isclose(cost_matrix[1, 0], cost_matrix[1, 1])
    # And it must prefer the slot that actually holds its constant value.
    assert cost_matrix[1, 1] > cost_matrix[1, 0]


def test_a_parameter_that_never_varies_anywhere_is_still_fittable():
    """With no spread to scale the ridge by, the floor falls back to an absolute one."""
    chain = np.full((20, 2, 1), 7.0)
    means, stds = uni_normal_fit_single_parameter(chain, 2, 0)
    assert np.all(means == 7.0)
    assert np.all(stds > 0.0)
    assert np.all(np.isfinite(stds))


def test_a_sparse_source_falls_back_to_the_same_parameter_pooled():
    """The fallback must pool one parameter, never mix parameter dimensions."""
    chain = np.full((40, 2, 2), np.nan)
    chain[:, 0, 0] = np.linspace(0.0, 1.0, 40)
    chain[:, 0, 1] = 100.0                      # a wildly different parameter
    chain[:3, 1, :] = 0.5                       # source 1 is below the sparse cutoff
    assert 3 < MIN_UNIVARIATE_SAMPLES

    means, stds = uni_normal_fit_single_parameter(chain, 2, fit_parameter=0)
    pooled = chain[:, :, 0][~np.isnan(chain[:, :, 0])]
    assert means[1] == pytest.approx(pooled.mean())
    assert stds[1] == pytest.approx(pooled.std(ddof=1), rel=1e-6)
    # Parameter 1 sits at 100.0; had it been pooled in, the std would be enormous.
    assert stds[1] < 1.0


@pytest.mark.parametrize("std,scale,expected", [
    (2.0, 1.0, 2.0),          # a healthy std is left alone
    (0.0, 1.0, 1e-4),         # sqrt(COVARIANCE_RIDGE) * scale
    (0.0, 0.0, 1e-4),         # a degenerate scale falls back to 1.0
    (np.nan, 2.0, 2e-4),      # NaN is treated as no spread at all
])
def test_regularize_std(std, scale, expected):
    assert _regularize_std(std, scale) == pytest.approx(expected, rel=1e-6)


def test_regularize_covariance_falls_back_when_there_is_no_scale():
    cov = _regularize_covariance(np.zeros((2, 2)))
    assert np.allclose(cov, 1e-8 * np.eye(2))
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_mv_normal_fit_falls_back_for_a_source_with_too_few_samples():
    """Fewer than n_params + 2 valid samples means the covariance cannot have full rank."""
    rng = np.random.default_rng(1)
    chain = rng.normal(size=(60, 2, 3))
    chain[3:, 1, :] = np.nan                     # 3 valid samples, 3 parameters

    means, covs = mv_normal_fit(chain, max_num_sources=2)
    assert means.shape == (2, 3)
    assert covs.shape == (2, 3, 3)
    # The fallback pools the whole chain, so the two fits are close but not identical
    # (source 1 contributes its three samples to the pool).
    assert np.all(np.linalg.eigvalsh(covs[1]) > 0.0)
    np.linalg.cholesky(covs[1])                  # must not raise


# ---------------------------------------------------------------------------
# The evaluators refuse a degenerate fit instead of returning NaN
# ---------------------------------------------------------------------------

def test_uni_normal_aux_rejects_a_non_positive_std():
    with pytest.raises(ValueError, match=r"source indices \[1\]"):
        uni_normal_aux_distribution_single_parameter(
            np.zeros((2, 1)), ([0.0, 1.0], [1.0, 0.0]), [0, 1], single_parameter=0)


def test_uni_normal_aux_rejects_a_nan_std():
    with pytest.raises(ValueError, match="not strictly positive"):
        uni_normal_aux_distribution_single_parameter(
            np.zeros((2, 1)), ([0.0, 1.0], [1.0, np.nan]), 1, single_parameter=0)


def test_mv_normal_aux_rejects_a_singular_covariance():
    sample = np.zeros((2, 2))
    means = [np.zeros(2), np.zeros(2)]
    covs = [np.eye(2), np.zeros((2, 2))]
    with pytest.raises(np.linalg.LinAlgError, match=r"source indices \[1\]"):
        mv_normal_aux_distribution(sample, (means, covs), [0, 1])


def test_mv_normal_aux_matches_scipy_on_a_healthy_fit():
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(2)
    sample = rng.normal(size=(3, 2))
    mean = np.array([0.5, -0.25])
    cov = np.array([[1.3, 0.4], [0.4, 0.8]])

    logpdf = mv_normal_aux_distribution(sample, ([mean], [cov]), 0)
    assert logpdf.shape == (3,)
    assert np.allclose(logpdf, multivariate_normal(mean, cov).logpdf(sample))


# ---------------------------------------------------------------------------
# flow_utils
# ---------------------------------------------------------------------------

class _ScalarFlow:
    """A distribution that returns one scalar no matter how many points it is given."""

    def log_prob(self, x):
        return -1.5


class _AngryFlow:
    """A distribution that raises the error jax emits for an internal NaN."""

    def log_prob(self, x):
        raise FloatingPointError("nan in log_prob")


def test_safe_flow_log_prob_broadcasts_a_scalar_over_a_batch():
    out = np.asarray(safe_flow_log_prob(_ScalarFlow(), np.zeros((4, 2))))
    assert out.shape == (4,)
    assert np.allclose(out, -1.5)


def test_safe_flow_log_prob_floors_a_flow_that_raises_on_nan():
    out = np.asarray(safe_flow_log_prob(_AngryFlow(), np.zeros((3, 2))))
    assert out.shape == (3,)
    assert np.allclose(out, DEFAULT_LOG_PROB_FLOOR)


def test_safe_flow_log_prob_floors_an_empty_batch():
    prior = UniformPrior([0.0], [1.0])
    out = np.asarray(safe_flow_log_prob(prior, np.zeros((0, 1))))
    assert out.shape == (0,)


def test_safe_flow_log_prob_rejects_a_mis_shaped_input():
    prior = UniformPrior([0.0], [1.0])
    with pytest.raises(ValueError, match="1-D or 2-D"):
        safe_flow_log_prob(prior, np.zeros((2, 2, 1)))


def test_stack_flow_log_probs_rejects_a_single_point():
    prior = UniformPrior([0.0], [1.0])
    with pytest.raises(ValueError, match="must be 2-D"):
        stack_flow_log_probs([prior], np.zeros(1))


def test_stack_flow_log_probs_honours_a_custom_floor():
    prior = UniformPrior([0.0], [1.0])
    sample = np.array([[0.5], [9.0]])            # the second point is outside the support
    out = np.asarray(stack_flow_log_probs([prior, None], sample, floor=-7.0))
    assert out.shape == (2, 2)
    assert out[0, 1] == pytest.approx(-7.0)
    assert np.allclose(out[1], -7.0)             # an unfitted slot is floored everywhere


# ---------------------------------------------------------------------------
# resolve_deprecated_kwargs: the "not both" check
# ---------------------------------------------------------------------------

def _entry_point(posterior_chain, *, num_iterations=50, init_with_mv_normal=True,
                 **deprecated):
    """Stand-in with the shared keyword-only prefix of the make_catalog_* family."""


def test_resolve_deprecated_kwargs_rejects_both_spellings():
    with pytest.raises(TypeError, match="not both"):
        resolve_deprecated_kwargs(_entry_point, {"n_phases": 7},
                                  current={"num_iterations": 3})


def test_resolve_deprecated_kwargs_accepts_the_alias_when_the_new_name_is_default():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        resolved = resolve_deprecated_kwargs(_entry_point, {"n_phases": 7},
                                             current={"num_iterations": 50})
    assert resolved == {"num_iterations": 7}


def test_resolve_deprecated_kwargs_without_current_keeps_the_old_behaviour():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        resolved = resolve_deprecated_kwargs(_entry_point, {"n_phases": 7})
    assert resolved == {"num_iterations": 7}


def test_resolve_deprecated_kwargs_treats_a_parameter_without_a_default_as_passed():
    def f(*, num_iterations, **deprecated):
        """A required keyword has no default, so any value counts as explicit."""

    with pytest.raises(TypeError, match="not both"):
        resolve_deprecated_kwargs(f, {"n_phases": 7}, current={"num_iterations": 7})


def test_deprecated_keyword_rejects_both_spellings():
    """The single-alias helper has always raised; `current` gives the bulk one parity."""
    with pytest.warns(DeprecationWarning) as caught:
        with pytest.raises(TypeError, match="not both"):
            deprecated_keyword({"n_phases": 7}, "n_phases", "num_iterations", 3, 50)
    # Both warnings must survive the error path: reversing warn and raise would
    # hide the deprecation notice exactly when two spellings collide.
    messages = [str(w.message) for w in caught]
    assert any("resolve_deprecated_kwargs" in m for m in messages)
    assert any("'n_phases' is deprecated" in m for m in messages)


def test_deprecated_keyword_passes_the_current_value_through_untouched():
    """
    With the old spelling absent, the helper returns `current` and touches nothing.

    `deprecated_keyword` is exported in ``petra.utils.__all__`` but no longer has
    a caller inside the package -- every entry point resolves its whole
    ``**deprecated`` catch-all through `resolve_deprecated_kwargs` now.  This
    pins the no-op path that those call sites used to cover incidentally.
    """
    kwargs = {"unrelated": 1}
    with pytest.warns(DeprecationWarning) as caught:
        assert deprecated_keyword(kwargs, "n_phases", "num_iterations", 3, 50) == 3
    assert kwargs == {"unrelated": 1}, "a keyword it does not own must be left alone"
    # Only the helper's own retirement notice: there was no old spelling to warn about.
    assert [str(w.message) for w in caught] == [
        "petra.utils.deprecated_keyword() is deprecated in favour of "
        "resolve_deprecated_kwargs() and is scheduled for removal in petra 1.2."
    ]


def test_deprecated_keyword_announces_its_own_scheduled_removal():
    """The trap this closes: a public helper the package itself refuses to call."""
    with pytest.warns(DeprecationWarning, match="scheduled for removal in petra 1.2") as caught:
        deprecated_keyword({}, "n_phases", "num_iterations", 3, 50)
    message = str(caught[0].message)
    assert "deprecated_keyword" in message
    assert "resolve_deprecated_kwargs" in message, "the notice must name the replacement"


def test_deprecated_keyword_still_maps_the_old_spelling_and_warns_twice():
    """Behaviour is unchanged apart from the extra notice; the mapping still happens."""
    kwargs = {"n_phases": 7, "unrelated": 1}
    with pytest.warns(DeprecationWarning) as caught:
        assert deprecated_keyword(kwargs, "n_phases", "num_iterations", 50, 50) == 7
    assert kwargs == {"unrelated": 1}, "the resolved alias must be consumed"
    messages = [str(w.message) for w in caught]
    assert len(messages) == 2, f"expected the retirement notice and the alias notice, got {messages}"
    assert "resolve_deprecated_kwargs" in messages[0]
    assert messages[1] == ("'n_phases' is deprecated and will be removed; "
                           "use 'num_iterations' instead.")


def test_deprecated_keyword_points_its_retirement_notice_at_the_caller():
    """stacklevel must blame the call site, not utils.py, or the notice is useless."""
    def caller():
        """Stand-in for a petra module that still reaches for the old helper."""
        return deprecated_keyword({}, "n_phases", "num_iterations", 3, 50)

    with pytest.warns(DeprecationWarning, match="resolve_deprecated_kwargs") as caught:
        caller()
    assert caught[0].filename == __file__
    assert caught[0].lineno == caller.__code__.co_firstlineno + 2


# ---------------------------------------------------------------------------
# resolve_entry_point_kwargs: the settings object has to be of the right class
# ---------------------------------------------------------------------------

def _options_entry(posterior_chain, *, num_iterations=50,
                   initialization: Initialization | None = None,
                   flow_fit: CopulaFlowFit | None = None,
                   **flat):
    """Stand-in entry point carrying flow and initialization options."""


def test_an_options_object_of_the_wrong_class_is_rejected_at_the_boundary():
    """An initialization object cannot supply flow training settings."""
    with pytest.raises(TypeError) as excinfo:
        resolve_entry_point_kwargs(_options_entry, {},
                                   options={"flow_fit": Initialization()})
    assert str(excinfo.value) == (
        "_options_entry(): 'flow_fit' must be a CopulaFlowFit, got Initialization."
    )


def test_the_wrong_class_message_names_the_parameter_it_arrived_under():
    """With two options parameters, "wrong type" alone does not say which one."""
    with pytest.raises(TypeError) as excinfo:
        resolve_entry_point_kwargs(_options_entry, {},
                                   options={"initialization": CopulaFlowFit()})
    assert str(excinfo.value) == (
        "_options_entry(): 'initialization' must be a Initialization, got CopulaFlowFit."
    )


def test_a_non_options_object_entirely_is_rejected_too():
    """A bare dict of settings is the other easy mistake, and is not a CopulaFlowFit."""
    with pytest.raises(TypeError, match=r"'flow_fit' must be a CopulaFlowFit, got dict\."):
        resolve_entry_point_kwargs(_options_entry, {}, options={"flow_fit": {"knots": 8}})


def test_a_subclass_of_the_expected_options_class_is_still_accepted():
    """The check is ``isinstance``, so a caller may carry extra fields of their own."""
    @dataclasses.dataclass(frozen=True)
    class TaggedCopulaFlowFit(CopulaFlowFit):
        """A `CopulaFlowFit` with one field of the caller's own bolted on."""

        run_label: str = "sweep-3"

    passed = TaggedCopulaFlowFit(knots=3)
    _, resolved = resolve_entry_point_kwargs(_options_entry, {},
                                             options={"flow_fit": passed})
    assert resolved["flow_fit"] is passed


@pytest.mark.parametrize("parameter,wrong,expected_name,got_name", [
    ("flow_fit", Initialization(), "CopulaFlowFit", "Initialization"),
    ("initialization", CopulaFlowFit(), "Initialization", "CopulaFlowFit"),
])
def test_copula_entry_point_rejects_a_settings_object_under_the_wrong_parameter(
        parameter, wrong, expected_name, got_name):
    """Type errors name both the public parameter and the expected settings type."""
    chain = petra.PosteriorChain(np.random.default_rng(0).normal(size=(12, 2, 1)),
                                 num_sources=2, num_params_per_source=1)
    with pytest.raises(TypeError) as excinfo:
        petra.make_catalog_copula_flows(chain, 2, progress=False, **{parameter: wrong})
    assert str(excinfo.value) == (
        f"make_catalog_copula_flows(): '{parameter}' must be a {expected_name}, got {got_name}."
    )


def test_unset_sentinel_is_readable():
    assert repr(_UNSET) == "<unset>"


def test_resolve_deprecated_kwargs_compares_array_valued_defaults_elementwise():
    default = np.array([1.0, 2.0])

    def f(*, num_iterations=default, **deprecated):
        """An array default cannot be compared with a bare ``!=``."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        resolved = resolve_deprecated_kwargs(f, {"n_phases": 7},
                                             current={"num_iterations": np.array([1.0, 2.0])})
    assert resolved == {"num_iterations": 7}
    with pytest.raises(TypeError, match="not both"):
        resolve_deprecated_kwargs(f, {"n_phases": 7},
                                  current={"num_iterations": np.array([9.0, 9.0])})
