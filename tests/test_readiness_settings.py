"""Reject settings that cannot define a finite, executable training run."""

import numpy as np
import pytest
import petra

from petra.copula_flows import make_copula_flows_fit
from petra.options import CopulaFlowFit, Initialization
from petra.aux_distributions import mv_normal_aux_distribution
from petra.cost_matrix import create_compute_cost_matrix
from petra.parametric_fits import create_parametric_fit, mv_normal_fit
from petra.posterior_chain import PosteriorChain
from petra.relabel import run_relabeling_loop
from petra.utils import find_prob_in_model


@pytest.mark.parametrize("field", ["knots", "flow_layers", "max_epochs", "max_patience"])
@pytest.mark.parametrize("value", [1.5, np.nan, np.inf, True])
def test_flow_counts_must_be_integers(field, value):
    with pytest.raises(ValueError, match=field + ".*integer"):
        CopulaFlowFit(**{field: value})


@pytest.mark.parametrize("value", [1.5, np.nan, np.inf, True])
def test_initialization_budget_must_be_an_integer(value):
    with pytest.raises(ValueError, match="init_num_iterations.*integer"):
        Initialization(num_iterations=value)


def test_numpy_integer_settings_are_supported():
    assert CopulaFlowFit(knots=np.int64(4)).knots == 4
    assert Initialization(num_iterations=np.int64(2)).num_iterations == 2


def test_learning_rate_must_be_finite():
    with pytest.raises(ValueError, match="learning_rate.*finite"):
        CopulaFlowFit(learning_rate=np.inf)


@pytest.mark.parametrize("kwargs", [{"knots": 1.5}, {"learning_rate": np.inf}])
def test_direct_fitter_validates_settings_before_building_a_model(kwargs):
    chain = np.arange(12.0).reshape(6, 1, 2)
    with pytest.raises(ValueError):
        make_copula_flows_fit(chain, **kwargs)


@pytest.mark.parametrize("eps", [-0.1, 0.6, np.inf, np.nan])
def test_inclusion_clipping_rejects_invalid_bounds(eps):
    chain = np.arange(12.0).reshape(6, 2, 1)
    chain[:2, 0] = np.nan
    with pytest.raises(ValueError, match="eps"):
        find_prob_in_model(chain, 2, eps=eps)


def test_inclusion_clipping_supports_unclipped_and_equal_bounds():
    chain = np.array([[[1.0], [np.nan]], [[2.0], [np.nan]]])
    np.testing.assert_array_equal(find_prob_in_model(chain, 2, eps=0), [1, 0])
    np.testing.assert_array_equal(find_prob_in_model(chain, 2, eps=0.5), [0.5, 0.5])


@pytest.mark.parametrize("iterations", [1.5, np.nan, np.inf, True])
def test_shared_relabeling_loop_requires_an_integer_budget(iterations):
    chain = np.arange(6.0).reshape(6, 1, 1)
    with pytest.raises(ValueError, match="num_iterations.*integer"):
        run_relabeling_loop(
            PosteriorChain(chain, 1, 1),
            create_parametric_fit(mv_normal_fit),
            create_compute_cost_matrix(mv_normal_aux_distribution),
            num_iterations=iterations,
            progress=False,
        )


@pytest.mark.parametrize("method", [
    "make_catalog_mv_normal", "make_catalog_bayesian_gaussian", "make_catalog_copula_flows",
])
def test_catalog_entry_points_reject_invalid_clipping(method):
    values = np.random.default_rng(0).normal(size=(6, 2, 2))
    with pytest.raises(ValueError, match="eps"):
        getattr(petra, method)(
            PosteriorChain(values, 2, 2), 2,
            num_iterations=1, eps=0.6, progress=False,
        )
