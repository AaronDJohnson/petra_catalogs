"""
The shared ``make_catalog_*`` keyword contract.

Every entry point takes the same keyword names, accepts the legacy spellings of
the ones that were renamed (with a :class:`DeprecationWarning`), and rejects
anything it does not recognise with a :class:`TypeError` rather than silently
ignoring it.  A typo in a hyperparameter that is quietly swallowed is a whole
run wasted, so the rejection matters as much as the acceptance.

Every call here is deliberately degenerate -- ``threshold_samples`` is larger
than the chain, so every source slot falls back to the uniform prior and no flow
is ever trained.  These tests are about argument handling, not about the answer.
"""

import numpy as np
import pytest

from petra.bayesian_gaussian import make_catalog_bayesian_gaussian
from petra.copula_flows import make_catalog_copula_flows
from petra.make_catalog import make_catalog_mv_normal
from petra.posterior_chain import PosteriorChain
from petra.utils import DEPRECATED_KEYWORDS, resolve_deprecated_kwargs

#: Big enough that no slot ever qualifies for a trained flow.
NO_FLOWS = 10 ** 6


@pytest.fixture
def tiny_chain():
    """A 12-sample, two-source, one-parameter chain -- just enough to run on."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 1)) + np.array([[0.0], [8.0]])
    return PosteriorChain(chain, 2, 1, trans_dimensional=True)


def test_resolve_deprecated_kwargs_maps_every_documented_alias():
    """`DEPRECATED_KEYWORDS` is the single source of truth; nothing may bypass it."""
    for old, new in DEPRECATED_KEYWORDS.items():
        with pytest.warns(DeprecationWarning, match=old):
            assert resolve_deprecated_kwargs("f", {old: "sentinel"}) == {new: "sentinel"}


def test_copula_flows_accepts_the_legacy_spellings(tiny_chain):
    with pytest.warns(DeprecationWarning, match="mv_normal_init_iterations"):
        catalog = make_catalog_copula_flows(
            tiny_chain, 2, num_iterations=1, threshold_samples=NO_FLOWS,
            initialization_param_index=None, progress=False,
            mv_normal_init_iterations=1,
        )
    assert catalog.chain.shape == (12, 2, 1)


def test_bayesian_gaussian_accepts_the_legacy_spellings(tiny_chain):
    with pytest.warns(DeprecationWarning, match="mv_normal_init"):
        catalog = make_catalog_bayesian_gaussian(
            tiny_chain, 2, num_iterations=1, init_num_iterations=1,
            progress=False, mv_normal_init=False,
        )
    assert catalog.chain.shape == (12, 2, 1)


@pytest.mark.parametrize("entry_point", [
    make_catalog_copula_flows,
    make_catalog_bayesian_gaussian,
], ids=lambda f: f.__name__)
def test_unknown_keywords_are_rejected(entry_point, tiny_chain):
    """A misspelled hyperparameter must fail loudly, not be silently dropped."""
    with pytest.raises(TypeError):
        entry_point(tiny_chain, 2, progress=False, nnots=8)


def test_alias_not_applicable_to_an_entry_point_is_rejected(tiny_chain):
    """
    `mv_normal_init` renames to `init_with_mv_normal`, which
    `make_catalog_mv_normal` does not have -- that entry point *is* the
    multivariate-normal method, so there is nothing to initialize with.

    The shared DEPRECATED_KEYWORDS table used to be applied blindly, so this
    warned "use 'init_with_mv_normal' instead" and then silently discarded the
    value, while the keyword it recommended raised TypeError. A deprecation must
    never point at a keyword the callee rejects.
    """
    with pytest.raises(TypeError, match="not applicable to this entry point"):
        make_catalog_mv_normal(tiny_chain, 2, num_iterations=1, mv_normal_init=True,
                               progress=False)

    # The recommendation in the message must itself be rejected, not half-accepted.
    with pytest.raises(TypeError):
        make_catalog_mv_normal(tiny_chain, 2, num_iterations=1,
                               init_with_mv_normal=True, progress=False)


def test_applicable_alias_still_works_on_the_same_entry_point(tiny_chain):
    """The other alias on that function does map to a real keyword, and must survive."""
    with pytest.warns(DeprecationWarning, match="init_num_iterations"):
        catalog = make_catalog_mv_normal(tiny_chain, 2, num_iterations=1,
                                         mv_normal_init_iterations=1, progress=False)
    assert catalog.chain.shape == tiny_chain.chain.shape


# ---------------------------------------------------------------------------
# The contract, checked across all three entry points at once
# ---------------------------------------------------------------------------
#
# These are the integration tests for the keyword contract.  The per-module
# tests above each cover one entry point, which is how the alias handling
# managed to drift between methods in the first place: `n_phases` raised on
# `make_catalog_bayesian_gaussian` while working elsewhere, and the
# "not both" check was live on only one method.  Parametrizing over the whole
# set is what makes a future divergence fail the build.

ENTRY_POINTS = [
    make_catalog_mv_normal,
    make_catalog_copula_flows,
    make_catalog_bayesian_gaussian,
]

#: Keyword arguments that keep every entry point fast and flow-free.
FAST = dict(num_iterations=1, progress=False, threshold_samples=NO_FLOWS)


def _fast_kwargs(entry_point):
    """Drop keywords a given entry point does not accept (`threshold_samples`)."""
    import inspect
    accepted = inspect.signature(entry_point).parameters
    return {k: v for k, v in FAST.items() if k in accepted}


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_every_entry_point_accepts_the_n_phases_alias(entry_point, tiny_chain):
    """
    `n_phases` renames to `num_iterations`, which all three entry points have.

    `make_catalog_bayesian_gaussian` used to resolve its two aliases with
    individual `deprecated_keyword` calls and then raise ``TypeError: Unexpected
    keyword arguments: ['n_phases']`` -- rejecting an alias for a keyword it does
    support, alone among the methods.
    """
    kwargs = _fast_kwargs(entry_point)
    kwargs.pop("num_iterations")
    with pytest.warns(DeprecationWarning, match="n_phases"):
        catalog = entry_point(tiny_chain, 2, n_phases=1, **kwargs)
    assert catalog.chain.shape == tiny_chain.chain.shape


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_every_entry_point_rejects_both_spellings(entry_point, tiny_chain):
    """
    Passing an alias *and* its replacement is an error, not a silent override.

    Several entry points called `resolve_deprecated_kwargs` without `current=`, so
    the alias won and the explicitly passed `num_iterations` was discarded
    without a word.
    """
    kwargs = _fast_kwargs(entry_point)
    kwargs["num_iterations"] = 3          # explicitly passed, differs from default
    with pytest.warns(DeprecationWarning, match="n_phases"):
        with pytest.raises(TypeError, match="not both"):
            entry_point(tiny_chain, 2, n_phases=1, **kwargs)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_alias_is_allowed_when_the_new_name_is_left_at_its_default(entry_point, tiny_chain):
    """
    The "not both" check keys off *differing from the default*, not presence.

    Spelling out the default explicitly is not a conflict, so this must still be
    accepted -- otherwise wrapper code that forwards a full keyword set would be
    unable to use the alias at all.
    """
    import inspect
    default = inspect.signature(entry_point).parameters["num_iterations"].default
    kwargs = _fast_kwargs(entry_point)
    kwargs["num_iterations"] = default
    with pytest.warns(DeprecationWarning, match="n_phases"):
        catalog = entry_point(tiny_chain, 2, n_phases=1, **kwargs)
    assert catalog.chain.shape == tiny_chain.chain.shape


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda f: f.__name__)
def test_deprecation_warning_fires_even_when_the_call_is_rejected(entry_point, tiny_chain):
    """
    A deprecated spelling warns whether or not the call then turns out to be an
    error.  `resolve_deprecated_kwargs` used to run its "not both" check *before*
    warning, so the warning vanished exactly when two spellings collided --
    the case where knowing which one is deprecated matters most.
    """
    kwargs = _fast_kwargs(entry_point)
    kwargs["num_iterations"] = 3
    with pytest.warns(DeprecationWarning, match="n_phases"):
        with pytest.raises(TypeError, match="not both"):
            entry_point(tiny_chain, 2, n_phases=1, **kwargs)
