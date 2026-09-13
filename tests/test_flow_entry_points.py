"""
The coppuccino entry point's argument forwarding and deprecated aliases.

The downstream relabeler is replaced with a recorder to check that settings
reach their intended boundary. General options validation lives in
``test_options.py``; real copula training equivalence lives in
``test_end_to_end.py``.
"""

import dataclasses
import inspect
import warnings

import numpy as np
import pytest

import petra.copula_flows as copula_flows
from petra.copula_flows import DEFAULT_MAX_PATIENCE, make_catalog_copula_flows
from petra.options import CopulaFlowFit, Initialization
from petra.posterior_chain import PosteriorChain


@pytest.fixture
def tiny_chain():
    """A 12-sample, two-source, one-parameter chain -- just enough to run on."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 1)) + np.array([[0.0], [8.0]])
    return PosteriorChain(chain, 2, 1, trans_dimensional=True)


@pytest.fixture
def record_relabel(monkeypatch):
    """
    Replace the copula relabeler with a recorder.

    Returns
    -------
    calls : dict
        Populated with the keyword arguments the entry point forwarded, under
        the key ``"kwargs"``.  The stub returns its input chain unchanged.
    """
    calls = {}

    def fake_relabel(posterior_chain, **kwargs):
        calls["kwargs"] = kwargs
        return posterior_chain

    monkeypatch.setattr(copula_flows, "relabel_copula_flows", fake_relabel)
    return calls


def test_n_phases_alias_is_honoured_not_just_warned(tiny_chain, record_relabel):
    """
    ``n_phases`` renames to ``num_iterations``, and the value must survive.

    The entry point used to call ``resolve_deprecated_kwargs`` and then read
    back only the two initialization keys, so the ``num_iterations`` the
    resolver returned was discarded: the caller was told their keyword had a new
    name, and then silently got the default (50) instead of the 7 they asked
    for.
    """
    with pytest.warns(DeprecationWarning, match="n_phases"):
        make_catalog_copula_flows(
            tiny_chain, 2, init_with_mv_normal=False, progress=False, n_phases=7)

    assert record_relabel["kwargs"]["num_iterations"] == 7


def test_explicit_num_iterations_is_kept_without_the_alias(tiny_chain, record_relabel):
    """The alias handling must not disturb the ordinary spelling."""
    make_catalog_copula_flows(
        tiny_chain, 2, num_iterations=3, init_with_mv_normal=False, progress=False)

    assert record_relabel["kwargs"]["num_iterations"] == 3


def test_copula_flows_accepts_the_legacy_patience_spelling(tiny_chain, record_relabel):
    """
    ``patience`` is coppuccino's name for the early-stopping patience.

    petra spells it ``max_patience`` everywhere else, so the copula entry point
    was the only one whose training keywords could not be copied across from a
    sibling.  The old name still works, with a warning.
    """
    with pytest.warns(DeprecationWarning, match="patience"):
        make_catalog_copula_flows(tiny_chain, 2, num_iterations=1,
                                  init_with_mv_normal=False, progress=False,
                                  patience=5)

    assert record_relabel["kwargs"]["max_patience"] == 5


def test_copula_flows_defaults_to_the_documented_patience(tiny_chain, record_relabel):
    """Without either spelling, the module-level default is what is forwarded."""
    make_catalog_copula_flows(tiny_chain, 2, num_iterations=1,
                              init_with_mv_normal=False, progress=False)

    assert record_relabel["kwargs"]["max_patience"] == DEFAULT_MAX_PATIENCE


def test_copula_flows_rejects_both_patience_spellings(tiny_chain, record_relabel):
    """Passing both spellings is a mistake, not a precedence question."""
    with pytest.raises(TypeError, match="not both"):
        with pytest.warns(DeprecationWarning):
            make_catalog_copula_flows(tiny_chain, 2, init_with_mv_normal=False,
                                      progress=False, patience=5, max_patience=9)


def test_relabel_copula_flows_also_accepts_the_legacy_spelling(monkeypatch, tiny_chain):
    """The rename reaches the mid-level relabeler, not only the entry point."""
    seen = {}

    def fake_make_fit(chain, **kwargs):
        seen.update(kwargs)
        return lambda chain, max_num_sources: [None] * max_num_sources

    monkeypatch.setattr(copula_flows, "make_copula_flows_fit", fake_make_fit)
    monkeypatch.setattr(copula_flows, "create_relabel_samples",
                        lambda *args, **kwargs: (lambda pc, **kw: pc))

    with pytest.warns(DeprecationWarning, match="patience"):
        copula_flows.relabel_copula_flows(tiny_chain, 2, patience=3, progress=False)

    assert seen["max_patience"] == 3


@pytest.mark.parametrize("entry_point", [
    make_catalog_copula_flows,
    copula_flows.relabel_copula_flows,
], ids=lambda f: f.__name__)
def test_eps_default_matches_the_rest_of_the_package(entry_point):
    """
    ``eps`` clips the inclusion probabilities, and its value sets the scale of
    every ``log(1 - p)`` in the cost matrix.  It is ``1e-6`` in
    ``relabel_mv_normal``, ``relabel_univariate_normal`` and the Bayesian
    relabeler; the flow relabelers used to default to ``1e-2``, four orders of
    magnitude away, which made their reported costs incomparable.
    """
    assert inspect.signature(entry_point).parameters["eps"].default == 1e-6


# ---------------------------------------------------------------------------
# make_catalog_copula_flows: the grouped settings objects
# ---------------------------------------------------------------------------

#: Non-default values for every field of both settings objects.
NON_DEFAULT_FLAT = dict(
    init_with_mv_normal=True, init_num_iterations=7,
    knots=4, flow_layers=2, max_epochs=3, max_patience=2, learning_rate=2e-3,
    log_prob_floor=-33.0,
)


def test_the_settings_objects_are_unpacked_for_the_relabeler(tiny_chain, record_relabel):
    """
    Flow settings must reach the relabeler after initialization is handled.

    :func:`petra.copula_flows.relabel_copula_flows` keeps its flat
    signature for direct callers, so every field of `flow_fit` has to be spelled out on the
    way down.  A field that was dropped there would not fail: the relabeler would
    quietly supply its own default, and the caller's ``max_epochs=3`` would
    become 800.
    """
    make_catalog_copula_flows(
        tiny_chain, 2, num_iterations=1, progress=False,
        initialization=Initialization(with_mv_normal=False),
        flow_fit=CopulaFlowFit(knots=4, flow_layers=2, max_epochs=3, max_patience=2,
                               learning_rate=2e-3, log_prob_floor=-33.0),
    )

    forwarded = record_relabel["kwargs"]
    assert forwarded["knots"] == 4
    assert forwarded["flow_layers"] == 2
    assert forwarded["max_epochs"] == 3
    assert forwarded["max_patience"] == 2
    assert forwarded["learning_rate"] == 2e-3
    assert forwarded["log_prob_floor"] == -33.0


def test_an_omitted_settings_object_still_arrives_as_its_defaults(tiny_chain, record_relabel):
    """
    ``None`` means ``CopulaFlowFit()``, and the class is where those numbers live now.

    They used to be restated on the entry point, on the relabeler and on the
    fitter factory -- three copies, which is how a default drifts.  Reading them
    back off the class is what says the entry point no longer has its own.
    """
    make_catalog_copula_flows(tiny_chain, 2, num_iterations=1, progress=False,
                              init_with_mv_normal=False)

    forwarded = record_relabel["kwargs"]
    for field in dataclasses.fields(CopulaFlowFit):
        assert forwarded[field.name] == getattr(CopulaFlowFit(), field.name), field.name


def test_the_legacy_flat_keywords_are_still_accepted_without_a_warning(
        tiny_chain, record_relabel):
    """
    Every flat spelling keeps working, silently.

    A `DeprecationWarning` here would be wrong: the flat form is not deprecated,
    it is the *other* way of writing the same call, and it is what every existing
    script and notebook uses.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        make_catalog_copula_flows(tiny_chain, 2, num_iterations=1, progress=False,
                                  **NON_DEFAULT_FLAT)

    for field in dataclasses.fields(CopulaFlowFit):
        assert record_relabel["kwargs"][field.name] == NON_DEFAULT_FLAT[field.name]


def test_a_deprecated_alias_onto_an_options_field_still_warns_and_is_honoured(
        tiny_chain, record_relabel, monkeypatch):
    """
    ``mv_normal_init_iterations`` renames to ``init_num_iterations``, which now
    lives on `Initialization` rather than in the signature.

    The alias has to stay applicable -- ``resolve_deprecated_kwargs`` checks its
    replacement against the entry point's accepted keywords, and that set is only
    right if the options index is unioned into it -- and the value has to reach
    the initializer instead of being warned about and then dropped.
    """
    import petra.make_catalog as make_catalog

    seen = {}

    def fake_mv_normal(posterior_chain, **kwargs):
        seen.update(kwargs)
        return posterior_chain

    monkeypatch.setattr(make_catalog, "relabel_mv_normal", fake_mv_normal)

    with pytest.warns(DeprecationWarning, match="mv_normal_init_iterations"):
        make_catalog_copula_flows(tiny_chain, 2, num_iterations=1, progress=False,
                                  initialization_param_index=None,
                                  mv_normal_init_iterations=3)

    assert seen["num_iterations"] == 3


def test_every_make_catalog_entry_point_exposes_eps():
    """
    ``eps`` is part of the shared ``make_catalog_*`` keyword contract.

    `make_catalog_mv_normal` used 1e-6 internally via `relabel_mv_normal` but
    gave the caller no way to set it. Every retained method must expose the
    same knob so callers can compare assignment costs across methods.
    """
    import petra
    names = ["make_catalog_mv_normal", "make_catalog_copula_flows",
             "make_catalog_bayesian_gaussian"]
    for name in names:
        params = inspect.signature(getattr(petra, name)).parameters
        assert "eps" in params, f"{name} does not expose eps"
        assert params["eps"].default == 1e-6, f"{name} has a divergent eps default"
        assert params["eps"].kind is inspect.Parameter.KEYWORD_ONLY, name
