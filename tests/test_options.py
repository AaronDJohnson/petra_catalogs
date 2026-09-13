"""
The grouped settings objects, and the keyword routing that feeds them.

:mod:`petra.options` groups flow training and initialization keywords in
two frozen dataclasses. These tests cover both flat and grouped spellings of
the retained copula entry point and the shared resolver that connects them.

The rest pins the pieces: the defaults, the validation each class
does on construction, and the ``FLAT_KEYWORDS`` marker that keeps
:class:`petra.posterior_chain.PosteriorChain` -- also a dataclass, and the first
argument of every entry point -- from being mistaken for a settings object.
"""

import dataclasses
import inspect
import warnings

import numpy as np
import pytest

from petra.copula_flows import DEFAULT_MAX_PATIENCE, make_catalog_copula_flows
from petra.flow_utils import DEFAULT_LOG_PROB_FLOOR
from petra.options import CopulaFlowFit, Initialization
from petra.posterior_chain import PosteriorChain
from petra.utils import option_field_keywords, resolve_entry_point_kwargs

#: Big enough that no slot ever qualifies for a trained flow, so a test that is
#: about argument routing never pays for flow training.
NO_FLOWS = 10 ** 6

OPTIONS_CLASSES = [CopulaFlowFit, Initialization]


@pytest.fixture
def tiny_chain():
    """A 12-sample, two-source, one-parameter chain -- just enough to run on."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 1)) + np.array([[0.0], [8.0]])
    return PosteriorChain(chain, 2, 1, trans_dimensional=True)


# ---------------------------------------------------------------------------
# The dataclasses themselves
# ---------------------------------------------------------------------------

def test_defaults_are_the_documented_ones():
    """
    The numbers the docstrings and the README quote, pinned.

    They used to be restated on the entry point, on the mid-level relabeler and
    on the fitter factory, which is how ``eps`` managed to drift by four orders
    of magnitude between two entry points.  Now they live here alone, so here is
    where a change to one has to be noticed.
    """
    assert dataclasses.astuple(CopulaFlowFit()) == (16, 8, 800, 20, 1e-3, -50.0)
    assert dataclasses.astuple(Initialization()) == (True, 200)


def test_the_log_prob_floor_literal_matches_the_one_it_duplicates():
    """
    ``petra.options`` may not import ``petra.flow_utils``, which imports jax.

    The shared evaluator and settings class must use the same floor so an
    omitted setting and an explicit default produce the same assignment costs.
    """
    assert CopulaFlowFit().log_prob_floor == DEFAULT_LOG_PROB_FLOOR


def test_copula_constant_is_read_off_the_options_class():
    """The constants that survived must have exactly one definition, not two."""
    assert DEFAULT_MAX_PATIENCE == CopulaFlowFit().max_patience


@pytest.mark.parametrize("options_class", OPTIONS_CLASSES, ids=lambda c: c.__name__)
def test_settings_objects_are_frozen(options_class):
    """
    Mutating a settings object after a run has read it would be a silent lie.

    A caller who mutated one after the run began could get behavior nobody
    could reconstruct from the original arguments.
    """
    settings = options_class()
    field = dataclasses.fields(settings)[0].name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(settings, field, 1)


@pytest.mark.parametrize("options_class, kwargs, message", [
    (CopulaFlowFit, {"knots": 0}, "knots must be at least 1"),
    (CopulaFlowFit, {"max_epochs": 0}, "max_epochs must be at least 1"),
    (CopulaFlowFit, {"max_patience": 0}, "max_patience must be at least 1"),
    (CopulaFlowFit, {"learning_rate": 0.0}, "learning_rate must be strictly positive"),
    (CopulaFlowFit, {"learning_rate": -1e-3}, "learning_rate must be strictly positive"),
    (CopulaFlowFit, {"log_prob_floor": float("-inf")}, "log_prob_floor must be finite"),
    (CopulaFlowFit, {"log_prob_floor": float("nan")}, "log_prob_floor must be finite"),
    (CopulaFlowFit, {"flow_layers": 0}, "flow_layers must be at least 1"),
    (CopulaFlowFit, {"max_epochs": -1}, "max_epochs must be at least 1"),
    (CopulaFlowFit, {"log_prob_floor": float("inf")}, "log_prob_floor must be finite"),
    (Initialization, {"num_iterations": 0}, "init_num_iterations must be at least 1"),
])
def test_out_of_range_settings_are_rejected_on_construction(options_class, kwargs, message):
    """
    Each bound names something that would otherwise fail far from its cause.

    For example, ``knots=0`` reaches ``jax.nn.softmax`` as an empty array.
    """
    with pytest.raises(ValueError, match=message):
        options_class(**kwargs)


@pytest.mark.parametrize("options_class", OPTIONS_CLASSES, ids=lambda c: c.__name__)
def test_flat_keywords_name_real_fields(options_class):
    """
    A typo in ``FLAT_KEYWORDS`` would route a keyword at a field that is not
    there, and ``dataclasses.replace`` would raise from deep inside the
    resolver rather than from the entry point the user called.
    """
    fields = {f.name for f in dataclasses.fields(options_class)}
    assert set(options_class.FLAT_KEYWORDS.values()) == fields
    # FLAT_KEYWORDS is a ClassVar, so it must not itself have become a field.
    assert "FLAT_KEYWORDS" not in fields


def test_the_initialization_map_is_deliberately_not_the_identity():
    """
    The field is `num_iterations`; the flat spelling must be
    ``init_num_iterations``.

    Every entry point already has a `num_iterations` of its own, and it means
    something else -- the top-level relabeling budget, not the initializer's.
    An identity map here would put both names in play for one field and route
    the caller's whole iteration budget into the pre-relabeling.
    """
    assert Initialization.FLAT_KEYWORDS["init_num_iterations"] == "num_iterations"
    assert "num_iterations" not in Initialization.FLAT_KEYWORDS
    assert "num_iterations" not in option_field_keywords(make_catalog_copula_flows)
    assert inspect.signature(make_catalog_copula_flows).parameters["num_iterations"].default == 50


def test_the_initialization_docstring_says_which_pass_with_mv_normal_gates():
    """
    The published description of this field outlived the behaviour it described.

    ``with_mv_normal`` documented itself as "run the initialization at all",
    which was true only while the flow entry points gated *both*
    initialization passes on it -- the coupling the shared preamble removed.
    Under the shipped behaviour the univariate pass is gated on
    ``initialization_param_index`` alone, so that sentence described the bug
    rather than the code, and a reader following it would turn off a pass that
    keeps running.  The field description therefore has to name the pass it
    switches *and* the keyword that switches the other one.
    """
    doc = inspect.getdoc(Initialization)
    field_doc = doc.split("with_mv_normal : bool")[1].split("num_iterations : int")[0]

    assert "multivariate" in field_doc
    assert "initialization_param_index" in field_doc


# ---------------------------------------------------------------------------
# The routing in petra.utils
# ---------------------------------------------------------------------------

def _options_parameters_of(func):
    """
    Options-parameter name to class, read back off the public index.

    Parameters
    ----------
    func : callable
        Entry point to introspect.

    Returns
    -------
    dict of str to type
        One entry per options-object parameter of `func`.
    """
    return {parameter: options_class
            for parameter, _, options_class in option_field_keywords(func).values()}


def test_no_flat_keyword_is_claimed_by_two_options_objects():
    """
    The copula entry point carries both `CopulaFlowFit` and `Initialization`.

    If their field names ever overlapped, the index would silently keep whichever
    parameter the signature scan reached last and the keyword would land in the
    wrong object.  Counting is what catches that; inspecting the two field lists
    by eye is what let it happen.
    """
    entry_point = make_catalog_copula_flows
    claimed = [flat
               for parameter, options_class in _options_parameters_of(entry_point).items()
               for flat in options_class.FLAT_KEYWORDS]
    assert len(claimed) == len(set(claimed))
    assert set(claimed) == set(option_field_keywords(entry_point))


def test_a_plain_dataclass_parameter_is_not_an_options_object(tiny_chain):
    """
    ``PosteriorChain`` is a dataclass, and it is every entry point's first
    argument.

    Detecting options objects with ``dataclasses.is_dataclass`` would therefore
    have made ``chain``, ``num_sources``, ``prob_in_model`` and ``cost_dict``
    accepted keywords on the catalog entry points -- and each of them would have
    quietly ``replace()``d the caller's chain instead of relabeling it.  The
    ``FLAT_KEYWORDS`` marker is what stops that, so the fields of the chain must
    still be unknown keywords.
    """
    index = option_field_keywords(make_catalog_copula_flows)
    for field in dataclasses.fields(PosteriorChain):
        assert field.name not in index

    with pytest.raises(TypeError, match="cost_dict"):
        make_catalog_copula_flows(tiny_chain, 2, progress=False, cost_dict={})
    with pytest.raises(TypeError, match="num_sources"):
        make_catalog_copula_flows(tiny_chain, 2, progress=False, num_sources=3)


def test_a_flat_keyword_is_validated_exactly_as_the_object_would_be():
    """
    ``dataclasses.replace`` re-runs ``__post_init__``, and it must.

    Otherwise ``knots=0`` spelled flat would sail past the check that
    ``CopulaFlowFit(knots=0)`` fails, and the two spellings would stop being the same
    call.
    """
    def entry(chain, *, flow_fit: CopulaFlowFit | None = None, **flat):
        """Stand-in entry point taking one options object."""

    with pytest.raises(ValueError, match="knots must be at least 1"):
        resolve_entry_point_kwargs(entry, {"knots": 0}, options={"flow_fit": None})


def _entry_with_string_annotation(chain, *, flow_fit: "CopulaFlowFit | None" = None, **flat):
    """
    Stand-in entry point whose annotation is a string, as ``from __future__
    import annotations`` would make every annotation in a module.
    """


def test_a_string_annotation_is_resolved_before_it_is_matched():
    """
    ``inspect.signature`` hands back the *text* of an annotation whenever the
    defining module uses postponed evaluation, and ``"CopulaFlowFit | None"`` matches
    no class at all.  Left unresolved, every options object in such a module
    would go unrecognised and its flat keywords would start raising
    ``unexpected keyword argument``.
    """
    index = option_field_keywords(_entry_with_string_annotation)
    assert index["knots"] == ("flow_fit", "knots", CopulaFlowFit)


def test_an_options_parameter_left_alone_still_arrives_built():
    """`resolved_options` is never None-valued, so callers read fields directly."""
    def entry(chain, *, flow_fit: CopulaFlowFit | None = None, **flat):
        """Stand-in entry point taking one options object."""

    _, resolved = resolve_entry_point_kwargs(entry, {}, options={"flow_fit": None})
    assert resolved["flow_fit"] == CopulaFlowFit()

    passed = CopulaFlowFit(knots=3)
    _, resolved = resolve_entry_point_kwargs(entry, {}, options={"flow_fit": passed})
    assert resolved["flow_fit"] is passed


# ---------------------------------------------------------------------------
# make_catalog_copula_flows: both spellings and their collisions
# ---------------------------------------------------------------------------

def test_the_entry_point_accepts_the_options_objects(tiny_chain):
    """The entry point accepts its two grouped settings objects."""
    catalog = make_catalog_copula_flows(
        tiny_chain, 2, num_iterations=1, threshold_samples=NO_FLOWS, progress=False,
        initialization=Initialization(num_iterations=1),
        flow_fit=CopulaFlowFit(knots=4),
    )
    assert catalog.chain.shape == (12, 2, 1)


def test_the_legacy_flat_keywords_are_still_accepted_without_a_warning(tiny_chain):
    """
    Every flat spelling keeps working, silently.

    A `DeprecationWarning` here would be wrong: the flat form is not deprecated,
    it is the *other* way of writing the same call, and it is what every
    existing script and notebook uses.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        catalog = make_catalog_copula_flows(
            tiny_chain, 2, num_iterations=1, threshold_samples=NO_FLOWS, progress=False,
            init_num_iterations=1, knots=4, flow_layers=2, log_prob_floor=-33.0,
        )
    assert catalog.chain.shape == (12, 2, 1)


def test_a_flat_keyword_reaches_the_field_it_names(tiny_chain, monkeypatch):
    """
    Accepted is not the same as honoured: the value has to arrive.

    ``log_prob_floor`` is watched at the relabeler boundary, after routing through
    the entry point and options object.
    """
    seen = []
    import petra.copula_flows as copula_flows

    real = copula_flows.relabel_copula_flows

    def spy(*args, **kwargs):
        seen.append(kwargs.get("log_prob_floor"))
        return real(*args, **kwargs)

    monkeypatch.setattr(copula_flows, "relabel_copula_flows", spy)
    make_catalog_copula_flows(
        tiny_chain, 2, num_iterations=1, init_num_iterations=1,
        threshold_samples=NO_FLOWS, log_prob_floor=-7.5, progress=False)
    assert seen == [-7.5]

    seen.clear()
    make_catalog_copula_flows(
        tiny_chain, 2, num_iterations=1, threshold_samples=NO_FLOWS,
        initialization=Initialization(num_iterations=1),
        flow_fit=CopulaFlowFit(log_prob_floor=-7.5), progress=False)
    assert seen == [-7.5]


@pytest.mark.parametrize("kwargs, message", [
    ({"flow_fit": CopulaFlowFit(), "knots": 8}, "either 'knots' or 'flow_fit', not both"),
    ({"initialization": Initialization(), "init_with_mv_normal": False},
     "either 'init_with_mv_normal' or 'initialization', not both"),
])
def test_an_options_object_and_a_colliding_flat_keyword_is_an_error(tiny_chain, kwargs, message):
    """
    Not a merge, and not a precedence rule.

    Merging would mean answering "does ``CopulaFlowFit(knots=8)`` plus
    ``max_epochs=20`` keep knots=8?" -- an answer that has to be remembered
    rather than read.  Refusing means ``replace`` is only ever applied to the
    class's own defaults, so there is no precedence to get wrong later.
    """
    with pytest.raises(TypeError, match=message):
        make_catalog_copula_flows(tiny_chain, 2, num_iterations=1,
                                  progress=False, **kwargs)


def test_a_flow_keyword_alongside_an_initialization_object_is_fine(tiny_chain):
    """
    The rule is per object, not per call.

    Passing an `Initialization` must not make the flow keywords unreachable.
    """
    catalog = make_catalog_copula_flows(
        tiny_chain, 2, num_iterations=1, threshold_samples=NO_FLOWS, progress=False,
        initialization=Initialization(num_iterations=1), knots=4,
    )
    assert catalog.chain.shape == (12, 2, 1)


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
            make_catalog_copula_flows(
                tiny_chain, 2, progress=False,
                initialization=Initialization(), mv_normal_init=False)


def test_an_alias_and_its_replacement_both_spelled_flat_is_still_an_error(tiny_chain):
    """
    Neither spelling is a named parameter any more, so both land in the
    catch-all and ``resolve_deprecated_kwargs``' own "not both" check cannot see
    the collision.  ``resolve_entry_point_kwargs`` has to catch it instead, and
    the message must name the keyword the caller actually typed.
    """
    with pytest.warns(DeprecationWarning, match="mv_normal_init_iterations"):
        with pytest.raises(
            TypeError,
            match="either 'mv_normal_init_iterations' or 'init_num_iterations'",
        ):
            make_catalog_copula_flows(
                tiny_chain, 2, progress=False,
                init_num_iterations=3, mv_normal_init_iterations=2)


def test_an_unknown_keyword_is_still_rejected(tiny_chain):
    """A misspelled hyperparameter must fail loudly, not be silently dropped."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'nnots'"):
        make_catalog_copula_flows(tiny_chain, 2, progress=False, nnots=8)


def test_the_deprecation_warning_still_blames_the_caller(tiny_chain):
    """
    ``resolve_entry_point_kwargs`` added a frame between the entry point and
    ``warnings.warn``, so the default ``stacklevel=3`` started naming a line
    inside ``petra/copula_flows.py``.  A DeprecationWarning that points at the
    library rather than the call site is invisible to the default filter, which
    only shows warnings attributed to ``__main__``.
    """
    with pytest.warns(DeprecationWarning) as caught:
        make_catalog_copula_flows(
            tiny_chain, 2, n_phases=1, init_num_iterations=1,
            threshold_samples=NO_FLOWS, progress=False)
    assert caught[0].filename == __file__
