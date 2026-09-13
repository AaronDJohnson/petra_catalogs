"""
Edge cases of :mod:`petra.posterior_chain`, :mod:`petra.initialization` and the
multivariate-normal entry point.

These are the branches that only fire on unusual input -- a cost key that is not
a number, a chain with the wrong number of axes, a sample in which every source
is absent -- and that therefore go unexercised by the happy-path tests.

The last section is different in kind: it is the cross-entry-point test of the
shared initialization preamble, parametrized over both entry points that
have one.  Per-module tests are how that preamble drifted -- the same pair of
keywords ran the univariate pass on ``make_catalog_bayesian_gaussian`` and
skipped it on the copula-flow method, and every module's own tests agreed with its own
module -- so the parity assertions live in one place and cover the whole set.
"""

import logging

import numpy as np
import pytest

import petra.bayesian_gaussian as bayesian_gaussian
import petra.copula_flows as copula_flows
import petra.initialization as initialization
import petra.make_catalog as make_catalog
from petra.initialization import (_validate_initialization_param_index,
                                  label_chain_by_histogram,
                                  run_initialization_passes, tetris_rise_nd)
from petra.make_catalog import make_catalog_mv_normal
from petra.posterior_chain import (PosteriorChain, _decode_cost_dict,
                                   _encode_cost_dict)
from petra.relabel import create_relabel_samples, prepare_chain
from petra.aux_distributions import mv_normal_aux_distribution
from petra.parametric_fits import mv_normal_fit
from petra.utils import find_prob_in_model, source_present


# --------------------------------------------------------------------------
# cost_dict encoding
# --------------------------------------------------------------------------

def test_cost_dict_encoding_keeps_unconvertible_keys_and_values():
    """A key that is not an int is stringified; a value that is not a float is kept."""
    encoded = _encode_cost_dict({"total": None})
    assert encoded == [["total", None]]
    assert _decode_cost_dict(encoded) == {"total": None}


def test_cost_dict_encoding_survives_a_feather_round_trip(tmp_path):
    pc = PosteriorChain(np.zeros((4, 2, 1)), 2, 1, cost_dict={"total": None, 2: -3.5})
    path = str(tmp_path / "pc.feather")
    pc.to_feather(path)
    assert PosteriorChain.read_feather(path).cost_dict == {"total": None, 2: -3.5}


# --------------------------------------------------------------------------
# PosteriorChain validation
# --------------------------------------------------------------------------

def test_flat_chain_with_a_leftover_is_rejected():
    with pytest.raises(ValueError, match="not a multiple of"):
        PosteriorChain(np.zeros(7), 2, 3)


def test_four_dimensional_chain_is_rejected():
    with pytest.raises(ValueError, match="must have 1, 2, or 3 dimensions"):
        PosteriorChain(np.zeros((4, 2, 3, 1)), 2, 3)


def test_boolean_dimensions_are_rejected():
    """``True`` is an ``int`` in Python; a chain width of ``True`` is a mistake."""
    with pytest.raises(ValueError, match="num_sources must be a positive integer"):
        PosteriorChain(np.zeros((4, 1, 3)), True, 3)


def test_repr_shows_the_underlying_array():
    pc = PosteriorChain(np.zeros((2, 1, 1)), 1, 1)
    assert repr(pc) == repr(np.zeros((2, 1, 1)))


def test_every_chain_writer_in_petra_moves_whole_source_rows():
    """
    Why the convention is rejected at the boundary instead of handled downstream:
    nothing inside petra can split a source row.  Every writer here moves or
    blanks whole rows, so presence read row-wise still agrees with presence read
    from any single column -- the agreement that a split row would destroy.
    """
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((12, 3, 4))
    arr[rng.random((12, 3)) < 0.4] = np.nan
    pc = PosteriorChain(arr, 3, 4, trans_dimensional=True)

    written = [
        pc.expand_chain(5).chain,                                  # NaN padding
        pc.randomize_entries(seed=1).chain,                        # per-sample shuffle
        prepare_chain(pc, 5, shuffle_seed=1).chain,                # shuffle then pad
        pc.get_chain(burn=2, thin=3),                              # burn and thin
        tetris_rise_nd(pc.chain),                                  # histogram compaction
    ]
    for chain in written:
        present = source_present(chain)
        for k in range(chain.shape[2]):
            assert np.array_equal(present, ~np.isnan(chain[:, :, k]))


# --------------------------------------------------------------------------
# histogram initialization
# --------------------------------------------------------------------------

def test_histogram_labeling_handles_a_sample_with_no_sources():
    """A sample in which every entry is NaN contributes nothing and is not an error."""
    frequency = 1.0 / (365.25 * 24 * 3600)  # roughly one frequency bin
    arr = np.full((6, 2, 1), np.nan)
    arr[:, 0, 0] = 100 * frequency
    arr[:, 1, 0] = 400 * frequency
    arr[3, :, 0] = np.nan  # sample 3 has no sources at all
    pc = PosteriorChain(arr, 2, 1, trans_dimensional=True)

    labeled = label_chain_by_histogram(pc, low_num_samples=1, num_extra_entries=2)

    assert labeled.num_sources == 2
    # The empty sample stayed empty, and no value was invented for it.
    assert np.all(np.isnan(labeled.chain[3]))


def test_histogram_labeling_compacts_sparsely_populated_labels():
    """A label holding fewer than `low_num_samples` samples is treated as sparse."""
    frequency = 1.0 / (365.25 * 24 * 3600)
    arr = np.full((8, 2, 1), np.nan)
    arr[:, 0, 0] = 100 * frequency
    arr[:2, 1, 0] = 400 * frequency  # only two samples: below low_num_samples
    pc = PosteriorChain(arr, 2, 1, trans_dimensional=True)

    labeled = label_chain_by_histogram(pc, low_num_samples=5, num_extra_entries=2)

    assert labeled.num_sources == 2
    # The well-populated label comes first.
    assert np.count_nonzero(~np.isnan(labeled.chain[:, 0, 0])) == 8
    assert np.count_nonzero(~np.isnan(labeled.chain[:, 1, 0])) == 2


# --------------------------------------------------------------------------
# entry point plumbing
# --------------------------------------------------------------------------

def test_prepare_chain_defaults_to_the_chain_width():
    pc = PosteriorChain(np.zeros((5, 3, 2)), 3, 2)
    assert prepare_chain(pc) is pc


def test_prepare_chain_hands_on_a_prob_in_model_that_describes_what_it_returns():
    """
    The shared preamble inherits whatever ``expand_chain`` and ``randomize_entries`` do.

    Both steps change which slots hold which sources -- the shuffle moves them,
    the widening adds empty ones -- and both used to carry the caller's array
    through untouched, so every entry point began with a ``prob_in_model`` that
    described neither the chain it came from nor the one it was attached to.
    """
    rng = np.random.default_rng(1)
    chain = rng.standard_normal((20, 2, 3))
    chain[::3, 1, :] = np.nan               # source 1 is absent from a third of the samples
    pc = PosteriorChain(chain, 2, 3, trans_dimensional=True,
                        prob_in_model=find_prob_in_model(chain, 2, eps=0))

    prepared = prepare_chain(pc, 4, shuffle_seed=1)

    assert prepared.num_sources == 4
    assert np.array_equal(prepared.prob_in_model,
                          find_prob_in_model(prepared.get_chain(), 4, eps=0))
    # The two slots the widening added are empty in every sample: exactly 0, not eps.
    assert prepared.prob_in_model[2] == 0.0 and prepared.prob_in_model[3] == 0.0


def test_relabel_samples_rejects_zero_iterations():
    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
    pc = PosteriorChain(np.zeros((5, 2, 1)), 2, 1)
    with pytest.raises(ValueError, match="num_iterations must be at least 1"):
        relabel(pc, max_num_sources=2, num_iterations=0)


def test_make_catalog_mv_normal_can_skip_the_univariate_initialization():
    rng = np.random.default_rng(0)
    chain = rng.normal(scale=0.2, size=(30, 2, 1)) + np.array([[0.0], [5.0]])
    catalog = make_catalog_mv_normal(PosteriorChain(chain, 2, 1), max_num_sources=2,
                                     num_iterations=3, initialization_param_index=None,
                                     progress=False)
    assert catalog.chain.shape == (30, 2, 1)
    assert np.allclose(np.sort(catalog.chain, axis=1), np.sort(chain, axis=1))


# --------------------------------------------------------------------------
# the shared initialization preamble, across every entry point that has one
# --------------------------------------------------------------------------

#: The two entry points that run :func:`petra.initialization.run_initialization_passes`,
#: each with the module attribute naming the method that follows it.  That
#: attribute is stubbed out below so the tests measure the preamble alone: with
#: the real method in place ``make_catalog_copula_flows`` would train flows to
#: decide nothing these tests assert on.
INITIALIZATION_ENTRY_POINTS = [
    (copula_flows, "make_catalog_copula_flows", "relabel_copula_flows"),
    (bayesian_gaussian, "make_catalog_bayesian_gaussian", "bayesian_relabel_loop"),
]


#: Every combination of the two switches, with the passes it must produce.
#: ``(False, 0)`` is the pair that used to mean two different things: the
#: copula-flow entry point skipped the univariate pass along with the multivariate
#: one, so ``init_with_mv_normal`` silently overrode ``initialization_param_index``.
INITIALIZATION_COMBINATIONS = [
    (True, 0, ["univariate", "mv_normal"]),
    (True, None, ["mv_normal"]),
    (False, 0, ["univariate"]),
    (False, None, []),
]


@pytest.fixture
def tiny_two_source_chain():
    """A 12-sample, two-source, one-parameter chain -- just enough to run on."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 1)) + np.array([[0.0], [8.0]])
    return PosteriorChain(chain, 2, 1, trans_dimensional=True)


def _spy_on_the_initialization_passes(monkeypatch, module, downstream):
    """
    Record which initialization passes an entry point runs, and stub its method.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the relabelers and the downstream method.
    module : module
        Module holding the entry point under test.
    downstream : str
        Name, in `module`, of the method the entry point calls after the
        preamble.  Replaced by a stub that hands the chain straight back.

    Returns
    -------
    passes : list of tuple
        One ``(name, num_iterations)`` entry per pass that ran, in order.
    """
    passes = []

    def univariate_spy(chain, **kwargs):
        assert kwargs["init_parameter_index"] is not None, "the univariate pass needs a parameter"
        passes.append(("univariate", kwargs["num_iterations"]))
        return chain

    def mv_normal_spy(chain, **kwargs):
        passes.append(("mv_normal", kwargs["num_iterations"]))
        return chain

    # Patch the two names in every module that binds them, because the entry
    # points do not agree on how they reach them: one imports them inside the
    # function body and one at module scope, and which namespace a pass is
    # looked up in is not what these tests are about.
    for holder in (initialization, make_catalog, copula_flows,
                   bayesian_gaussian):
        if hasattr(holder, "relabel_univariate_normal"):
            monkeypatch.setattr(holder, "relabel_univariate_normal", univariate_spy)
        if hasattr(holder, "relabel_mv_normal"):
            monkeypatch.setattr(holder, "relabel_mv_normal", mv_normal_spy)

    monkeypatch.setattr(module, downstream,
                        lambda posterior_chain, *args, **kwargs: posterior_chain)
    return passes


@pytest.mark.parametrize("module, entry_point_name, downstream", INITIALIZATION_ENTRY_POINTS,
                         ids=lambda value: getattr(value, "__name__", value))
@pytest.mark.parametrize("with_mv_normal, param_index, expected", INITIALIZATION_COMBINATIONS,
                         ids=["both", "mv-normal-only", "univariate-only", "neither"])
def test_the_two_initialization_switches_are_independent_everywhere(
        monkeypatch, tiny_two_source_chain, module, entry_point_name, downstream,
        with_mv_normal, param_index, expected):
    """
    One keyword, one pass -- and the same one on both entry points.

    ``initialization_param_index`` selects the univariate pass and
    ``init_with_mv_normal`` the multivariate one.  The combination that pins the
    fix is ``init_with_mv_normal=False`` with ``initialization_param_index=0``:
    the copula entry point used to skip the univariate pass as well, so a
    caller who turned off the multivariate initialization silently lost the
    univariate one too, with no warning and no
    log line.  Parametrizing over the entry points is the point of the test: a
    per-module version of it is what let the two spellings diverge.
    """
    passes = _spy_on_the_initialization_passes(monkeypatch, module, downstream)

    getattr(module, entry_point_name)(
        tiny_two_source_chain, 2, num_iterations=1, init_num_iterations=3,
        init_with_mv_normal=with_mv_normal, initialization_param_index=param_index,
        progress=False,
    )

    assert [name for name, _ in passes] == expected
    # The per-pass budget reaches every pass that ran: `init_num_iterations` is
    # spent once per pass, not once per call.
    assert all(num_iterations == 3 for _, num_iterations in passes)


@pytest.mark.parametrize("module, entry_point_name, downstream", INITIALIZATION_ENTRY_POINTS,
                         ids=lambda value: getattr(value, "__name__", value))
def test_resuming_skips_both_initialization_passes_everywhere(
        monkeypatch, tiny_two_source_chain, tmp_path, module, entry_point_name, downstream):
    """
    A checkpoint already reflects the initialization, so redoing it would undo it.

    Both switches are left at their defaults here, so only `resume_from` can be
    what stops the passes.
    """
    passes = _spy_on_the_initialization_passes(monkeypatch, module, downstream)

    getattr(module, entry_point_name)(
        tiny_two_source_chain, 2, num_iterations=1, init_num_iterations=3,
        resume_from=str(tmp_path / "iteration_1.feather"), progress=False,
    )

    assert passes == []


@pytest.mark.parametrize("with_mv_normal, param_index, logged_skip, silent_skip", [
    (False, 0, "init_with_mv_normal is False", "initialization_param_index is None"),
    (True, None, "initialization_param_index is None", "init_with_mv_normal is False"),
], ids=["univariate-only", "mv-normal-only"])
def test_a_skipped_initialization_pass_says_which_keyword_skipped_it(
        caplog, with_mv_normal, param_index, logged_skip, silent_skip):
    """
    Each skip names the keyword responsible, and only the skip that happened.

    The divergence this preamble carried was invisible from the outside: one
    keyword turned off a pass another keyword had asked for, with no error and
    no log line.  A message that named neither keyword, or that was emitted for
    a pass that did run, would leave it just as invisible.
    """
    chain = PosteriorChain(np.zeros((4, 2, 1)), 2, 1)
    with caplog.at_level(logging.INFO, logger="petra.initialization"):
        run_initialization_passes(
            chain, 2,
            univariate_relabeler=lambda pc, **kwargs: pc,
            mv_normal_relabeler=lambda pc, **kwargs: pc,
            with_mv_normal=with_mv_normal, param_index=param_index,
            num_iterations=3, progress=False,
        )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert logged_skip in logged
    assert silent_skip not in logged


# --------------------------------------------------------------------------
# initialization_param_index, checked at the boundary of every entry point
# --------------------------------------------------------------------------

#: All three entry points, because the keyword is part of the shared
#: ``make_catalog_*`` prefix and not just of the two that share the preamble.
#: The two reach the check through `run_initialization_passes`;
#: `make_catalog_mv_normal` does not call that function and carries the check
#: itself, which is exactly the entry point a per-module test would have missed.
PARAM_INDEX_ENTRY_POINTS = [
    make_catalog_mv_normal,
    copula_flows.make_catalog_copula_flows,
    bayesian_gaussian.make_catalog_bayesian_gaussian,
]


@pytest.fixture
def two_parameter_chain():
    """
    A 12-sample, two-source, *two*-parameter chain.

    Two parameters is what makes ``initialization_param_index=-1`` interesting:
    with one parameter it wraps onto the column the caller meant anyway.
    """
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(12, 2, 2)) + np.array([[0.0, 0.0], [8.0, 8.0]])
    return PosteriorChain(chain, 2, 2, trans_dimensional=True)


@pytest.mark.parametrize("entry_point", PARAM_INDEX_ENTRY_POINTS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("param_index", [-1, 2], ids=["wrapped", "past-the-end"])
def test_an_initialization_param_index_off_the_chain_is_refused(
        entry_point, two_parameter_chain, param_index):
    """
    Neither way of getting this keyword wrong used to report itself.

    ``-1`` is the worse one: numpy wraps it onto the last parameter, so the
    univariate pass sorted on a column the caller never named and the run
    finished with a plausible-looking wrong labeling.  ``2`` did raise, but as a
    bare ``IndexError`` from inside the fit, several frames from the keyword
    that was wrong.  Both are now a `ValueError` from the entry point, before
    any fitting -- which is why this test can afford to run all three.
    """
    with pytest.raises(ValueError,
                       match=r"initialization_param_index must be None or in \[0, 2\)"):
        entry_point(two_parameter_chain, 2, num_iterations=1, progress=False,
                    initialization_param_index=param_index)


@pytest.mark.parametrize("param_index", [0, 1, None], ids=["first", "last", "none"])
def test_every_column_of_the_chain_and_none_stay_valid(param_index):
    """
    The check must not be stricter than the keyword.

    ``None`` is a request to skip the univariate pass, not an omitted value, and
    the last column is as addressable as the first.
    """
    assert _validate_initialization_param_index(2, param_index) is None


def test_an_initialization_param_index_off_the_chain_is_refused_when_resuming():
    """
    The check runs before the `resume_from` early return, on purpose.

    A resumed run is spelled with the same keywords as the run it continues, so
    an index that was nonsense the first time is still nonsense; reporting it
    only on the run that would have used it would make the mistake appear to
    come and go with an unrelated keyword.
    """
    chain = PosteriorChain(np.zeros((4, 2, 2)), 2, 2)
    with pytest.raises(ValueError, match="initialization_param_index"):
        run_initialization_passes(
            chain, 2,
            univariate_relabeler=lambda pc, **kwargs: pc,
            mv_normal_relabeler=lambda pc, **kwargs: pc,
            with_mv_normal=True, param_index=-1, num_iterations=3,
            resume_from="run/iteration_3.feather", progress=False,
        )
