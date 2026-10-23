"""Tests of the multivariate-normal entry point, `relabel_mv_normal`, and the public surface of the package.

The ref-pinned cases build their chains with the ``tests.helpers.ref_*`` constructors and compare with
``tests.helpers.REF``, the behaviour reference taken before the restructure.
"""

import ast
import importlib.resources
import importlib.util
import inspect
import logging
import os
import runpy
import subprocess
import sys
import tomllib
import typing
import warnings
from pathlib import Path

import numpy as np
import pytest
from packaging.requirements import Requirement

import petra
from petra import checkpoint
from petra import make_catalog as make_catalog_module
from petra import relabel as relabel_module
from petra.initialization import relabel_univariate_normal, run_initialization_passes
from petra.make_catalog import make_catalog_mv_normal, relabel_mv_normal
from petra.posterior_chain import PosteriorChain
from petra.relabel import run_relabeling_loop
from petra.samples_io import (load_samples, load_samples_fixed_num_sources, load_samples_product_space,
                              load_samples_ucbmcmc)
from petra.utils import find_prob_in_model
from tests.helpers import (BUDGET_WARNING, EMPTY_LABEL_WARNING, REF, case_a_chain, case_b_ref_chain,
                           case_c_chain, case_c_reference, chain_hash, cut_short, golden_chain, label_purity,
                           never_cooccurring_chain, ref_purity, same_catalog, two_clusters, widening_chain)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFTEST = Path(__file__).with_name("conftest.py")

CASE_A_MEANS = [np.full(2, v) for v in (0.0, 10.0, 20.0, 30.0)]
CASE_B_MEANS = [np.full(8, 10.0 * k) for k in range(5)]

#: Three well-separated 2-D clusters, many times their width apart.
TRUE_MEANS = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 1.5]])
CLUSTER_SCALE = 0.25

INPUT_PROB_IN_MODEL = [1.0, 1.0, 0.6]
INPUT_COST_DICT = {3: -1.0}

PATH = str | os.PathLike[str]
OPTIONAL_PATH = PATH | None


def _per_sample_permutation(catalog: np.ndarray, chain: np.ndarray) -> bool:
    """Whether every sample of `catalog` holds the rows of the same sample of `chain`, padded with NaN rows."""
    if catalog.shape[1] > chain.shape[1]:
        padding = np.full((chain.shape[0], catalog.shape[1] - chain.shape[1], chain.shape[2]), np.nan)
        chain = np.concatenate([chain, padding], axis=1)
    for found, given in zip(catalog, chain):
        found = found[np.lexsort(found.T[::-1])]
        given = given[np.lexsort(given.T[::-1])]
        if not np.array_equal(found, given, equal_nan=True):
            return False
    return True


def _shuffled_clusters(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw `TRUE_MEANS` with the source axis permuted in every sample, and the true source of every row."""
    rng = np.random.default_rng(seed)
    n_sources, n_params = TRUE_MEANS.shape
    clean = rng.normal(loc=TRUE_MEANS, scale=CLUSTER_SCALE, size=(n_samples, n_sources, n_params))
    order = np.array([rng.permutation(n_sources) for _ in range(n_samples)])
    return np.take_along_axis(clean, order[:, :, None], axis=1), order


def _independently_absent(n_sources: int, absent: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw 200 shuffled samples of sources 10 apart in two parameters, each absent with probability `absent`."""
    rng = np.random.default_rng(seed)
    values = np.arange(n_sources)[None, :, None] * 10.0 + rng.normal(size=(200, n_sources, 2))
    order = np.array([rng.permutation(n_sources) for _ in range(200)])
    chain = np.take_along_axis(values, order[:, :, None], axis=1)
    true_ids = order.copy()
    missing = rng.random((200, n_sources)) < absent
    chain[missing] = np.nan
    true_ids[missing] = -1
    return chain, true_ids


def _packed(seed: int, n_sources: int, n_params: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw 200 samples of sources 10 apart, each present with probability 0.7, packed into the first slots."""
    rng = np.random.default_rng(seed)
    values = np.arange(n_sources)[None, :, None] * 10.0 + rng.normal(size=(200, n_sources, n_params))
    is_present = rng.random((200, n_sources)) < 0.7
    chain = np.full((200, n_sources, n_params), np.nan)
    true_ids = np.full((200, n_sources), -1)
    for i in range(200):
        sources = np.flatnonzero(is_present[i])
        rng.shuffle(sources)
        chain[i, :sources.size] = values[i, sources]
        true_ids[i, :sources.size] = sources
    return chain, true_ids


def _galactic_binaries() -> tuple[np.ndarray, np.ndarray]:
    """Draw 300 shuffled samples of three sources at galactic-binary scales, each absent in 20% of them."""
    rng = np.random.default_rng(0)
    centers = np.array([[3.0e-3, 1.0e-17, 2.0e-22, 0.5],
                        [3.0e-3 + 5e-9, 1.2e-17, 1.5e-22, 1.5],
                        [3.0e-3 + 1e-8, 0.8e-17, 3.0e-22, 2.5]])
    widths = np.array([1.5e-9, 1e-18, 3e-23, 0.3])
    values = centers[None] + rng.normal(size=(300, 3, 4)) * widths
    order = np.array([rng.permutation(3) for _ in range(300)])
    chain = np.take_along_axis(values, order[:, :, None], axis=1)
    true_ids = order.copy()
    missing = rng.random((300, 3)) < 0.2
    chain[missing] = np.nan
    true_ids[missing] = -1
    return chain, true_ids


def _integer_chain() -> np.ndarray:
    """Two integer-valued sources in two parameters, swapped in about half the samples."""
    rng = np.random.default_rng(0)
    base = np.stack([rng.integers(0, 10, size=(60, 2)), rng.integers(1000, 1010, size=(60, 2))], axis=1)
    swap = rng.random(60) < 0.5
    base[swap] = base[swap][:, ::-1]
    return base


def _gappy_chain() -> np.ndarray:
    """Three sources; source 2 absent from half the samples, source 1 from a quarter."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(8, 3, 2)) + np.array([0.0, 5.0, 10.0])[None, :, None]
    chain[:4, 2, :] = np.nan
    chain[:2, 1, :] = np.nan
    return chain


def _weak_stuck_chain() -> np.ndarray:
    """A and C in every sample; weak source B in slot 3 of 52 samples, twelve of them three repeated states."""
    rng = np.random.default_rng(2)
    chain = np.full((200, 4, 3), np.nan)
    chain[:, 0] = rng.standard_normal((200, 3))
    chain[:, 2] = rng.standard_normal((200, 3)) + 12.0
    states = 6.0 + rng.standard_normal((3, 3))
    chain[:12, 3] = states[np.arange(12) % 3]
    chain[100:140, 3] = 6.0 + rng.standard_normal((40, 3))
    return chain


def _stuck_chain(num_distinct: int, seed: int) -> np.ndarray:
    """A and C in every sample; B's stuck stretch of `num_distinct` states in slot 1, its forty draws in slot 3."""
    rng = np.random.default_rng(seed)
    chain = np.full((200, 4, 3), np.nan)
    chain[:, 0, :] = rng.standard_normal((200, 3))
    chain[:, 2, :] = rng.standard_normal((200, 3)) + 12.0
    states = 6.0 + rng.standard_normal((num_distinct, 3))
    chain[:12, 1, :] = states[np.arange(12) % num_distinct]
    chain[100:140, 3, :] = 6.0 + rng.standard_normal((40, 3))
    return chain


def _fixed_inside_chain(seed: int) -> np.ndarray:
    """Source A ~ N(0, I), source B held at 0.3 in every sample inside A's spread, and C far away."""
    rng = np.random.default_rng(seed)
    chain = np.empty((400, 3, 2))
    chain[:, 0] = rng.normal(size=(400, 2))
    chain[:, 1] = 0.3
    chain[:, 2] = rng.normal(size=(400, 2)) * 0.5 + 6.0
    return chain


def _write_ucbmcmc_chains(folder: Path) -> None:
    """Write shuffled 2- and 3-source UCBMCMC files of `TRUE_MEANS`, 40 samples each."""
    rng = np.random.default_rng(11)
    for n_sources in (2, 3):
        block = rng.normal(loc=TRUE_MEANS[:n_sources], scale=CLUSTER_SCALE, size=(40, n_sources, 2))
        for i in range(40):
            block[i] = block[i, rng.permutation(n_sources), :]
        np.savetxt(folder / f"dimension_chain.dat.{n_sources}", block.reshape(-1, 2))


def _immutability_input() -> tuple[np.ndarray, np.ndarray, PosteriorChain]:
    """Return a 40 x 3 x 2 float64 array, its copy, and the chain built on it without a copy."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(40, 3, 2)) + np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]])
    arr[rng.random(40) < 0.4, 2, :] = np.nan
    pc = PosteriorChain(arr, 3, 2, trans_dimensional=True, prob_in_model=np.array(INPUT_PROB_IN_MODEL),
                        cost_dict=dict(INPUT_COST_DICT))
    return arr, arr.copy(), pc


def _assert_input_untouched(arr: np.ndarray, snapshot: np.ndarray, pc: PosteriorChain) -> None:
    """Assert the caller's array, prob_in_model and cost_dict are byte for byte as they went in."""
    assert arr.dtype == snapshot.dtype and arr.tobytes() == snapshot.tobytes()
    assert pc.chain.tobytes() == snapshot.tobytes()
    assert pc.prob_in_model.tobytes() == np.array(INPUT_PROB_IN_MODEL).tobytes()
    assert pc.cost_dict == INPUT_COST_DICT


def _type_checking_imports() -> dict[str, str]:
    """Map each name ``petra/__init__.py`` imports under ``if TYPE_CHECKING:`` to its module."""
    tree = ast.parse(Path(petra.__file__).read_text(encoding="utf-8"))
    imported: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            for statement in node.body:
                assert isinstance(statement, ast.ImportFrom) and statement.level == 1
                for alias in statement.names:
                    imported[alias.asname or alias.name] = f"petra.{statement.module}"
    return imported


def _signature(function: typing.Callable[..., typing.Any]) -> list[tuple[str, str, typing.Any]]:
    """Return each parameter's name, kind and default."""
    return [(p.name, p.kind.name, p.default) for p in inspect.signature(function).parameters.values()]


@pytest.fixture(scope="module")
def case_c8_start_costs() -> tuple[float, float]:
    """The costs of case C seed 8 from one start and from three."""
    pc = PosteriorChain(case_c_chain(8)[0], 5, 3, True)
    return (make_catalog_mv_normal(pc, 5, progress=False).cost_dict[5],
            make_catalog_mv_normal(pc, 5, num_starts=3, progress=False).cost_dict[5])


def test_relabel_mv_normal_one_iteration_golden_values():
    """One iteration on the uniform golden chain gives the reference labeling and cost."""
    relabeled = relabel_mv_normal(golden_chain(), 2, num_iterations=1, split_moves=False, progress=False)

    ref = REF["golden_one_iter"]
    assert relabeled.cost_dict == {2: pytest.approx(ref["cost_dict"]["2"], abs=1e-12)}
    np.testing.assert_array_equal(relabeled.get_chain()[:3], ref["first3"])
    np.testing.assert_array_equal(relabeled.get_chain()[-3:], ref["last3"])


@pytest.mark.parametrize("key, settings", [("lisa_notebook", {"num_iterations": 3, "init_num_iterations": 3}),
                                           ("lisa_default", {})], ids=["cell", "default"])
def test_the_lisa_notebook_cell_and_default_run(lisa_sub, key, settings):
    """The LISA example gives the reference catalogs; split moves change the default run only for the better."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        catalog = make_catalog_mv_normal(lisa_sub, 20, progress=False, **settings)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        plain = make_catalog_mv_normal(lisa_sub, 20, split_moves=False, progress=False, **settings)

    ref = REF[key]
    assert catalog.cost_dict[20] == pytest.approx(ref["cost_dict"]["20"], rel=1e-12)
    np.testing.assert_allclose(catalog.prob_in_model, ref["prob_in_model"], rtol=0, atol=1e-12)
    assert chain_hash(catalog.get_chain()) == ref["chain_hash"]
    assert list(catalog.shape) == ref["shape"]
    if key == "lisa_notebook":
        assert len(caught) == 1
        assert str(caught[0].message).startswith("The budget of num_iterations=3 " + BUDGET_WARNING)
        assert same_catalog(catalog, plain)
    else:
        assert caught == []
        certain = np.flatnonzero(plain.prob_in_model == 1)
        assert certain.size == 10 and np.array_equal(np.flatnonzero(catalog.prob_in_model == 1), certain)
        for label in certain:
            rows = plain.chain[:, label]
            homes = {int(np.flatnonzero(np.all(catalog.chain[i] == rows[i], axis=-1))[0]) for i in range(len(rows))}
            assert len(homes) == 1, (label, homes)
        assert catalog.cost_dict[20] < plain.cost_dict[20]


def test_case_a_widening_fills_every_added_slot_and_round_trips_through_feather(tmp_path):
    """Four sources packed two wide fill four labels without a warning, and the catalog survives Feather."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        catalog = make_catalog_mv_normal(case_a_chain(), 4, progress=False)

    ref = REF["case_A"]
    assert ref_purity(catalog, CASE_A_MEANS) == ref["purity"] == 1.0
    assert catalog.cost_dict[4] == pytest.approx(ref["cost_dict"]["4"], rel=1e-12)
    np.testing.assert_allclose(catalog.prob_in_model, ref["prob_in_model"], rtol=0, atol=1e-12)

    catalog.to_feather(tmp_path / "catalog.feather")
    back = PosteriorChain.read_feather(tmp_path / "catalog.feather")

    ref = REF["feather"]
    assert np.array_equal(back.get_chain(), catalog.get_chain(), equal_nan=True) is ref["same_chain"]
    assert np.allclose(back.prob_in_model, catalog.prob_in_model) is ref["same_prob"]
    assert back.cost_dict == {4: pytest.approx(ref["cost_dict"]["4"], rel=1e-12)}
    assert [back.num_sources, back.num_params_per_source] == ref["dims"]


def test_without_split_moves_the_widening_trap_is_what_it_was():
    """``split_moves=False`` is the plain search: two added labels stay empty, at the cost recorded before F003."""
    chain, _ = widening_chain(1000)
    with pytest.warns(UserWarning, match=EMPTY_LABEL_WARNING):
        catalog = make_catalog_mv_normal(PosteriorChain(chain, 2, 2, trans_dimensional=True), 4,
                                         split_moves=False, progress=False)

    assert catalog.cost_dict[4] == pytest.approx(7.246324690080283, rel=1e-12)
    np.testing.assert_allclose(catalog.prob_in_model, [0.765, 0.356, 0.0, 0.0], atol=5e-4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_packed_chain_is_recovered_without_initialization(seed):
    """Packed chains of five sources in eight parameters, and of three in two, come back one source per label."""
    catalog = make_catalog_mv_normal(case_b_ref_chain(), 5, initialization_param_index=None, shuffle_seed=seed,
                                     progress=False)

    ref = REF["case_B"][str(seed)]
    assert ref_purity(catalog, CASE_B_MEANS) == ref["purity"] == 1.0
    assert catalog.cost_dict[5] == pytest.approx(ref["cost"], rel=1e-12)

    chain, true_ids = _packed(seed, n_sources=3, n_params=2)
    small = make_catalog_mv_normal(PosteriorChain(chain, 3, 2, trans_dimensional=True), 3,
                                   initialization_param_index=None, progress=False)
    assert label_purity(small.chain, chain, true_ids) >= 0.99


def test_case_c_seed_14_needs_split_moves():
    """Seed 14 reaches the cost of its true labeling with split moves and stays merged without them."""
    chain, true_ids = case_c_chain(14)
    reference = case_c_reference(chain, true_ids).cost_dict[5]

    catalog = make_catalog_mv_normal(PosteriorChain(chain, 5, 3, True), 5, progress=False)
    plain = make_catalog_mv_normal(PosteriorChain(chain, 5, 3, True), 5, split_moves=False, progress=False)

    assert catalog.cost_dict[5] == pytest.approx(reference, abs=1e-4)
    assert label_purity(catalog.chain, chain, true_ids) >= 0.95
    assert plain.cost_dict[5] > reference + 0.5
    assert label_purity(plain.chain, chain, true_ids) < 0.95


@pytest.mark.parametrize("split_exclusive", [False, True])
def test_never_seen_together_splits_only_when_exclusive_and_stuck_stretches_stay_whole(caplog, split_exclusive):
    """Only split_exclusive splits sources never seen together; no setting cuts a stuck or fixed source (F008)."""
    pc = PosteriorChain(never_cooccurring_chain(), 2, 2, trans_dimensional=True)
    if split_exclusive:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            catalog = make_catalog_mv_normal(pc, 3, split_exclusive=True, progress=False)
        np.testing.assert_allclose(np.sort(catalog.prob_in_model), [0.5, 0.5, 1.0], atol=0.05)
        assert max(np.nanstd(catalog.chain[:, label, 0]) for label in range(3)) < 1.5
    else:
        with pytest.warns(UserWarning, match=EMPTY_LABEL_WARNING) as caught:
            catalog = make_catalog_mv_normal(pc, 3, progress=False)
        (message,) = [str(warning.message) for warning in caught if EMPTY_LABEL_WARNING in str(warning.message)]
        assert "split_exclusive=True" in message
        np.testing.assert_array_equal(np.sort(catalog.prob_in_model), [0.0, 1.0, 1.0])
        assert max(np.std(catalog.chain[:, label, 0]) for label in np.flatnonzero(catalog.prob_in_model)) > 15

    weak = _weak_stuck_chain()
    with caplog.at_level(logging.INFO, logger="petra.relabel"):
        catalog = make_catalog_mv_normal(PosteriorChain(weak, 4, 3, True), 4, initialization_param_index=None,
                                         num_iterations=50, split_exclusive=split_exclusive, progress=False)
    np.testing.assert_array_equal(catalog.get_chain(), weak)
    assert ("Split move rejected after" in caplog.text) is split_exclusive

    stuck = _stuck_chain(num_distinct=3, seed=1)
    truth = stuck.copy()
    truth[:12, 3, :], truth[:12, 1, :] = stuck[:12, 1, :], np.nan
    settings = dict(initialization_param_index=None, num_iterations=50, split_exclusive=split_exclusive,
                    progress=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        catalog = make_catalog_mv_normal(PosteriorChain(stuck, 4, 3, True), 4, **settings).get_chain()
        from_truth = make_catalog_mv_normal(PosteriorChain(truth, 4, 3, True), 4, **settings).get_chain()
    np.testing.assert_array_equal(from_truth, truth)
    label_of_b = np.argmax(np.all(np.abs(catalog - 6.0) < 4.0, axis=2).sum(axis=0))
    assert all(np.array_equal(catalog[i, label_of_b], stuck[i, 1]) for i in range(12))

    settings = dict(num_iterations=50, init_num_iterations=50, split_exclusive=split_exclusive, progress=False)
    fixed = _fixed_inside_chain(seed=0)
    catalog = make_catalog_mv_normal(PosteriorChain(fixed, 3, 2), 3, **settings).get_chain()
    np.testing.assert_array_equal(catalog, fixed)
    fixed = _fixed_inside_chain(seed=1)
    catalog = make_catalog_mv_normal(PosteriorChain(fixed, 3, 2), 3, shuffle_seed=1, **settings).get_chain()
    assert np.sort(np.all(catalog == 0.3, axis=2).sum(axis=0)).tolist() == [0, 0, 400]


@pytest.mark.parametrize("costs, kept", [((1.0, 1.0, 1.0), 0), ((2.0, 1.0, 1.0), 1), ((1.0, 2.0, 0.5), 2)])
def test_several_starts_keep_the_earliest_cheapest_catalog(monkeypatch, case_c8_start_costs, costs, kept):
    """A later start wins only when strictly cheaper; start k is shuffled with seed + k, or k without a seed."""
    for shuffle_seed, seeds in ((None, [None, 1, 2]), (7, [7, 8, 9])):
        returned: list[PosteriorChain] = []
        seen: list[int | None] = []

        def stub(chain, **kwargs):
            seen.append(kwargs["run_info"]["shuffle_seed"])
            returned.append(PosteriorChain(chain.get_chain(), 2, 1, True, cost_dict={2: costs[len(returned)]}))
            return returned[-1]

        monkeypatch.setattr(make_catalog_module, "relabel_mv_normal", stub)
        best = make_catalog_mv_normal(PosteriorChain(two_clusters(), 2, 1), 2, num_starts=3, shuffle_seed=shuffle_seed,
                                      initialization_param_index=None, progress=False)

        assert len(returned) == 3 and best is returned[kept]
        assert seen == seeds

    one_start, three_starts = case_c8_start_costs
    assert three_starts <= one_start


@pytest.mark.parametrize("settings, anchor", [
    pytest.param({"num_starts": 0}, "must be an integer >=", id="num_starts-0"),
    pytest.param({"num_starts": -1}, "must be an integer >=", id="num_starts-negative"),
    pytest.param({"num_starts": 1.5}, "must be an integer >=", id="num_starts-float"),
    pytest.param({"num_starts": True}, "must be an integer >=", id="num_starts-bool"),
    pytest.param({"num_starts": "2"}, "must be an integer >=", id="num_starts-str"),
    pytest.param({"num_starts": None}, "must be an integer >=", id="num_starts-None"),
    pytest.param({"num_starts": 2}, "num_starts=", id="num_starts-with-checkpoint_dir"),
    pytest.param({"num_starts": 2, "resume_from": "ckpt"}, "num_starts=", id="num_starts-with-resume_from"),
    pytest.param({"init_num_iterations": 0, "initialization_param_index": None}, "must be an integer >=",
                 id="init_num_iterations-0-without-warm-start"),
    pytest.param({"max_num_sources": None}, "must be an integer >=", id="max_num_sources-None"),
    pytest.param({"keep_checkpoints": 0}, "must be an integer >=", id="keep_checkpoints-0"),
    pytest.param({"eps": 0.6}, "eps must be finite and between 0 and 0.5", id="eps-0.6"),
    pytest.param({"num_iterations": np.int64(3), "init_num_iterations": np.int64(3)}, None, id="numpy-budgets"),
    pytest.param({"initialization_param_index": None}, None, id="no-warm-start"),
])
def test_entry_point_arguments_are_checked_before_any_work(tmp_path, monkeypatch, settings, anchor):
    """A bad argument raises before any directory, shuffle, widening or relabeling; good ones run."""
    calls: list[str] = []

    def record(name, real):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return real(*args, **kwargs)
        return wrapped

    for name in ("prepare_chain", "relabel_univariate_normal", "relabel_mv_normal"):
        monkeypatch.setattr(make_catalog_module, name, record(name, getattr(make_catalog_module, name)))
    settings = dict(settings)
    max_num_sources = settings.pop("max_num_sources", 2)
    pc = PosteriorChain(two_clusters(), 2, 1)
    target = tmp_path / "ckpt"

    if anchor is None:
        catalog = make_catalog_mv_normal(pc, max_num_sources, split_moves=False, progress=False, **settings)
        warm_start = settings.get("initialization_param_index", 0) is not None
        assert calls == ["prepare_chain"] + ["relabel_univariate_normal"] * warm_start + ["relabel_mv_normal"]
        assert _per_sample_permutation(catalog.get_chain(), pc.get_chain())
        return
    if "resume_from" in settings:
        settings["resume_from"] = target
    else:
        settings["checkpoint_dir"] = target
    with pytest.raises(ValueError, match=anchor):
        make_catalog_mv_normal(pc, max_num_sources, progress=False, **settings)
    assert calls == []
    assert not target.exists()


@pytest.mark.parametrize("chain, settings, hint, not_hint", [
    pytest.param(None, {"split_moves": False}, " and split_moves=True", "split_exclusive=True", id="plain"),
    pytest.param(None, {}, "split_exclusive=True", " and split_moves=True", id="split-moves"),
    pytest.param(None, {"num_starts": 3}, "more starts than the num_starts=3", "num_starts > 1", id="three-starts"),
    pytest.param(np.concatenate([two_clusters(), np.full((60, 1, 1), np.nan)], axis=1), {}, None, None,
                 id="empty-in-the-input"),
])
def test_an_added_label_left_empty_is_reported_with_the_right_hint(chain, settings, hint, not_hint):
    """A label that widening added and the search left empty is reported; one empty in the input is not."""
    if chain is None:
        with pytest.warns(UserWarning, match=EMPTY_LABEL_WARNING) as caught:
            catalog = make_catalog_mv_normal(PosteriorChain(two_clusters(), 2, 1), 3, progress=False, **settings)
        (message,) = [str(warning.message) for warning in caught if EMPTY_LABEL_WARNING in str(warning.message)]
        assert hint in message and not_hint not in message
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            catalog = make_catalog_mv_normal(PosteriorChain(chain, 3, 1, True), 3, progress=False, **settings)
    assert np.count_nonzero(catalog.prob_in_model == 0) == 1


def test_eps_reaches_every_cost_matrix_of_both_passes(monkeypatch):
    """The caller's eps clips the probabilities of the warm start, the mv pass and every split evaluation (F042)."""
    eps = 1e-3
    phase = ["mv"]
    seen: dict[str, list[np.ndarray]] = {}
    real_one_iteration = relabel_module.relabel_samples_one_iteration

    def spy(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix, progress=True):
        seen.setdefault(phase[-1], []).append(np.array(prob_in_model, copy=True))
        return real_one_iteration(chain, aux_parameters, prob_in_model, max_num_sources, compute_cost_matrix,
                                  progress=progress)

    def in_phase(name, real):
        def wrapped(*args, **kwargs):
            phase.append(name)
            try:
                return real(*args, **kwargs)
            finally:
                phase.pop()
        return wrapped

    monkeypatch.setattr(relabel_module, "relabel_samples_one_iteration", spy)
    monkeypatch.setattr(relabel_module, "_evaluate_split", in_phase("split", relabel_module._evaluate_split))
    monkeypatch.setattr(make_catalog_module, "relabel_univariate_normal",
                        in_phase("univariate", make_catalog_module.relabel_univariate_normal))

    # A is in every sample and the fourth label in none, so every clipped vector holds both bounds.
    with pytest.warns(UserWarning, match=EMPTY_LABEL_WARNING):
        make_catalog_mv_normal(PosteriorChain(never_cooccurring_chain(), 2, 2, True), 4, split_exclusive=True,
                               eps=eps, progress=False)

    assert sorted(seen) == ["mv", "split", "univariate"]
    for name, arrays in seen.items():
        for prob_in_model in arrays:
            assert prob_in_model.min() == eps and prob_in_model.max() == 1.0 - eps, name
    for entry_point in (make_catalog_mv_normal, petra.make_catalog_copula_flows):
        assert inspect.signature(entry_point).parameters["eps"].default == 1e-06


@pytest.mark.filterwarnings(f"ignore:max_num_sources=.* {EMPTY_LABEL_WARNING}:UserWarning")
def test_the_callers_input_is_never_modified_and_a_resume_does_not_prepare_it(tmp_path, monkeypatch):
    """Fresh, checkpointed and resumed runs leave the input alone; a resume neither prepares nor warms it (F045)."""
    arr, snapshot, pc = _immutability_input()
    settings = dict(shuffle_seed=7, init_num_iterations=3, progress=False)

    make_catalog_mv_normal(pc, 5, num_iterations=3, **settings)
    _assert_input_untouched(arr, snapshot, pc)
    with cut_short():
        make_catalog_mv_normal(pc, 4, num_iterations=1, checkpoint_dir=tmp_path, **settings)
    _assert_input_untouched(arr, snapshot, pc)

    def refuse(*args, **kwargs):
        raise AssertionError("a resume must not shuffle, widen or initialize its input")

    monkeypatch.setattr(PosteriorChain, "randomize_entries", refuse)
    monkeypatch.setattr(PosteriorChain, "expand_chain", refuse)
    monkeypatch.setattr(make_catalog_module, "prepare_chain", refuse)
    monkeypatch.setattr(make_catalog_module, "relabel_univariate_normal", refuse)
    resumed = make_catalog_mv_normal(pc, 4, num_iterations=5, resume_from=tmp_path, **settings)

    assert resumed.shape == (40, 4, 2)
    _assert_input_untouched(arr, snapshot, pc)


def _recovery_case(name: str) -> tuple[np.ndarray, np.ndarray | None, typing.Any, int, dict[str, typing.Any]]:
    """Return a case's chain, its true sources, the twin it must match, the width and the run settings."""
    if name == "clusters":
        chain, true_ids = _shuffled_clusters(60, seed=1)
        return chain, true_ids, None, 3, dict(num_iterations=20, init_num_iterations=20, shuffle_seed=3)
    if name.startswith("absent"):
        _, n_sources, absent = name.split("-")
        chain, true_ids = _independently_absent(int(n_sources), float(absent))
        return chain, true_ids, None, int(n_sources), {}
    if name == "float32":
        chain, true_ids = _galactic_binaries()
        single = chain.astype(np.float32)
        return single, true_ids, single.astype(np.float64), 3, dict(num_iterations=30)
    if name.startswith("integer"):
        chain = _integer_chain()
        return chain, None, chain.astype(float), int(name[-1]), dict(num_iterations=5, init_num_iterations=5)
    chain = _gappy_chain()
    masked = np.ma.array(np.nan_to_num(chain), mask=np.isnan(chain))
    return masked, None, chain, 3, dict(num_iterations=5, init_num_iterations=5)


@pytest.mark.filterwarnings(f"ignore:max_num_sources=.* {EMPTY_LABEL_WARNING}:UserWarning")
@pytest.mark.parametrize("name", ["clusters", "absent-3-0.3", "absent-3-0.5", "absent-4-0.3", "absent-5-0.3",
                                  "float32", "integer-width-2", "integer-width-3", "masked"])
def test_recovery_of_absent_float32_integer_and_masked_chains(name):
    """Shuffled and absent sources are recovered; float32, integer and masked chains give their twin's catalog."""
    chain, true_ids, twin, width, settings = _recovery_case(name)
    num_sources = chain.shape[1]
    catalog = make_catalog_mv_normal(PosteriorChain(chain, num_sources, chain.shape[2], True), width,
                                     progress=False, **settings)

    assert catalog.chain.dtype == np.float64
    as_float = np.ma.filled(np.ma.asarray(chain).astype(np.float64), np.nan)
    assert _per_sample_permutation(catalog.get_chain(), as_float)
    assert np.array_equal(catalog.prob_in_model, find_prob_in_model(catalog.get_chain(), width, eps=0))
    if true_ids is not None:
        assert label_purity(catalog.chain, as_float, true_ids) >= 0.99
        occupancy = np.sort([np.count_nonzero(true_ids == k) / len(true_ids) for k in range(num_sources)])
        np.testing.assert_allclose(np.sort(catalog.prob_in_model), occupancy, atol=0.01)
    if twin is not None:
        from_twin = make_catalog_mv_normal(PosteriorChain(twin, num_sources, chain.shape[2], True), width,
                                           progress=False, **settings)
        assert same_catalog(catalog, from_twin)


def test_load_samples_to_catalog_to_feather_journey(tmp_path):
    """A UCBMCMC directory is loaded, cataloged, written to Feather and read back unchanged."""
    (tmp_path / "chains").mkdir()
    _write_ucbmcmc_chains(tmp_path / "chains")

    pc = petra.load_samples(tmp_path / "chains", num_params_per_source=2)
    catalog = petra.make_catalog_mv_normal(pc, 3, num_iterations=20, init_num_iterations=20, progress=False)
    catalog.to_feather(tmp_path / "catalog.feather")
    back = PosteriorChain.read_feather(tmp_path / "catalog.feather")

    assert pc.shape == (80, 3, 2)
    assert np.array_equal(back.chain, catalog.chain, equal_nan=True)
    assert np.array_equal(back.prob_in_model, catalog.prob_in_model)
    assert np.array_equal(back.prob_in_model, find_prob_in_model(back.chain, 3, eps=0))
    assert back.cost_dict == catalog.cost_dict


def test_the_public_surface_signatures_and_packaging(monkeypatch):
    """The package exports five names lazily and typed, with flat signatures, Path-typed paths and a pinned backend."""
    assert petra.__all__ == ["PosteriorChain", "load_samples", "make_catalog_mv_normal", "make_catalog_copula_flows",
                             "__version__"]
    saved = {name: petra.__dict__.get(name) for name in petra._LAZY_ATTRS}
    try:
        for name in petra._LAZY_ATTRS:
            assert callable(petra.__getattr__(name)), name
    finally:
        for name, value in saved.items():
            if value is None:
                petra.__dict__.pop(name, None)
            else:
                petra.__dict__[name] = value
    with pytest.raises(AttributeError):
        petra.__getattr__("not_an_entry_point")
    assert set(petra.__all__) | set(petra._LAZY_ATTRS) <= set(dir(petra))

    code = ("import sys, petra\n"
            "assert 'jax' not in sys.modules, 'import petra imported jax'\n"
            "assert 'make_catalog_mv_normal' not in vars(petra), 'an entry point was resolved eagerly'\n"
            "assert callable(petra.make_catalog_mv_normal)\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO_ROOT, capture_output=True, text=True,
                   timeout=120)

    marker = importlib.resources.files("petra").joinpath("py.typed")
    assert marker.is_file() and marker.read_text() == ""
    assert _type_checking_imports() == petra._LAZY_ATTRS

    keyword_only = [("num_iterations", 200), ("init_num_iterations", 200), ("initialization_param_index", 0),
                    ("shuffle_seed", None), ("rng_seed", 999), ("checkpoint_dir", None), ("resume_from", None),
                    ("progress", True), ("eps", 1e-06), ("keep_checkpoints", None), ("split_moves", True),
                    ("split_exclusive", False), ("num_starts", 1)]
    empty = inspect.Parameter.empty
    assert _signature(make_catalog_mv_normal) == (
        [("posterior_chain", "POSITIONAL_OR_KEYWORD", empty), ("max_num_sources", "POSITIONAL_OR_KEYWORD", empty)]
        + [(name, "KEYWORD_ONLY", default) for name, default in keyword_only])
    assert _signature(relabel_mv_normal) == (
        [("posterior_chain", "POSITIONAL_OR_KEYWORD", empty)]
        + [(name, "POSITIONAL_OR_KEYWORD", default) for name, default in (
            ("max_num_sources", None), ("num_iterations", 20), ("eps", 1e-06), ("checkpoint_dir", None),
            ("resume_from", None), ("progress", True))]
        + [(name, "KEYWORD_ONLY", default) for name, default in (
            ("run_info", None), ("keep_checkpoints", None), ("split_moves", True), ("split_exclusive", False))])
    pc = PosteriorChain(two_clusters(), 2, 1)
    for keyword in ("n_phases", "mv_normal_init", "mv_normal_init_iterations", "initialization", "chain",
                    "num_sources"):
        with pytest.raises(TypeError):
            make_catalog_mv_normal(pc, 2, progress=False, **{keyword: 1})

    path_parameters = [
        (make_catalog_mv_normal, "checkpoint_dir", OPTIONAL_PATH),
        (make_catalog_mv_normal, "resume_from", OPTIONAL_PATH),
        (relabel_mv_normal, "checkpoint_dir", OPTIONAL_PATH), (relabel_mv_normal, "resume_from", OPTIONAL_PATH),
        (relabel_univariate_normal, "checkpoint_dir", OPTIONAL_PATH),
        (relabel_univariate_normal, "resume_from", OPTIONAL_PATH),
        (run_initialization_passes, "resume_from", OPTIONAL_PATH),
        (run_relabeling_loop, "checkpoint_dir", OPTIONAL_PATH), (run_relabeling_loop, "resume_from", OPTIONAL_PATH),
        (checkpoint.prepare_checkpoint_dir, "checkpoint_dir", PATH),
        (checkpoint.write_checkpoint, "checkpoint_dir", PATH),
        (checkpoint.find_latest_checkpoint, "checkpoint_dir", PATH),
        (checkpoint.load_checkpoint, "resume_from", PATH),
        (PosteriorChain.to_feather, "feather_filepath", PATH), (PosteriorChain.read_feather, "feather_filepath", PATH),
        (load_samples, "path", PATH), (load_samples_product_space, "filepath", PATH),
        (load_samples_fixed_num_sources, "filepath", PATH), (load_samples_ucbmcmc, "chain_folder", PATH),
    ]
    for function, parameter, expected in path_parameters:
        assert typing.get_type_hints(function)[parameter] == expected, (function.__name__, parameter)

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    (hatchling,) = [r for r in map(Requirement, pyproject["build-system"]["requires"]) if r.name == "hatchling"]
    assert str(hatchling.specifier) == str(Requirement("hatchling>=1.27,<2").specifier)

    with monkeypatch.context() as blocked:
        blocked.setitem(sys.modules, "matplotlib", None)
        namespace = runpy.run_path(str(CONFTEST))
    assert "matplotlib" not in namespace and callable(namespace["scripted_costs"])
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib
        matplotlib.use("svg")
        runpy.run_path(str(CONFTEST))
        assert matplotlib.get_backend().lower() == "agg"
