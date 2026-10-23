"""Checkpoints: what a run writes and when, resuming to the uninterrupted result, and refusing another run's files.

A checkpoint holds the best labeling so far with the run state of `petra.checkpoint`; a resume never comes back
worse than it, and a checkpoint resumes only the run that wrote it, from a directory holding only that run.
"""

import json
import logging
import os
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pytest

import petra
from petra import checkpoint as checkpoint_module
from petra import make_catalog
from petra.bayesian_gaussian import bayesian_relabel_loop
from petra.checkpoint import (
    CHECKPOINT_METADATA_KEY,
    CHECKPOINT_TEMPLATE,
    find_latest_checkpoint,
    load_checkpoint,
    run_fingerprint,
    write_checkpoint,
)
from petra.fits import compute_cost_matrix, mv_normal_log_prob
from petra.initialization import relabel_univariate_normal
from petra.make_catalog import make_catalog_mv_normal, relabel_mv_normal
from petra.posterior_chain import FEATHER_METADATA_KEY, PosteriorChain
from petra.relabel import run_relabeling_loop
from petra.utils import find_prob_in_model
from tests import helpers
from tests.helpers import (
    BUDGET_WARNING,
    EMPTY_LABEL_WARNING,
    REF,
    case_a_chain,
    checkpoint_files,
    cut_short,
    empty_slot_chain,
    run_state,
    same_catalog,
    separated_chain,
    stops_on_a_cost_increase,
    three_full_labels_chain,
    widening_chain,
)


def name(iteration: int) -> str:
    """Return the file name of checkpoint `iteration`."""
    return CHECKPOINT_TEMPLATE.format(iteration=iteration)


def state_for(posterior_chain: PosteriorChain, iteration: int, *, eps: float = 1e-6, **changes) -> dict:
    """Return the state a multivariate-normal run of `posterior_chain`, at its own width, writes after `iteration`."""
    fingerprint = run_fingerprint(posterior_chain, method="mv_normal", max_num_sources=posterior_chain.num_sources,
                                  eps=eps)
    return {**fingerprint, "iteration": iteration, "best_iteration": iteration,
            "best_cost": float(posterior_chain.cost_dict.get(posterior_chain.num_sources, 0.0)),
            "stopped": False, "stop_reason": None, **changes}


def snapshot(directory) -> dict:
    """Return every file of `directory` with its bytes."""
    return {entry: (Path(directory) / entry).read_bytes() for entry in checkpoint_files(directory)}


def mv(posterior_chain: PosteriorChain, max_num_sources: int, **kwargs) -> PosteriorChain:
    """Run `relabel_mv_normal` without a progress bar unless asked for one."""
    return relabel_mv_normal(posterior_chain, max_num_sources, **{"progress": False, **kwargs})


def shifted(posterior_chain: PosteriorChain) -> PosteriorChain:
    """Return a chain of the same shape and other data."""
    return PosteriorChain(posterior_chain.get_chain() + 100.0, posterior_chain.num_sources,
                          posterior_chain.num_params_per_source)


# What is written, and when


def test_a_checkpoint_is_written_each_time_the_best_labeling_improves(tmp_path, two_cluster_chain):
    """Files 001 and 002, distinct and each the best so far; the last records 4 iterations and the convergence."""
    result = mv(two_cluster_chain, 2, num_iterations=10, checkpoint_dir=str(tmp_path), split_moves=False)

    assert checkpoint_files(tmp_path) == [name(1), name(2)]
    stored = [load_checkpoint(str(tmp_path / name(k))) for k in (1, 2)]
    assert not np.array_equal(stored[0].posterior_chain.get_chain(), stored[1].posterior_chain.get_chain())
    costs = [checkpoint.posterior_chain.cost_dict[2] for checkpoint in stored]
    assert costs == [checkpoint.state["best_cost"] for checkpoint in stored] and costs[0] > costs[1]
    assert (stored[0].state["iteration"], stored[0].state["stopped"]) == (1, False)

    latest = load_checkpoint(str(tmp_path))
    assert (latest.file_iteration, latest.completed_iterations) == (2, 4)
    assert (latest.state["iteration"], latest.state["best_iteration"]) == (4, 2)
    assert latest.state["stopped"] and latest.state["stop_reason"] == "converged"
    assert result.cost_dict[2] == min(costs)
    assert same_catalog(result, latest.posterior_chain)


def test_the_run_state_records_the_fingerprint_and_progress(tmp_path, two_cluster_chain):
    """The state holds exactly the fingerprint and progress keys, as strict JSON, for the entry point's run."""
    with pytest.warns(UserWarning, match=EMPTY_LABEL_WARNING):
        result = make_catalog_mv_normal(two_cluster_chain, 3, num_iterations=2, init_num_iterations=2,
                                        shuffle_seed=5, eps=1e-4, checkpoint_dir=str(tmp_path), progress=False)
    latest, _ = find_latest_checkpoint(str(tmp_path))
    raw = feather.read_table(latest).schema.metadata[CHECKPOINT_METADATA_KEY]

    def refuse(token):
        raise AssertionError(f"non-strict JSON token {token}")

    state = json.loads(raw.decode("utf-8"), parse_constant=refuse)
    assert list(state) == ["version", "petra_version", "method", "settings", "max_num_sources", "eps", "n_samples",
                           "num_params_per_source", "input_sha256", "validate_nan_convention", "run_info",
                           "iteration", "best_iteration", "best_cost", "stopped", "stop_reason"]
    assert (state["version"], state["petra_version"], state["method"], state["settings"]) == (
        2, petra.__version__, "mv_normal", {})
    assert state["run_info"] == {"entry_point": "make_catalog_mv_normal", "shuffle_seed": 5}
    assert (state["max_num_sources"], state["eps"], state["n_samples"], state["num_params_per_source"]) == (
        3, 1e-4, 30, 1)
    assert len(state["input_sha256"]) == 64 and int(state["input_sha256"], 16) >= 0
    assert state["validate_nan_convention"] is True
    assert state["best_cost"] == result.cost_dict[3]


def test_the_chain_digest_ignores_slot_order_padding_and_signed_zero_but_not_values(monkeypatch):
    """Shuffling, widening, relabeling, -0.0 and the block size keep the digest; a changed or moved value does not."""
    rng = np.random.default_rng(3)
    chain = rng.normal(size=(20, 3, 2))
    chain[rng.random((20, 3)) < 0.3] = np.nan
    chain[0, 0] = [0.0, 1.0]
    chain[1, 2] = np.nan
    base = PosteriorChain(chain, 3, 2)

    def digest(posterior_chain):
        return run_fingerprint(posterior_chain, method="mv_normal", max_num_sources=5, eps=1e-6)["input_sha256"]

    negative_zero = chain.copy()
    negative_zero[0, 0, 0] = -0.0
    one_ulp = chain.copy()
    one_ulp[0, 0, 1] = np.nextafter(1.0, 2.0)
    moved = chain.copy()
    moved[1, 2], moved[0, 0] = chain[0, 0], np.nan

    expected = digest(base)
    assert digest(base.randomize_entries(seed=1)) == expected
    assert digest(base.expand_chain(5)) == expected
    assert digest(mv(base, 3, num_iterations=1, split_moves=False)) == expected
    assert digest(PosteriorChain(negative_zero, 3, 2)) == expected
    monkeypatch.setattr(checkpoint_module, "_DIGEST_BLOCK_ROWS", 4)
    assert digest(base) == expected
    assert digest(PosteriorChain(one_ulp, 3, 2)) != expected
    assert digest(PosteriorChain(moved, 3, 2)) != expected


def several_improvements_chain() -> PosteriorChain:
    """A shuffled 60-sample, three-source chain whose run improves its labeling several times."""
    rng = np.random.default_rng(4)
    chain = rng.normal(size=(60, 3, 2)) + np.array([[0.0, 0.0], [4.0, 4.0], [8.0, 0.0]])
    return PosteriorChain(np.stack([chain[i, rng.permutation(3)] for i in range(60)]), 3, 2)


@pytest.mark.parametrize("keep", [1, 2])
def test_keep_checkpoints_keeps_the_newest_files(tmp_path, monkeypatch, two_cluster_chain, keep):
    """keep_checkpoints leaves the newest files, prunes the run a resume continues, and writes nothing alone (F027)."""
    everything, pruned = tmp_path / "everything", tmp_path / "pruned"
    settings = dict(num_iterations=20, initialization_param_index=None, progress=False)
    full = make_catalog_mv_normal(several_improvements_chain(), 3, checkpoint_dir=str(everything), **settings)
    kept = make_catalog_mv_normal(several_improvements_chain(), 3, checkpoint_dir=str(pruned),
                                  keep_checkpoints=keep, **settings)
    all_names = checkpoint_files(everything)
    assert len(all_names) > keep
    assert checkpoint_files(pruned) == all_names[-keep:]
    assert {entry: run_state(pruned / entry) for entry in all_names[-keep:]} == {
        entry: run_state(everything / entry) for entry in all_names[-keep:]}
    assert same_catalog(kept, full)

    resumed_dir = tmp_path / "resumed"
    with cut_short():
        mv(two_cluster_chain, 2, num_iterations=2, checkpoint_dir=str(resumed_dir))
    resumed = mv(two_cluster_chain, 2, num_iterations=10, resume_from=str(resumed_dir),
                 checkpoint_dir=str(resumed_dir), keep_checkpoints=keep)
    assert checkpoint_files(resumed_dir) == [name(1), name(2)][-keep:]
    assert same_catalog(resumed, mv(two_cluster_chain, 2, num_iterations=10))
    assert run_state(resumed_dir / name(2))["stopped"]

    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    alone = make_catalog_mv_normal(two_cluster_chain, 2, keep_checkpoints=keep, progress=False)
    assert same_catalog(alone, make_catalog_mv_normal(two_cluster_chain, 2, progress=False))
    assert checkpoint_files(working) == []


def test_a_failed_write_leaves_the_previous_checkpoint_and_no_partial_file(tmp_path, monkeypatch, two_cluster_chain):
    """A write killed half way leaves 001 intact and no temporary; temporaries and other names never count (F007)."""
    original = write_checkpoint(two_cluster_chain, tmp_path, 1, state_for(two_cluster_chain, 1))
    before = snapshot(tmp_path)

    class Killed(BaseException):
        """Stands in for a signal arriving mid-write."""

    real_write = checkpoint_module.feather.write_feather

    def killed_mid_write(table, path, *args, **kwargs):
        real_write(table, path, *args, **kwargs)
        with open(path, "r+b") as handle:
            handle.truncate(os.path.getsize(path) // 2)
        raise Killed

    monkeypatch.setattr(checkpoint_module.feather, "write_feather", killed_mid_write)
    replacement = PosteriorChain(two_cluster_chain.get_chain() + 1.0, 2, 1)
    for iteration in (1, 2):
        with pytest.raises(Killed):
            write_checkpoint(replacement, tmp_path, iteration, state_for(two_cluster_chain, iteration))
    monkeypatch.undo()

    assert snapshot(tmp_path) == before
    assert load_checkpoint(original).completed_iterations == 1
    (tmp_path / f".{name(9)}.deadbeef.tmp").write_bytes(b"partial")
    (tmp_path / "notes.txt").write_text("not a checkpoint")
    assert find_latest_checkpoint(tmp_path) == (original, 1)

    only_temporaries = tmp_path / "only_temporaries"
    only_temporaries.mkdir()
    (only_temporaries / f".{name(3)}.cafe.tmp").write_bytes(b"left by a killed write")
    (only_temporaries / "notes.txt").write_text("not a checkpoint")
    assert find_latest_checkpoint(only_temporaries) is None
    with cut_short():
        mv(two_cluster_chain, 2, num_iterations=1, checkpoint_dir=str(only_temporaries))
    assert find_latest_checkpoint(only_temporaries)[1] == 1


# Reading checkpoints


def missing_str(directory: Path, chain: PosteriorChain):
    """A path that does not exist, as a str."""
    target = str(directory / "nothing_here.feather")
    return target, FileNotFoundError, ["No such checkpoint file or directory", repr(target)]


def missing_path(directory: Path, chain: PosteriorChain):
    """A path that does not exist, as a Path."""
    target = directory / "no_such_checkpoint"
    return target, FileNotFoundError, ["No such checkpoint file or directory", repr(str(target))]


def empty_directory(directory: Path, chain: PosteriorChain):
    """A directory holding no checkpoint."""
    return str(directory), FileNotFoundError, ["No relabeling checkpoint found in", repr(str(directory))]


def truncated_newest(directory: Path, chain: PosteriorChain):
    """Two checkpoints, the newest truncated."""
    older = write_checkpoint(chain, directory, 1, state_for(chain, 1))
    newest = write_checkpoint(chain, directory, 2, state_for(chain, 2))
    with open(newest, "r+b") as handle:
        handle.truncate(os.path.getsize(newest) // 2)
    return str(directory), ValueError, ["cannot be read", repr(newest), f"resume_from={older!r}"]


def truncated_only(directory: Path, chain: PosteriorChain):
    """One checkpoint, truncated."""
    path = write_checkpoint(chain, directory, 1, state_for(chain, 1))
    with open(path, "r+b") as handle:
        handle.truncate(10)
    return path, ValueError, ["cannot be read", repr(path), "There is no older checkpoint"]


def plain_feather(directory: Path, chain: PosteriorChain):
    """A chain saved with to_feather under a checkpoint's name: it carries no run state."""
    path = str(directory / name(1))
    chain.to_feather(path)
    return path, ValueError, ["is not a petra checkpoint", repr(path)]


def bad_json(directory: Path, chain: PosteriorChain):
    """A checkpoint whose run state is not JSON."""
    path = write_checkpoint(chain, directory, 1, state_for(chain, 1))
    table = feather.read_table(path)
    feather.write_feather(table.replace_schema_metadata(
        {**table.schema.metadata, CHECKPOINT_METADATA_KEY: b"{not json"}), path)
    return path, ValueError, ["unreadable run metadata", repr(path)]


def version_1(directory: Path, chain: PosteriorChain):
    """A checkpoint of the earlier layout."""
    path = write_checkpoint(chain, directory, 1, state_for(chain, 1, version=1))
    return path, ValueError, ["layout version", repr(path)]


@pytest.mark.parametrize("make_case", [missing_str, missing_path, empty_directory, truncated_newest, truncated_only,
                                       plain_feather, bad_json, version_1], ids=lambda make_case: make_case.__name__)
def test_load_checkpoint_refuses_what_it_cannot_resume_and_names_the_file(tmp_path, two_cluster_chain, make_case):
    """Each refusal raises, says why and names the file as a str; a truncated newest file names an older one."""
    resume_from, error, expected = make_case(tmp_path, two_cluster_chain)
    with pytest.raises(error) as excinfo:
        load_checkpoint(resume_from)
    message = str(excinfo.value)
    assert all(part in message for part in expected), message
    assert "Path(" not in message


def test_load_checkpoint_resolves_directories_and_files(tmp_path, two_cluster_chain):
    """A directory gives its newest file; a named file resumes from its own number; an unnumbered copy from 0."""
    with cut_short():
        mv(two_cluster_chain, 2, num_iterations=2, checkpoint_dir=str(tmp_path / "run"))
    from_dir = load_checkpoint(tmp_path / "run")
    assert (from_dir.path, from_dir.file_iteration) == (str(tmp_path / "run" / name(2)), 2)
    from_file = load_checkpoint(str(tmp_path / "run" / name(1)))
    assert (from_file.file_iteration, from_file.completed_iterations) == (1, 1)

    with cut_short():
        mv(two_cluster_chain, 2, num_iterations=3, resume_from=str(tmp_path / "run" / name(1)),
           checkpoint_dir=str(tmp_path / "continued"))
    assert checkpoint_files(tmp_path / "continued")[0] == name(2)

    copy = shutil.copy(tmp_path / "run" / name(2), tmp_path / "hand_saved.feather")
    unnumbered = load_checkpoint(copy)
    assert unnumbered.file_iteration == 0
    assert unnumbered.completed_iterations == unnumbered.state["iteration"] == 2
    assert same_catalog(unnumbered.posterior_chain, from_dir.posterior_chain)


# Resuming the same run


@pytest.fixture(scope="module")
def case_a_uninterrupted() -> PosteriorChain:
    """The reference's uninterrupted 8-iteration run of case A at width 4."""
    with cut_short():
        return mv(case_a_chain(), 4, num_iterations=8)


@pytest.mark.parametrize("where", ["same-dir", "fresh-path-dir", "file"])
def test_resume_continues_to_the_uninterrupted_result(tmp_path, case_a_uninterrupted, where):
    """Three checkpointed iterations resumed to 8 give the uninterrupted run's chain and cost."""
    directory = tmp_path / "run"
    with cut_short():
        mv(case_a_chain(), 4, num_iterations=3, checkpoint_dir=str(directory))
    assert checkpoint_files(directory) == REF["resume"]["files"]

    resume_from, checkpoint_dir = {
        "same-dir": (str(directory), str(directory)),
        "fresh-path-dir": (directory, tmp_path / "continued"),
        "file": (str(directory / name(1)), None),
    }[where]
    with cut_short():
        resumed = mv(case_a_chain(), 4, num_iterations=8, resume_from=resume_from, checkpoint_dir=checkpoint_dir)

    assert np.array_equal(resumed.get_chain(), case_a_uninterrupted.get_chain(), equal_nan=True)
    assert resumed.cost_dict == case_a_uninterrupted.cost_dict == {4: REF["resume"]["full_cost"]["4"]}


def stopped_run(directory: Path):
    """The cost-increase fixture run to its stop: a resume with budget left has nothing to do."""
    pc = stops_on_a_cost_increase()
    full = mv(pc, 4, num_iterations=60, checkpoint_dir=str(directory))
    stored = load_checkpoint(str(directory))
    assert (stored.file_iteration, stored.completed_iterations, stored.state["stop_reason"]) == (
        2, 3, "cost increased")
    resume = partial(mv, pc, 4, num_iterations=60, resume_from=str(directory), checkpoint_dir=str(directory))
    return resume, full, ["already stopped after 3 iterations because its cost increased"]


def spent_budget(directory: Path):
    """A checkpoint holding 2 iterations and a clipped prob_in_model, resumed with num_iterations=2."""
    original = empty_slot_chain()
    clipped = find_prob_in_model(original.get_chain(), 4, eps=1e-2)
    assert clipped[3] == 1e-2
    on_disk = PosteriorChain(original.get_chain(), 4, 2, trans_dimensional=True, prob_in_model=clipped,
                             cost_dict={4: -7.0})
    write_checkpoint(on_disk, directory, 2, state_for(original, 2, eps=1e-2, best_cost=-7.0))
    resume = partial(mv, original, 4, num_iterations=2, eps=1e-2, resume_from=str(directory))
    return resume, on_disk, ["already holds 2 completed iterations", "Pass num_iterations=3"]


def killed_while_stopping(directory: Path):
    """The cost-increase fixture killed during iteration 3, when the uninterrupted run stops."""
    pc = stops_on_a_cost_increase()
    with cut_short():
        mv(pc, 4, num_iterations=2, checkpoint_dir=str(directory))
    assert not load_checkpoint(str(directory)).state["stopped"]
    resume = partial(mv, pc, 4, num_iterations=60, resume_from=str(directory), checkpoint_dir=str(directory))
    return resume, mv(pc, 4, num_iterations=60), []


@pytest.mark.parametrize("make_case", [stopped_run, spent_budget, killed_while_stopping],
                         ids=lambda make_case: make_case.__name__)
def test_a_stopped_or_spent_run_resumes_unchanged_with_a_warning(tmp_path, make_case):
    """A stopped or spent run comes back as stored, unclipped, writing nothing; one killed while stopping finishes."""
    resume, expected, messages = make_case(tmp_path)
    before = snapshot(tmp_path)
    if messages:
        with pytest.warns(UserWarning) as caught:
            result = resume()
        assert len(caught) == 1 and all(part in str(caught[0].message) for part in messages)
        assert snapshot(tmp_path) == before
    else:
        result = resume()
        state = load_checkpoint(str(tmp_path)).state
        assert (state["iteration"], state["stopped"], state["stop_reason"]) == (3, True, "cost increased")

    assert np.array_equal(result.get_chain(), expected.get_chain(), equal_nan=True)
    assert result.cost_dict == expected.cost_dict
    assert np.array_equal(result.prob_in_model, find_prob_in_model(result.get_chain(), 4, eps=0))


# Refusing another run's checkpoints


def two_parameter_chain() -> PosteriorChain:
    """Two separated sources in two parameters, for the univariate relabeler."""
    rng = np.random.default_rng(1)
    return PosteriorChain(rng.normal(size=(30, 2, 2)) + np.array([[0.0, 0.0], [6.0, 6.0]]), 2, 2)


def mv_writer(width: int = 2):
    """Write one multivariate-normal iteration of the two-cluster chain at `width`."""
    return lambda directory: mv(helpers.two_cluster_chain(), width, num_iterations=1, split_moves=False,
                                checkpoint_dir=directory)


def mv_resume(chain=helpers.two_cluster_chain, width: int = 2, **kwargs):
    """Resume the multivariate-normal run of `chain()` at `width`."""
    return lambda directory: mv(chain(), width, num_iterations=10, resume_from=directory, **kwargs)


RESUME_CASES = {
    "other-data": (mv_writer(), mv_resume(lambda: shifted(helpers.two_cluster_chain())),
                   ["was written for a different input chain"]),
    "29-samples": (mv_writer(), mv_resume(lambda: PosteriorChain(helpers.two_cluster_chain().get_chain()[:29],
                                                                 2, 1)),
                   ["holds 30 samples of 1 parameters per source"]),
    "2-parameters": (mv_writer(), mv_resume(lambda: PosteriorChain(
                         np.repeat(helpers.two_cluster_chain().get_chain(), 2, axis=2), 2, 2)),
                     ["holds 30 samples of 1 parameters per source"]),
    "width-2": (mv_writer(3), mv_resume(width=2), ["3 source slots wide", "max_num_sources=2"]),
    "width-4": (mv_writer(3), mv_resume(width=4), ["3 source slots wide", "max_num_sources=4"]),
    "eps": (mv_writer(), mv_resume(eps=1e-4), ["was written with eps="]),
    "method": (mv_writer(),
               lambda directory: relabel_univariate_normal(helpers.two_cluster_chain(), 2, num_iterations=5,
                                                           resume_from=directory, progress=False),
               ["'mv_normal'", "'univariate_normal'"]),
    "univariate-parameter": (
        lambda directory: relabel_univariate_normal(two_parameter_chain(), 2, num_iterations=1, init_parameter_index=0,
                                                    checkpoint_dir=directory, progress=False),
        lambda directory: relabel_univariate_normal(two_parameter_chain(), 2, num_iterations=5, init_parameter_index=1,
                                                    resume_from=directory, progress=False),
        ["'init_parameter_index': 0", "'init_parameter_index': 1"]),
    "niw-kappa0": (
        lambda directory: bayesian_relabel_loop(helpers.two_cluster_chain(), 2, num_iterations=1,
                                                checkpoint_dir=directory, progress=False),
        lambda directory: bayesian_relabel_loop(helpers.two_cluster_chain(), 2, num_iterations=5, kappa0=0.1,
                                                resume_from=directory, progress=False),
        ["'kappa0': 0.01", "'kappa0': 0.1"]),
    "niw-nu0-5-and-5.0": (
        lambda directory: bayesian_relabel_loop(helpers.two_cluster_chain(), 2, num_iterations=1, nu0=5,
                                                checkpoint_dir=directory, progress=False),
        lambda directory: bayesian_relabel_loop(helpers.two_cluster_chain(), 2, num_iterations=5, nu0=5.0,
                                                resume_from=directory, progress=False),
        None),
    "progress-split-moves-run-info": (
        mv_writer(), mv_resume(progress=True, split_moves=True, run_info={"entry_point": "another"}), None),
}


@pytest.mark.parametrize("case", list(RESUME_CASES))
def test_a_resume_must_match_the_run_that_wrote_it(tmp_path, case):
    """Other data, shape, width, eps, method or settings are refused, naming the file; the rest is not compared."""
    write, resume, expected = RESUME_CASES[case]
    directory = str(tmp_path / "run")
    write(directory)
    path, _ = find_latest_checkpoint(directory)
    if expected is None:
        assert resume(directory).shape[0] == 30
        return
    with pytest.raises(ValueError) as excinfo:
        resume(directory)
    message = str(excinfo.value)
    assert repr(path) in message and all(part in message for part in expected), message


def fresh_run_into_a_used_directory(tmp_path: Path):
    """A second run, on other data, into a directory holding the first run's checkpoints."""
    directory = tmp_path / "run"
    make_catalog_mv_normal(helpers.two_cluster_chain(), 2, num_iterations=10, init_num_iterations=2,
                           checkpoint_dir=str(directory), progress=False)
    run = partial(make_catalog_mv_normal, shifted(helpers.two_cluster_chain()), 2, num_iterations=1,
                  checkpoint_dir=str(directory), progress=False)
    return run, directory, ["already holds checkpoints", f"resume_from={str(directory)!r}"]


def resume_an_older_file_into_its_directory(tmp_path: Path):
    """Resuming file 001 into its own directory, which holds 002."""
    directory = tmp_path / "run"
    with cut_short():
        mv(helpers.two_cluster_chain(), 2, num_iterations=2, checkpoint_dir=str(directory))
    run = partial(mv, helpers.two_cluster_chain(), 2, num_iterations=5, resume_from=str(directory / name(1)),
                  checkpoint_dir=str(directory))
    return run, directory, [name(2)]


def resume_into_a_directory_holding_the_same_number(tmp_path: Path):
    """Resuming a run whose newest file is 002 into another directory that holds a 002."""
    source, target = tmp_path / "source", tmp_path / "target"
    with cut_short():
        mv(helpers.two_cluster_chain(), 2, num_iterations=2, checkpoint_dir=str(source))
    target.mkdir()
    write_checkpoint(helpers.two_cluster_chain(), target, 2, state_for(helpers.two_cluster_chain(), 2))
    run = partial(mv, helpers.two_cluster_chain(), 2, num_iterations=5, resume_from=str(source),
                  checkpoint_dir=str(target))
    return run, target, [name(2)]


def resume_into_another_runs_directory_with_keep(tmp_path: Path):
    """Resuming run a into run b's directory with keep_checkpoints=1, which would prune b's file."""
    run_a, run_b = tmp_path / "a", tmp_path / "b"
    with cut_short():
        mv(shifted(helpers.two_cluster_chain()), 2, num_iterations=1, checkpoint_dir=str(run_b))
    with cut_short():
        mv(helpers.two_cluster_chain(), 2, num_iterations=2, checkpoint_dir=str(run_a))
    assert checkpoint_files(run_b) == [name(1)]
    run = partial(mv, helpers.two_cluster_chain(), 2, num_iterations=10, resume_from=str(run_a),
                  checkpoint_dir=str(run_b), keep_checkpoints=1)
    return run, run_b, [name(1)]


@pytest.mark.parametrize("make_case", [fresh_run_into_a_used_directory, resume_an_older_file_into_its_directory,
                                       resume_into_a_directory_holding_the_same_number,
                                       resume_into_another_runs_directory_with_keep],
                         ids=lambda make_case: make_case.__name__)
def test_one_directory_holds_one_run(tmp_path, make_case):
    """A directory holding another run's checkpoints is refused before anything is written or deleted (F004)."""
    run, directory, expected = make_case(tmp_path)
    before = snapshot(directory)
    with pytest.raises(ValueError) as excinfo:
        run()
    message = str(excinfo.value)
    assert all(part in message for part in expected), message
    assert snapshot(directory) == before


def read_only_directory(tmp_path: Path) -> Path:
    """A directory the current user cannot write to."""
    target = tmp_path / "read_only"
    target.mkdir()
    target.chmod(0o500)
    return target


@pytest.mark.parametrize("make_target, error", [
    (lambda tmp_path: tmp_path / "a_file", OSError),
    (lambda tmp_path: tmp_path / "a_file" / "below_it", OSError),
    pytest.param(read_only_directory, PermissionError,
                 marks=pytest.mark.skipif(getattr(os, "geteuid", lambda: 0)() == 0,
                                          reason="root writes to a read-only directory")),
], ids=["a-file", "under-a-file", "read-only"])
def test_a_bad_checkpoint_dir_is_refused_before_any_work(tmp_path, monkeypatch, two_cluster_chain, make_target,
                                                         error):
    """The directory is created and probed before any fit or initialization pass runs (F052)."""
    (tmp_path / "a_file").write_text("not a directory")
    target = make_target(tmp_path)
    work = []
    monkeypatch.setattr(make_catalog, "relabel_univariate_normal", lambda *args, **kwargs: work.append("initialize"))

    def fit(chain, max_num_sources):
        work.append("fit")

    try:
        with pytest.raises(error):
            run_relabeling_loop(two_cluster_chain, fit, partial(compute_cost_matrix, log_prob=mv_normal_log_prob), 2,
                                checkpoint_dir=str(target), progress=False, method="mv_normal")
        with pytest.raises(error):
            make_catalog_mv_normal(two_cluster_chain, 2, checkpoint_dir=str(target), progress=False)
    finally:
        if target.is_dir():
            target.chmod(0o700)
    assert work == []


# Resuming never comes back worse


@pytest.mark.parametrize("checkpoint_cost, costs, expected_offset, expected_cost", [
    (1.0, [2.0, 3.0, 4.0], 100.0, 1.0),
    (1.0, [0.5, 0.75], 1.0, 0.5),
    (None, [3.0, 2.0, 1.0], 3.0, 1.0),
], ids=["every-new-iteration-worse", "a-better-one", "fresh-run"])
def test_a_resume_never_comes_back_worse_than_its_checkpoint(tmp_path, scripted_costs, checkpoint_cost, costs,
                                                             expected_offset, expected_cost):
    """A checkpoint's cost is the one to beat, and only a resume has one: a fresh run's input cost 0.0 is not."""
    calls = scripted_costs(costs)
    if checkpoint_cost is None:
        result = mv(PosteriorChain(separated_chain(), 2, 1, cost_dict={2: 0.0}), 2, num_iterations=3,
                    split_moves=False)
    else:
        stored = PosteriorChain(separated_chain(offset=100.0), 2, 1, cost_dict={2: checkpoint_cost})
        write_checkpoint(stored, tmp_path, 2, state_for(PosteriorChain(separated_chain(), 2, 1), 2,
                                                        best_cost=checkpoint_cost))
        result = mv(PosteriorChain(separated_chain(), 2, 1), 2, num_iterations=5, resume_from=str(tmp_path),
                    split_moves=False)

    assert calls
    assert result.cost_dict == {2: expected_cost}
    assert np.array_equal(result.get_chain(), separated_chain(offset=expected_offset))


def test_an_opted_out_run_resumes_without_passing_the_flag_again(tmp_path, caplog):
    """The stored validate_nan_convention reads the checkpoint back; a short stored prob_in_model is rebuilt (F016)."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(40, 2, 2)) + np.array([[0.0, 0.0], [5.0, 5.0]])
    chain[3, 1, 1] = np.nan
    opted_out = PosteriorChain(chain, 2, 2, validate_nan_convention=False)
    mv(opted_out, 2, num_iterations=1, split_moves=False, checkpoint_dir=str(tmp_path / "opted_out"))
    assert load_checkpoint(str(tmp_path / "opted_out")).posterior_chain.validate_nan_convention is False

    with_default_flag = PosteriorChain(chain, 2, 2, validate_nan_convention=False)
    with_default_flag.validate_nan_convention = True
    resumed = mv(with_default_flag, 2, num_iterations=5, split_moves=False,
                 resume_from=str(tmp_path / "opted_out"))
    assert resumed.validate_nan_convention is False

    narrow = np.full((30, 3, 1), np.nan)
    narrow[:, :2, :] = rng.normal(scale=0.2, size=(30, 2, 1)) + np.array([[0.0], [6.0]])
    stored = PosteriorChain(narrow, 3, 1, trans_dimensional=True, prob_in_model=find_prob_in_model(narrow, 3, eps=0),
                            cost_dict={3: -7.0})
    path = write_checkpoint(stored, tmp_path / "short", 2, state_for(stored, 2))
    table = feather.read_table(path)
    payload = json.loads(table.schema.metadata[FEATHER_METADATA_KEY].decode("utf-8"))
    payload["prob_in_model"] = payload["prob_in_model"][:2]
    feather.write_feather(table.replace_schema_metadata(
        {**table.schema.metadata, FEATHER_METADATA_KEY: json.dumps(payload).encode("utf-8")}), path)

    with caplog.at_level(logging.WARNING, logger="petra.posterior_chain"):
        repaired = load_checkpoint(path).posterior_chain
    assert [record.name for record in caplog.records if os.path.basename(path) in record.getMessage()] == [
        "petra.posterior_chain"]
    assert np.array_equal(repaired.prob_in_model, find_prob_in_model(narrow, 3, eps=0))

    resumed = mv(stored, 3, num_iterations=4, resume_from=str(tmp_path / "short"),
                 checkpoint_dir=str(tmp_path / "continued"))
    assert checkpoint_files(tmp_path / "continued") == [name(2)]
    assert load_checkpoint(str(tmp_path / "continued")).completed_iterations > 2
    assert np.array_equal(resumed.prob_in_model, find_prob_in_model(resumed.get_chain(), 3, eps=0))
    assert resumed.prob_in_model[2] == 0.0
    assert resumed.cost_dict[3] <= -7.0


# Split moves


def widening_runs(tmp_path: Path):
    """Case A, 300 samples, no warm start: accepted splits at 14 and 17, and a resume from every earlier file."""
    pc = PosteriorChain(widening_chain(300)[0], 2, 2, True)
    full = make_catalog_mv_normal(pc, 4, initialization_param_index=None, checkpoint_dir=str(tmp_path / "full"),
                                  progress=False)
    names = checkpoint_files(tmp_path / "full")
    states = [run_state(tmp_path / "full" / entry) for entry in names]
    numbers = [int(entry.split("_")[-1].split(".")[0]) for entry in names]
    assert numbers == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 17]
    assert [state["best_iteration"] for state in states] == numbers
    assert [(state["iteration"], state["stopped"]) for state in states[-3:]] == [(11, False), (14, False), (18, True)]
    assert states[-1]["best_cost"] == full.cost_dict[4]

    resumed = []
    for entry in names[:-1]:
        killed = tmp_path / f"killed_{entry}"
        killed.mkdir()
        shutil.copy(tmp_path / "full" / entry, killed / entry)
        resumed.append(make_catalog_mv_normal(pc, 4, initialization_param_index=None, resume_from=str(killed),
                                              progress=False))
    return resumed, full


def budget_runs(short: int, longer: int, empty_label_warnings: int):
    """Case A, 300 samples, no warm start, cut short at `short` and resumed to `longer`, against a fresh `longer`."""
    def run(tmp_path: Path):
        pc = PosteriorChain(widening_chain(300)[0], 2, 2, True)
        settings = dict(initialization_param_index=None, progress=False)
        with pytest.warns(UserWarning, match=EMPTY_LABEL_WARNING), cut_short():
            make_catalog_mv_normal(pc, 4, num_iterations=short, checkpoint_dir=str(tmp_path), **settings)
        state = run_state(find_latest_checkpoint(str(tmp_path))[0])
        assert (state["iteration"], state["best_iteration"], state["stopped"]) == (short, short, False)

        catalogs = []
        for resume_from in (str(tmp_path), None):
            with pytest.warns(UserWarning) as caught:
                catalogs.append(make_catalog_mv_normal(pc, 4, num_iterations=longer, resume_from=resume_from,
                                                       **settings))
            assert [BUDGET_WARNING in str(warning.message) for warning in caught] == (
                [True] + [False] * empty_label_warnings)
        return catalogs[:1], catalogs[1]
    return run


def three_full_labels_runs(tmp_path: Path):
    """Three full labels, split_exclusive: a budget of 6 ends inside the last evaluation; resumed to 20."""
    pc = PosteriorChain(three_full_labels_chain(), 3, 2)
    with cut_short():
        mv(pc, 3, num_iterations=6, checkpoint_dir=str(tmp_path), split_exclusive=True)
    state = load_checkpoint(str(tmp_path)).state
    assert (state["iteration"], state["stopped"], state["stop_reason"]) == (1, False, None)

    resumed = mv(pc, 3, num_iterations=20, resume_from=str(tmp_path), checkpoint_dir=str(tmp_path),
                 split_exclusive=True)
    state = load_checkpoint(str(tmp_path)).state
    assert (state["iteration"], state["stopped"], state["stop_reason"]) == (8, True, "converged")
    return [resumed], mv(pc, 3, num_iterations=20, split_exclusive=True)


@pytest.mark.parametrize("make_case", [widening_runs, budget_runs(13, 16, 0), budget_runs(14, 17, 0),
                                       budget_runs(3, 12, 1), three_full_labels_runs],
                         ids=["any-checkpoint", "13-to-16", "14-to-17", "3-to-12", "three-full-labels"])
def test_split_move_runs_resume_like_uninterrupted_runs(tmp_path, make_case):
    """An accepted split is checkpointed where it was found, and every resume ends where an uninterrupted run does."""
    resumed, uninterrupted = make_case(tmp_path)
    assert resumed and all(same_catalog(catalog, uninterrupted) for catalog in resumed)
