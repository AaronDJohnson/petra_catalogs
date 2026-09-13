"""
Checkpoint and resume, the feature that makes a long relabeling restartable.

A run killed at a walltime limit has to be able to pick up where it stopped, so
these tests check the whole loop: that checkpoints are written, that the latest
one is found, that resuming from either a file or a directory continues from the
right iteration, and that the failure modes raise instead of silently starting
over from scratch.

Two properties are worth naming, because they are the ones that used to be
broken.  **A checkpoint holds what the run would have returned had it stopped
there** -- the labeling, its unclipped ``prob_in_model`` and its cost.  And
**resuming can never come back worse than the checkpoint it resumed from**: the
cost stored with the chain is the cost the resumed run has to beat, so a job
killed and restarted cannot lose the labeling it already had on disk.
"""

import json
import logging
import os

import numpy as np
import pyarrow.feather as feather
import pytest

from petra import relabel as relabel_module
from petra.aux_distributions import mv_normal_aux_distribution
from petra.make_catalog import relabel_mv_normal
from petra.parametric_fits import mv_normal_fit
from petra.posterior_chain import FEATHER_METADATA_KEY, PosteriorChain
from petra.relabel import (
    CHECKPOINT_TEMPLATE,
    create_relabel_samples,
    find_latest_checkpoint,
    load_checkpoint,
)
from petra.utils import find_prob_in_model


def separated_chain(offset: float = 0.0) -> np.ndarray:
    """Two well-separated one-dimensional clusters, shifted by `offset`."""
    rng = np.random.default_rng(0)
    return rng.normal(scale=0.2, size=(30, 2, 1)) + np.array([[0.0], [8.0]]) + offset


def write_checkpoint(directory, posterior_chain, iteration) -> str:
    """Put `posterior_chain` where a resume will find it as iteration `iteration`."""
    filepath = os.path.join(str(directory), CHECKPOINT_TEMPLATE.format(iteration=iteration))
    posterior_chain.to_feather(filepath)
    return filepath


@pytest.fixture
def two_cluster_chain():
    """A 30-sample, two-source chain whose clusters are 8 units apart."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(30, 2, 1)) + np.array([[0.0], [8.0]])
    shuffled = np.stack([chain[i, rng.permutation(2), :] for i in range(30)])
    return PosteriorChain(shuffled, 2, 1, trans_dimensional=True)


@pytest.fixture
def scripted_costs(monkeypatch):
    """
    Replace one relabeling step with a stub that follows a scripted cost sequence.

    Returns a callable taking the list of costs to hand out, one per iteration,
    and returning the list of chains the stub produced, so a test can assert
    *which* labeling came back rather than hoping a random chain happens to
    produce the cost ordering it needs.
    """
    def install(costs):
        produced = []

        def fake_one_iteration(posterior_chain, aux_parameters, prob_in_model,
                               max_num_sources, compute_cost_matrix, progress=True):
            index = len(produced)
            # A distinguishable, non-degenerate chain per iteration, so that the
            # parametric fit the loop runs on it succeeds and the test can tell
            # every iteration apart from every other and from the checkpoint.
            new = PosteriorChain(separated_chain(offset=float(index + 1)), 2, 1,
                                 trans_dimensional=True, prob_in_model=prob_in_model,
                                 cost_dict={max_num_sources: costs[index]})
            produced.append(new)
            return new

        monkeypatch.setattr(relabel_module, "relabel_posterior_chain_one_iteration",
                            fake_one_iteration)
        return produced

    return install


def test_relabeling_writes_one_checkpoint_per_iteration(tmp_path, two_cluster_chain):
    result = relabel_mv_normal(two_cluster_chain, max_num_sources=2, num_iterations=3,
                               checkpoint_dir=str(tmp_path), progress=False)

    written = sorted(os.listdir(tmp_path))
    assert written, "no checkpoint was written"
    assert written[0] == CHECKPOINT_TEMPLATE.format(iteration=1)

    latest_path, completed = find_latest_checkpoint(str(tmp_path))
    assert completed == len(written)
    assert os.path.basename(latest_path) == CHECKPOINT_TEMPLATE.format(iteration=completed)

    restored, restored_completed = load_checkpoint(latest_path)
    assert restored_completed == completed
    assert restored.chain.shape == result.chain.shape


def test_resume_from_a_directory_continues_the_run(tmp_path, two_cluster_chain):
    """
    "Continues" means it picks up at iteration 3, not that it returns the right shape.

    The iteration numbers of the checkpoints the *resumed* run writes are what
    say so: a resume that quietly restarted from scratch would come back with a
    chain of the same shape and the same probabilities, and would write
    ``iteration_001`` again.  The cost is the second half of the claim -- a
    resumed run may not come back worse than the checkpoint it read.
    """
    relabel_mv_normal(two_cluster_chain, max_num_sources=2, num_iterations=2,
                      checkpoint_dir=str(tmp_path), progress=False)
    latest_path, completed = find_latest_checkpoint(str(tmp_path))
    stored, _ = load_checkpoint(latest_path)

    continued = tmp_path / "continued"
    resumed = relabel_mv_normal(two_cluster_chain, max_num_sources=2,
                                num_iterations=completed + 2,
                                resume_from=str(tmp_path),
                                checkpoint_dir=str(continued), progress=False)
    assert resumed.chain.shape == two_cluster_chain.chain.shape

    # The loop may stop early, so only the *lowest* number written is pinned.
    written = sorted(path.name for path in continued.iterdir())
    assert written, "the resumed run did no work"
    assert written[0] == CHECKPOINT_TEMPLATE.format(iteration=completed + 1)
    assert resumed.cost_dict[2] <= stored.cost_dict[2]


def test_resume_from_a_single_file(tmp_path, two_cluster_chain):
    """
    Naming a file resumes from *that* file's iteration, not from the latest one.

    Both checkpoints are on disk and the second is the one a directory resume
    would pick, so the iteration the resumed run numbers its own checkpoints
    from is what distinguishes the two spellings.
    """
    relabel_mv_normal(two_cluster_chain, max_num_sources=2, num_iterations=2,
                      checkpoint_dir=str(tmp_path), progress=False)
    checkpoint_path = os.path.join(tmp_path, CHECKPOINT_TEMPLATE.format(iteration=1))

    continued = tmp_path / "continued"
    resumed = relabel_mv_normal(two_cluster_chain, max_num_sources=2, num_iterations=3,
                                resume_from=checkpoint_path,
                                checkpoint_dir=str(continued), progress=False)
    assert resumed.chain.shape == two_cluster_chain.chain.shape

    written = sorted(path.name for path in continued.iterdir())
    assert written, "the resumed run did no work"
    assert written[0] == CHECKPOINT_TEMPLATE.format(iteration=2)


def test_resume_past_the_requested_iterations_is_a_no_op(tmp_path, two_cluster_chain):
    relabel_mv_normal(two_cluster_chain, max_num_sources=2, num_iterations=3,
                      checkpoint_dir=str(tmp_path), progress=False)
    latest_path, completed = find_latest_checkpoint(str(tmp_path))
    stored, _ = load_checkpoint(latest_path)

    resumed = relabel_mv_normal(two_cluster_chain, max_num_sources=2,
                                num_iterations=completed, resume_from=str(tmp_path),
                                progress=False)
    assert np.array_equal(resumed.chain, stored.chain, equal_nan=True)


def test_load_checkpoint_reports_a_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such checkpoint"):
        load_checkpoint(os.path.join(tmp_path, "nothing_here.feather"))


def test_load_checkpoint_reports_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No relabeling checkpoint found"):
        load_checkpoint(str(empty))


def test_find_latest_checkpoint_ignores_unrelated_files(tmp_path):
    (tmp_path / "notes.txt").write_text("not a checkpoint")
    assert find_latest_checkpoint(str(tmp_path)) is None


def test_load_checkpoint_of_an_unnumbered_file_reports_zero_iterations(
        tmp_path, two_cluster_chain):
    path = os.path.join(tmp_path, "hand_saved.feather")
    two_cluster_chain.to_feather(path)
    restored, completed = load_checkpoint(path)
    assert completed == 0
    assert np.array_equal(restored.chain, two_cluster_chain.chain, equal_nan=True)


def test_resume_returns_the_checkpoint_when_every_new_iteration_is_worse(
        tmp_path, scripted_costs):
    """
    Resuming must never come back worse than the checkpoint it resumed from.

    ``best_cost_of_assignment`` used to start at ``None`` on the resume path, so
    the first post-resume iteration became "best" unconditionally -- a job killed
    and restarted could return a labeling more expensive than the one already on
    disk.  The costs here are scripted so the ordering is deliberate: the
    checkpoint at -500 beats every iteration this run can produce.
    """
    checkpoint_chain = PosteriorChain(separated_chain(offset=100.0), 2, 1,
                                      trans_dimensional=True, cost_dict={2: -500.0})
    write_checkpoint(tmp_path, checkpoint_chain, 2)
    produced = scripted_costs([-1.0, -2.0, -3.0])

    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
    result = relabel(PosteriorChain(separated_chain(), 2, 1, trans_dimensional=True),
                     max_num_sources=2, num_iterations=5, resume_from=str(tmp_path),
                     progress=False)

    assert produced, "the resumed run never ran an iteration to be tempted by"
    assert result.cost_dict[2] == -500.0
    assert np.array_equal(result.get_chain(), checkpoint_chain.get_chain())
    for index, candidate in enumerate(produced):
        assert not np.array_equal(result.get_chain(), candidate.get_chain()), (
            f"the resumed run returned iteration {index}, which cost "
            f"{candidate.cost_dict[2]} against the checkpoint's -500.0"
        )


def test_resume_takes_the_better_new_iteration_when_there_is_one(tmp_path, scripted_costs):
    """The seeded cost is a floor to beat, not a lock on the checkpoint."""
    checkpoint_chain = PosteriorChain(separated_chain(offset=100.0), 2, 1,
                                      trans_dimensional=True, cost_dict={2: -1.0})
    write_checkpoint(tmp_path, checkpoint_chain, 2)
    produced = scripted_costs([-500.0, -400.0])

    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
    result = relabel(PosteriorChain(separated_chain(), 2, 1, trans_dimensional=True),
                     max_num_sources=2, num_iterations=4, resume_from=str(tmp_path),
                     progress=False)

    assert result.cost_dict[2] == -500.0
    assert np.array_equal(result.get_chain(), produced[0].get_chain())


def test_a_fresh_run_does_not_treat_its_input_chains_cost_as_one_to_beat(scripted_costs):
    """
    Only a resume has a previous cost; a fresh run's input does not supply one.

    ``make_catalog_*`` runs an initialization relabeler first and hands its
    result on, so the chain a fresh run starts from often carries a
    ``cost_dict`` entry under this very key -- computed from a different fit,
    and so not comparable with anything this loop produces.  Seeding from it
    would make the relabeler hand its input straight back.
    """
    produced = scripted_costs([-1.0, -2.0, -3.0])
    from_initialization = PosteriorChain(separated_chain(), 2, 1, trans_dimensional=True,
                                         cost_dict={2: -500.0})

    relabel = create_relabel_samples(mv_normal_fit, mv_normal_aux_distribution)
    result = relabel(from_initialization, max_num_sources=2, num_iterations=3,
                     progress=False)

    assert result.cost_dict[2] == -3.0
    assert np.array_equal(result.get_chain(), produced[-1].get_chain())


def test_a_resume_with_nothing_left_to_do_still_describes_its_own_chain(tmp_path):
    """
    The do-nothing path hands back a chain straight off disk, and owes the contract.

    A checkpoint may carry a ``prob_in_model`` that does not describe it -- the
    clipped, pre-relabel array of the iteration that wrote it, or nothing at all.
    The relabeler has to normalize it before returning, or a caller gets a chain
    that cannot describe itself.
    """
    rng = np.random.default_rng(2)
    chain = np.full((40, 3, 1), np.nan)
    chain[:, :2, :] = rng.normal(scale=0.2, size=(40, 2, 1)) + np.array([[0.0], [6.0]])
    stored = PosteriorChain(chain, 3, 1, trans_dimensional=True, prob_in_model=None,
                            cost_dict={3: -7.0})
    write_checkpoint(tmp_path, stored, 2)

    result = relabel_mv_normal(stored, max_num_sources=3, num_iterations=2,
                               resume_from=str(tmp_path), progress=False)

    assert result.prob_in_model is not None
    assert np.array_equal(result.prob_in_model,
                          find_prob_in_model(result.get_chain(), 3, eps=0))
    # The third slot is occupied in no sample: its probability is 0, not eps.
    assert result.prob_in_model[2] == 0.0
    # ... and the labeling itself came back untouched, since there was nothing to do.
    assert np.array_equal(result.get_chain(), stored.get_chain(), equal_nan=True)


def test_a_checkpoint_whose_prob_in_model_was_never_widened_can_still_be_resumed(
        tmp_path, caplog):
    """
    A multi-day run on disk must not become unresumable over its metadata.

    ``expand_chain`` used to widen the chain without widening ``prob_in_model``,
    and ``to_feather`` wrote that pair out faithfully, so the files the bug
    produced carry a chain of `n` slots beside an array of fewer.  Requiring the
    two to agree at construction made ``resume_from`` hard-fail on exactly those
    files -- the samples intact, the run lost.  The array is a function of the
    samples, so the reader recovers it, says so, and the resume proceeds.

    The checkpoint is edited after the fact because the constructor now refuses
    to build such a chain at all; that is the point, and it is why the file has
    to be met where it is.
    """
    rng = np.random.default_rng(0)
    chain = np.full((30, 3, 1), np.nan)
    chain[:, :2, :] = rng.normal(scale=0.2, size=(30, 2, 1)) + np.array([[0.0], [6.0]])
    stored = PosteriorChain(chain, 3, 1, trans_dimensional=True,
                            prob_in_model=find_prob_in_model(chain, 3, eps=0),
                            cost_dict={3: -7.0})
    path = write_checkpoint(tmp_path, stored, 2)

    table = feather.read_table(path)
    metadata = dict(table.schema.metadata)
    payload = json.loads(metadata[FEATHER_METADATA_KEY].decode("utf-8"))
    payload["prob_in_model"] = payload["prob_in_model"][:2]   # what the old widening left
    metadata[FEATHER_METADATA_KEY] = json.dumps(payload).encode("utf-8")
    feather.write_feather(table.replace_schema_metadata(metadata), path)

    continued = tmp_path / "continued"
    with caplog.at_level(logging.WARNING, logger="petra.posterior_chain"):
        resumed = relabel_mv_normal(stored, max_num_sources=3, num_iterations=4,
                                    resume_from=str(tmp_path),
                                    checkpoint_dir=str(continued), progress=False)

    # It resumed rather than restarted, and it resumed from the file's own chain.
    written = sorted(path.name for path in continued.iterdir())
    assert written and written[0] == CHECKPOINT_TEMPLATE.format(iteration=3)
    assert os.path.basename(path) in caplog.text
    # The recovered chain describes itself, as every chain petra hands back does.
    assert np.array_equal(resumed.prob_in_model,
                          find_prob_in_model(resumed.get_chain(), 3, eps=0))
    assert resumed.prob_in_model[2] == 0.0
    # The resume floor survived the repair: only prob_in_model was touched.
    assert resumed.cost_dict[3] <= -7.0
