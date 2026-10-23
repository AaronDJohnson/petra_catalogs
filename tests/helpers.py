"""Shared chains, constants and reference values for the test suite.

The chain builders keep the bodies of the old suite's helpers; the ``ref_*`` builders and ``REF`` reproduce the
behaviour reference the restructure was checked against, so a ref-pinned test must build its chain with them.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import pytest

from petra.checkpoint import CHECKPOINT_METADATA_KEY
from petra.make_catalog import relabel_mv_normal
from petra.posterior_chain import PosteriorChain

#: Larger than any chain here, so every slot falls back to the uniform prior and no flow is trained.
NO_FLOWS = 10 ** 6

#: The warning `make_catalog_mv_normal` gives when a slot it added ends empty.
EMPTY_LABEL_WARNING = "widened the chain"

#: The warning `run_relabeling_loop` gives when the budget ends a search with split moves.
BUDGET_WARNING = "ran out before the search with split moves stopped"

LISA_EXAMPLE = Path(__file__).resolve().parents[1] / "example_notebooks" / "data" / "lisa_example.feather"


def budget_message(num_iterations: int) -> str:
    """Return the whole text of the budget warning for a budget of `num_iterations`."""
    return (f"The budget of num_iterations={num_iterations} ran out before the search with split moves stopped, "
            f"so the search did not finish, and the labeling returned is the best found so far. Each split "
            f"proposal can take up to 10 iterations. Pass a larger num_iterations to finish the search -- a run "
            f"with a checkpoint_dir can be resumed with one -- or split_moves=False for the plain search.")


# Checkpointing


def separated_chain(offset: float = 0.0) -> np.ndarray:
    """Two well-separated one-dimensional clusters, shifted by `offset`."""
    rng = np.random.default_rng(0)
    return rng.normal(scale=0.2, size=(30, 2, 1)) + np.array([[0.0], [8.0]]) + offset


def two_cluster_chain() -> PosteriorChain:
    """A shuffled 30-sample, two-source chain 3 units apart: a run improves on iterations 1 and 2, stops after 4."""
    rng = np.random.default_rng(0)
    chain = rng.normal(size=(30, 2, 1)) + np.array([[0.0], [3.0]])
    shuffled = np.stack([chain[i, rng.permutation(2), :] for i in range(30)])
    return PosteriorChain(shuffled, 2, 1, trans_dimensional=True)


def stops_on_a_cost_increase() -> PosteriorChain:
    """A 30-sample, 4-slot, 3-parameter chain whose run improves twice, then stops on an increase."""
    rng = np.random.default_rng(25)
    chain = rng.normal(size=(4, 3))[None] + rng.normal(size=(30, 4, 3))
    chain[rng.random((30, 4)) < 0.5] = np.nan
    chain = np.stack([chain[i, rng.permutation(4)] for i in range(30)])
    return PosteriorChain(chain, 4, 3, trans_dimensional=True)


def run_state(path: str | os.PathLike[str]) -> dict:
    """Return the run fingerprint and progress a checkpoint carries."""
    raw = feather.read_table(str(path)).schema.metadata[CHECKPOINT_METADATA_KEY]
    return json.loads(raw.decode("utf-8"))


def checkpoint_files(directory: str | os.PathLike[str]) -> list:
    """Return every file in `directory`, temporary ones included, sorted."""
    return sorted(os.listdir(str(directory)))


def cut_short() -> pytest.WarningsRecorder:
    """Expect the budget warning of a multivariate-normal run that a test cuts short on purpose."""
    return pytest.warns(UserWarning, match=BUDGET_WARNING)


# The relabeling loop


def dense_chain() -> PosteriorChain:
    """Three separated sources, every one present in every sample: truth is exactly 1.0."""
    rng = np.random.default_rng(0)
    chain = rng.normal(scale=0.3, size=(40, 3, 2)) + np.array([[0.0, 0.0],
                                                              [6.0, 0.0],
                                                              [0.0, 6.0]])
    return PosteriorChain(chain, 3, 2, trans_dimensional=True)


def empty_slot_chain() -> PosteriorChain:
    """Three separated sources in four slots: the spare slot's truth is exactly 0.0."""
    rng = np.random.default_rng(1)
    chain = np.full((50, 4, 2), np.nan)
    chain[:, :3, :] = rng.normal(scale=0.2, size=(50, 3, 2)) + np.array([[0.0, 0.0],
                                                                        [8.0, 0.0],
                                                                        [0.0, 8.0]])
    return PosteriorChain(chain, 4, 2, trans_dimensional=True)


def trans_dimensional_chain() -> PosteriorChain:
    """Overlapping sources with scattered NaN rows: every probability is inside (0, 1)."""
    rng = np.random.default_rng(119)
    chain = rng.normal(size=(60, 4, 2)) * rng.uniform(0.2, 3.0)
    chain[rng.random((60, 4)) < 0.5] = np.nan
    return PosteriorChain(chain, 4, 2, trans_dimensional=True)


# Split moves


def label_purity(catalog: np.ndarray, original: np.ndarray, true_ids: np.ndarray) -> float:
    """Return the fraction of present rows that sit in a label whose majority source they belong to."""
    n_samples, n_slots, _ = catalog.shape
    catalog_ids = np.full((n_samples, n_slots), -1)
    for i in range(n_samples):
        for j in np.flatnonzero(~np.isnan(catalog[i]).any(axis=-1)):
            match = np.flatnonzero(np.all(original[i] == catalog[i, j], axis=-1))[0]
            catalog_ids[i, j] = true_ids[i, match]
    pure = present = 0
    for j in range(n_slots):
        ids = catalog_ids[catalog_ids[:, j] >= 0, j]
        if ids.size:
            pure += np.bincount(ids).max()
            present += ids.size
    return pure / present


def packed_chain(rng: np.random.Generator, centres: np.ndarray, presence: np.ndarray, n_samples: int,
                 width: int, std: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Draw a packed chain and the true source of every row (-1 where absent), as the F003 reproducers do."""
    n_sources, n_params = centres.shape
    chain = np.full((n_samples, width, n_params), np.nan)
    true_ids = np.full((n_samples, width), -1)
    for i in range(n_samples):
        present = np.flatnonzero(rng.random(n_sources) < presence)
        rng.shuffle(present)
        present = present[:width]
        chain[i, :present.size] = centres[present] + std * rng.normal(size=(present.size, n_params))
        true_ids[i, :present.size] = present
    return chain, true_ids


def widening_chain(n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """F003 case A: four sources at 0, 10, 20, 30 in two parameters, each present in 30%, packed two wide."""
    centres = np.arange(4)[:, None] * 10.0 * np.ones(2)
    return packed_chain(np.random.default_rng(0), centres, np.full(4, 0.3), n_samples, 2)


def never_cooccurring_chain() -> np.ndarray:
    """Source A at 0 in every sample, and exactly one of B at +20 or C at -20 beside it."""
    rng = np.random.default_rng(0)
    chain = np.empty((1000, 2, 2))
    chain[:, 0] = rng.normal(size=(1000, 2))
    plus = rng.random(1000) < 0.5
    chain[:, 1] = np.where(plus[:, None], 20.0, -20.0) + rng.normal(size=(1000, 2))
    return chain


def case_b_chain(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """F003 case B: five sources in eight parameters, at least 10 sigma apart, present in 70%, packed."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(5, 8))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    centres = np.arange(5)[:, None] * 10.0 * directions
    return packed_chain(rng, centres, np.full(5, 0.7), 200, 5)


def case_c_chain(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """F003 case C: five sources in three parameters, std 0.5, presence U(0.45, 0.97), packed."""
    rng = np.random.default_rng(1000 + seed)
    centres = rng.uniform(-3, 3, size=(5, 3))
    presence = rng.uniform(0.45, 0.97, size=5)
    return packed_chain(rng, centres, presence, 400, 5, std=0.5)


def true_labeling(chain: np.ndarray, true_ids: np.ndarray, n_labels: int) -> np.ndarray:
    """Put every row of `chain` under the label of its true source."""
    truth = np.full((chain.shape[0], n_labels, chain.shape[2]), np.nan)
    for i, j in zip(*np.nonzero(true_ids >= 0)):
        truth[i, true_ids[i, j]] = chain[i, j]
    return truth


def case_c_reference(chain: np.ndarray, true_ids: np.ndarray) -> PosteriorChain:
    """The loop run from the true labeling of a case-C chain, without split moves."""
    truth = true_labeling(chain, true_ids, 5)
    return relabel_mv_normal(PosteriorChain(truth, 5, 3, True), 5, num_iterations=200, progress=False,
                             split_moves=False)


def three_full_labels_chain() -> np.ndarray:
    """A at 0, B or C (never both) at +-20, and D at 50, each in every sample: nothing to gain by a split."""
    chain = np.empty((300, 3, 2))
    chain[:, :2] = never_cooccurring_chain()[:300]
    chain[:, 2] = 50.0 + np.random.default_rng(1).normal(size=(300, 2))
    return chain


def two_clusters(n_samples: int = 60) -> np.ndarray:
    """Two well-separated sources in one parameter, both in every sample."""
    rng = np.random.default_rng(0)
    return rng.normal(scale=0.2, size=(n_samples, 2, 1)) + np.array([[0.0], [8.0]])


def same_catalog(first: PosteriorChain, second: PosteriorChain) -> bool:
    """Same labeling, same recorded cost and same prob_in_model."""
    return (np.array_equal(first.get_chain(), second.get_chain(), equal_nan=True)
            and first.cost_dict == second.cost_dict
            and np.array_equal(first.prob_in_model, second.prob_in_model))


# The behaviour reference


def ref_packed_chain(rng: np.random.Generator, means: list, P: int, N: int, presence: float,
                     width: int) -> np.ndarray:
    """Draw the reference's packed chain: each source present with probability `presence`, at most `width`."""
    rows = []
    for _ in range(N):
        present = [k for k in range(len(means)) if rng.random() < presence][:width]
        s = np.full((width, P), np.nan)
        for j, k in enumerate(present):
            s[j] = means[k] + rng.standard_normal(P)
        rows.append(s)
    return np.array(rows)


def ref_purity(catalog: PosteriorChain, means: list) -> float:
    """Return the reference's purity: the share of present rows nearest the majority mean of their label."""
    c = catalog.get_chain()
    m = np.array(means)
    tot = 0
    agree = 0
    for j in range(c.shape[1]):
        rows = c[:, j, :]
        rows = rows[~np.isnan(rows[:, 0])]
        if len(rows) == 0:
            continue
        src = np.argmin(((rows[:, None, :] - m[None, :, :]) ** 2).sum(-1), axis=1)
        agree += np.bincount(src).max()
        tot += len(rows)
    return float(agree / tot)


def golden_chain() -> PosteriorChain:
    """The 10,000 x 2 x 1 uniform chain drawn after np.random.seed(0)."""
    np.random.seed(0)
    chain = np.zeros((10_000, 2))
    chain_signal_1 = np.random.uniform(0, 1, 10000)
    chain_signal_2 = np.random.uniform(0, 1, 10000)
    chain[:, 0] = chain_signal_1
    chain[:, 1] = chain_signal_2
    return PosteriorChain(chain, 2, 1)


def chain_hash(a: np.ndarray) -> str:
    """Return the first 16 hex digits of the sha256 of the array's bytes."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def case_a_chain() -> PosteriorChain:
    """The reference's case A: four sources at 0, 10, 20, 30, present in 30%, packed two wide."""
    return PosteriorChain(ref_packed_chain(np.random.default_rng(0), [np.full(2, v) for v in (0., 10., 20., 30.)],
                                           2, 1000, 0.3, 2), 2, 2)


def case_b_ref_chain() -> PosteriorChain:
    """The reference's case B: five sources 10 apart in eight parameters, present in 70%, packed five wide."""
    return PosteriorChain(ref_packed_chain(np.random.default_rng(1), [np.full(8, 10.0 * k) for k in range(5)],
                                           8, 200, 0.7, 5), 5, 8)


#: The behaviour reference, copied once from the run of the code before the restructure.
REF = {
    'golden_one_iter': {
        'cost_dict': {'2': 0.36164766568844664},
        'first3': [
            [[0.5488135039273248], [0.7482679812896987]], [[0.7151893663724195], [0.18020270731629007]],
            [[0.6027633760716439], [0.38902314453248277]]
        ],
        'last3': [
            [[0.44645576033499734], [0.758429520713077]], [[0.36012661384309896], [0.023787427470386757]],
            [[0.625886645991811], [0.8135750799512289]]
        ],
    },
    'golden_low_level': {'total_cost': 0.3616456656874457, 'chain_hash': '1779bb77eee332ea'},
    'lisa_notebook': {
        'cost_dict': {'20': -1360.720117223451},
        'prob_in_model': [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.74, 0.55, 0.525, 0.595, 0.515, 0.22, 0.305, 0.135,
            0.015, 0.0
        ],
        'chain_hash': '528dca0e9c513f3f',
        'shape': [200, 20, 8],
    },
    'lisa_default': {
        'cost_dict': {'20': -1366.7279417145512},
        'prob_in_model': [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.74, 0.435, 0.405, 0.305, 0.46, 0.285, 0.19, 0.305,
            0.175, 0.3
        ],
        'chain_hash': 'c2ca8f850999e7bf',
        'shape': [200, 20, 8],
    },
    'case_A': {'purity': 1.0, 'cost_dict': {'4': 5.6165022367509945}, 'prob_in_model': [0.282, 0.274, 0.275, 0.31]},
    'case_B': {
        '0': {'purity': 1.0, 'cost': 42.82271061470644},
        '1': {'purity': 1.0, 'cost': 42.82271061470644},
        '2': {'purity': 1.0, 'cost': 42.82271061470644},
    },
    'resume': {
        'full_cost': {'4': 7.05029542031757},
        'resumed_cost': {'4': 7.05029542031757},
        'same_chain': True,
        'files': ['posterior_chain_iteration_001.feather'],
    },
    'feather': {'same_chain': True, 'same_prob': True, 'cost_dict': {'4': 5.6165022367509945}, 'dims': [4, 2]},
}
