"""Waveform-match refinement of a finished catalog: split labels that hold two sources, merge labels that are one.

The match is the normalized noise-weighted inner product of two sources' TDI waveforms (Littenberg et al. 2020), so
two posterior modes with identical waveforms count as one source and two sources that are close in parameter space
but orthogonal as waveforms stay apart. The waveforms come from a callable the caller supplies; see
:mod:`petra.gb_waveforms` for the optional galactic-binary adapter.
"""

import logging
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from petra.posterior_chain import PosteriorChain
from petra.utils import check_int, find_prob_in_model, source_present

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WaveformBank:
    """Whitened TDI waveforms of many sources on one frequency grid of bin width ``df``.

    Attributes:
        start: (n,) int64 index of each waveform's first bin, so its bins are ``start + arange(N)``.
        data: (n, C, N) complex128 channel values divided by the square root of the PSD at each bin.
        df: bin width in Hz.
    """

    start: np.ndarray
    data: np.ndarray
    df: float


@dataclass(frozen=True)
class Action:
    """One move of :func:`refine_catalog`.

    Attributes:
        kind: "split" or "merge".
        source_label: label the rows left.
        target_label: label the rows went to.
        rows: number of rows moved.
        match: split: mean match of the moved rows to their reference; merge: match of the two references.
    """

    kind: str
    source_label: int
    target_label: int
    rows: int
    match: float


@dataclass(frozen=True)
class MatchResult:
    """What :func:`refine_catalog` returns.

    Attributes:
        catalog: the refined catalog.
        actions: the moves, in the order they were made.
        rounds: rounds run, the last one included.
    """

    catalog: PosteriorChain
    actions: tuple[Action, ...]
    rounds: int


def _unit_waveforms(bank: WaveformBank, index: np.ndarray, lo: int, width: int) -> np.ndarray:
    """Return the normalized waveforms `index` on bins ``lo + arange(width)``, flattened to real (n, 2 C width)."""
    data = np.asarray(bank.data[index], dtype=np.complex128)
    norm = np.sqrt(np.sum(np.abs(data) ** 2, axis=(1, 2)))
    data = data * np.divide(1.0, norm, out=np.zeros_like(norm), where=norm > 0)[:, None, None]
    dense = np.zeros(data.shape[:2] + (width,), dtype=np.complex128)
    bins = (bank.start[index] - lo)[:, None, None] + np.arange(data.shape[2])
    np.put_along_axis(dense, np.broadcast_to(bins, data.shape), data, axis=2)
    return dense.reshape(len(index), -1).view(np.float64)


def _overlap_bands(start: np.ndarray, n_bins: int) -> np.ndarray:
    """Return the band of each window of `n_bins` bins starting at `start`; windows in different bands never overlap."""
    order = np.argsort(start, kind="stable")
    band = np.empty(start.size, dtype=np.int64)
    # Sorted by start, a band ends where the next window starts past the last one's end.
    band[order] = np.cumsum(np.diff(start[order], prepend=start[order[0]]) >= n_bins)
    return band


def bank_matches(bank: WaveformBank, rows: np.ndarray, columns: np.ndarray | None = None) -> np.ndarray:
    """Return the matches ``Re sum conj(a) b / (|a| |b|)`` of the waveforms `rows` against `columns`.

    The sum runs over channels and bins aligned by bin index; a zero-norm waveform matches 0. Rows are grouped into
    bands of overlapping windows and only same-band pairs are computed: windows that do not overlap match exactly 0.

    Parameters:
        bank: the waveforms.
        rows: bank indices of the rows.
        columns: bank indices of the columns; None means `rows`.

    Returns:
        (len(rows), len(columns)) float64 matrix.
    """
    rows = np.asarray(rows, dtype=np.int64)
    columns = rows if columns is None else np.asarray(columns, dtype=np.int64)
    out = np.zeros((rows.size, columns.size))
    if out.size == 0:
        return out
    n_bins = bank.data.shape[2]
    index = np.unique(np.concatenate([rows, columns]))
    band = _overlap_bands(bank.start[index], n_bins)
    row_band, column_band = band[np.searchsorted(index, rows)], band[np.searchsorted(index, columns)]
    for b in np.intersect1d(row_band, column_band):
        r, c = np.flatnonzero(row_band == b), np.flatnonzero(column_band == b)
        start = bank.start[np.concatenate([rows[r], columns[c]])]
        lo, width = int(start.min()), int(start.max() - start.min()) + n_bins
        out[np.ix_(r, c)] = _unit_waveforms(bank, rows[r], lo, width) @ _unit_waveforms(bank, columns[c], lo, width).T
    return out


def greedy_clusters(matches: np.ndarray, threshold: float) -> list[np.ndarray]:
    """Cluster rows around reference rows, the reference-sample rule of Littenberg et al. made deterministic.

    Among unassigned rows the reference is the one with the largest summed match to the others; it and every
    unassigned row matching it above `threshold` form a cluster. This repeats until every row is assigned.

    Parameters:
        matches: (n, n) symmetric match matrix.
        threshold: a row joins a reference when its match exceeds this.

    Returns:
        index arrays, each starting with its reference, sorted by size descending and then by smallest index.
    """
    matches = np.asarray(matches, dtype=np.float64)
    free = np.ones(len(matches), dtype=bool)
    score = matches.sum(axis=1) - np.diagonal(matches)
    clusters = []
    while free.any():
        reference = int(np.flatnonzero(free)[np.argmax(score[free])])
        joined = free & (matches[reference] > threshold)
        joined[reference] = False
        cluster = np.r_[reference, np.flatnonzero(joined)]
        clusters.append(cluster)
        free[cluster] = False
        score -= matches[:, cluster].sum(axis=1)
    return sorted(clusters, key=lambda cluster: (-cluster.size, cluster.min()))


def _clusters_one_per_sample(units: np.ndarray, sample: np.ndarray, threshold: float) -> list[np.ndarray]:
    """Run the rule of :func:`cluster_by_match` on unit waveforms (n, d), never forming the (n, n) matches."""
    score = units @ units.sum(axis=0) - np.einsum("ij,ij->i", units, units)
    free = np.ones(len(units), dtype=bool)
    clusters = []
    while free.any():
        reference = int(np.flatnonzero(free)[np.argmax(score[free])])
        match = units @ units[reference]
        joined = np.flatnonzero(free & (match > threshold) & (sample != sample[reference]))
        # Best match first, so np.unique's first index per sample is that sample's best row.
        joined = joined[np.argsort(-match[joined], kind="stable")]
        cluster = np.r_[reference, joined[np.unique(sample[joined], return_index=True)[1]]]
        clusters.append(cluster)
        free[cluster] = False
        score -= units @ units[cluster].sum(axis=0)
    return clusters


def cluster_by_match(catalog: PosteriorChain, waveforms: Callable[[np.ndarray], WaveformBank], *,
                     threshold: float = 0.5, batch_rows: int = 2000) -> PosteriorChain:
    """Catalog the present rows of a chain by the waveform match alone, ignoring the labels they carry.

    The reference-sample rule of :func:`greedy_clusters`, with one source per sample: among unassigned rows the
    reference is the one with the largest summed match to the others, and it takes from each other sample the
    unassigned row that matches it best, if that match exceeds `threshold`; the rest of the reference's own sample
    stays unassigned. This repeats until every row is assigned, and each cluster becomes a label, ordered by size
    descending and then by first row. Rows whose waveform windows do not overlap are never clustered together. The
    match matrix is never formed: memory grows as rows times the bins their windows span, not as rows squared.

    Parameters:
        catalog: chain whose present rows to cluster, such as the unclustered chain or any catalog of it.
        waveforms: maps an (n, P) block of present rows to their WaveformBank; called once.
        threshold: rows matching above this are one source, in (-1, 1).
        batch_rows: rows normalized at a time, >= 1.

    Returns:
        a trans-dimensional catalog of one label per cluster, with prob_in_model and an empty cost_dict.

    Raises:
        ValueError: an argument is out of range, the chain has no present row, or the bank does not hold one
            waveform per present row.
    """
    if not -1 < threshold < 1:
        raise ValueError(f"threshold must be finite and in (-1, 1), got {threshold!r}.")
    batch_rows = check_int("batch_rows", batch_rows)
    chain = catalog.chain
    samples, slots = present_rows(chain)
    if samples.size == 0:
        raise ValueError("cluster_by_match needs a chain with at least one present row.")
    bank = _bank_for(waveforms, chain[samples, slots])
    n_bins = bank.data.shape[2]
    band = _overlap_bands(bank.start, n_bins)
    clusters = []
    for b in np.unique(band):
        rows = np.flatnonzero(band == b)
        lo = int(bank.start[rows].min())
        width = int(bank.start[rows].max()) - lo + n_bins
        units = np.empty((rows.size, 2 * bank.data.shape[1] * width))
        for begin in range(0, rows.size, batch_rows):
            units[begin:begin + batch_rows] = _unit_waveforms(bank, rows[begin:begin + batch_rows], lo, width)
        clusters += [rows[cluster] for cluster in _clusters_one_per_sample(units, samples[rows], threshold)]
        del units
    clusters.sort(key=lambda cluster: (-cluster.size, cluster.min()))
    packed = np.full((chain.shape[0], len(clusters), chain.shape[2]), np.nan)
    for label, cluster in enumerate(clusters):
        packed[samples[cluster], label] = chain[samples[cluster], slots[cluster]]
    logger.info("Waveform match: %d present rows in %d labels.", samples.size, len(clusters))
    return PosteriorChain(packed, len(clusters), catalog.num_params_per_source, trans_dimensional=True,
                          prob_in_model=find_prob_in_model(packed, len(clusters), eps=0), cost_dict={},
                          validate_nan_convention=catalog.validate_nan_convention)


def _bank_for(waveforms: Callable[[np.ndarray], WaveformBank], rows: np.ndarray) -> WaveformBank:
    """Return the bank of `rows`, checked to hold one waveform per row."""
    bank = waveforms(rows)
    if bank.data.shape[0] != len(rows) or bank.start.shape != (len(rows),):
        raise ValueError(f"waveforms returned a bank with data of shape {bank.data.shape} and start of shape "
                         f"{bank.start.shape} for {len(rows)} present rows; both need {len(rows)} rows.")
    return bank


def present_rows(chain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the (samples, slots) indices of the present rows of an (N, S, P) chain, in row-major order."""
    samples, slots = np.nonzero(source_present(chain))
    return samples, slots


def refine_catalog(catalog: PosteriorChain, waveforms: Callable[[np.ndarray], WaveformBank], *,
                   threshold: float = 0.5, freq_gate_bins: int = 10, min_cluster_rows: int = 10,
                   min_cluster_fraction: float = 0.05, split: bool = True, merge: bool = True, max_rounds: int = 5,
                   grow: bool = False, progress: bool = False) -> MatchResult:
    """Split labels whose rows are several sources and merge labels that are one, by the waveform match.

    Each round runs a split pass over the labels, then a merge pass. A split keeps a label's largest cluster of
    :func:`greedy_clusters` and moves each other cluster of at least
    ``max(min_cluster_rows, ceil(min_cluster_fraction * rows))`` rows to a label absent from all its samples: the one
    whose rows best match the cluster's reference (median above `threshold`), else the lowest empty one; a cluster
    with neither goes to a new label with `grow`, and otherwise stays, and the call warns once. A label that received rows is not split in the same round. A merge moves the rows of a label into a lower one when the two never share a sample, their
    reference rows (the largest summed match within a label) start within `freq_gate_bins` bins and match above
    `threshold`; a label merges at most once per round. The loop stops after a round with no action. Moves permute
    rows within samples. Without `grow`, a one-slot catalog cannot split; widen it with
    :meth:`PosteriorChain.expand_chain` first.

    Parameters:
        catalog: a finished petra catalog; never modified.
        waveforms: maps an (n, P) block of present rows to their WaveformBank; called once.
        threshold: rows matching above this are one source, in (-1, 1); 0.5 in Littenberg et al. 2020, 0.8 in 2023.
        freq_gate_bins: largest difference in start bin of two merged labels' references, >= 0.
        min_cluster_rows: smallest cluster a split moves, >= 1; a label with fewer than twice this is not split.
        min_cluster_fraction: smallest cluster a split moves as a share of the label's rows, in [0, 1).
        split: run the split pass.
        merge: run the merge pass.
        max_rounds: most rounds to run, >= 1.
        grow: append a label for a split cluster that has nowhere to go, so that no catalog needs widening first.
        progress: show a bar over the labels in each split pass.

    Returns:
        the refined catalog, with prob_in_model recomputed and an empty cost_dict (the match is not a cost), the
        actions and the number of rounds run. With `grow` it has a label more for each cluster that needed one.

    Raises:
        ValueError: an argument is out of range, or the bank does not hold one waveform per present row.
    """
    if not -1 < threshold < 1:
        raise ValueError(f"threshold must be finite and in (-1, 1), got {threshold!r}.")
    if not 0 <= min_cluster_fraction < 1:
        raise ValueError(f"min_cluster_fraction must be in [0, 1), got {min_cluster_fraction!r}.")
    freq_gate_bins = check_int("freq_gate_bins", freq_gate_bins, 0)
    min_cluster_rows = check_int("min_cluster_rows", min_cluster_rows)
    max_rounds = check_int("max_rounds", max_rounds)
    num_labels = catalog.num_sources
    chain = catalog.chain.copy()
    samples, slots = present_rows(chain)
    bank = _bank_for(waveforms, chain[samples, slots])
    row_at = np.full(chain.shape[:2], -1, dtype=np.int64)
    row_at[samples, slots] = np.arange(samples.size)
    actions: list[Action] = []
    unplaced = 0

    def rows_of(label: int) -> np.ndarray:
        return row_at[row_at[:, label] >= 0, label]

    def move(sample: np.ndarray, source: int, target: int) -> None:
        for array in (chain, row_at):
            array[sample[:, None], [source, target]] = array[sample[:, None], [target, source]]

    def receiver(label: int, reference: int, sample: np.ndarray) -> int | None:
        absent = np.flatnonzero((np.arange(num_labels) != label) & np.all(row_at[sample] < 0, axis=0))
        held = [rows_of(int(t)) for t in absent]
        median = np.array([np.median(bank_matches(bank, np.array([reference]), r)) if r.size else -np.inf
                           for r in held])
        if median.size and median.max() > threshold:
            return int(absent[np.argmax(median)])
        empty = [int(t) for t, r in zip(absent, held) if r.size == 0]
        return empty[0] if empty else None

    for rounds in range(1, max_rounds + 1):
        done = len(actions)
        cache: dict[int, np.ndarray] = {}
        receivers: set[int] = set()
        split_labels = range(num_labels if split and (num_labels > 1 or grow) else 0)
        for k in tqdm(split_labels, disable=not (progress and split_labels), desc="Match split pass"):
            rows = rows_of(k)
            if k in receivers or rows.size < 2 * min_cluster_rows:
                continue
            cache[k] = matches = bank_matches(bank, rows)
            minimum = max(min_cluster_rows, math.ceil(min_cluster_fraction * rows.size))
            for cluster in greedy_clusters(matches, threshold)[1:]:
                if cluster.size < minimum:
                    break
                sample = samples[rows[cluster]]
                target = receiver(k, int(rows[cluster[0]]), sample)
                if target is None and grow:
                    target, num_labels = num_labels, num_labels + 1
                    chain = np.concatenate([chain, np.full((len(chain), 1, chain.shape[2]), np.nan)], axis=1)
                    row_at = np.concatenate([row_at, np.full((len(row_at), 1), -1)], axis=1)
                if target is None:
                    unplaced += cluster.size
                    continue
                move(sample, k, target)
                match = float(matches[cluster[0], cluster[1:]].mean()) if cluster.size > 1 else 1.0
                actions.append(Action("split", k, target, cluster.size, match))
                logger.info("Split: label %d gave %d rows (mean match %.3f to their reference) to label %d.",
                            k, cluster.size, match, target)
                receivers.add(target)
                cache.pop(k, None)
                cache.pop(target, None)
        if merge:
            labels: list[int] = []
            references: list[int] = []
            for k in range(num_labels):
                rows = rows_of(k)
                if rows.size:
                    matches = cache[k] if k in cache else bank_matches(bank, rows)
                    labels.append(k)
                    references.append(int(rows[np.argmax(matches.sum(axis=1) - np.diagonal(matches))]))
            present = (row_at[:, labels] >= 0).astype(np.float64)
            start = bank.start[references]
            matches = bank_matches(bank, np.array(references, dtype=np.int64))
            candidates = ((present.T @ present == 0) & (np.abs(start[:, None] - start[None, :]) <= freq_gate_bins)
                          & (matches > threshold))
            first, second = np.nonzero(np.triu(candidates, 1))
            merged: set[int] = set()
            for i in np.argsort(-matches[first, second], kind="stable"):
                a, b = labels[first[i]], labels[second[i]]
                if a in merged or b in merged:
                    continue
                sample = np.flatnonzero(row_at[:, b] >= 0)
                move(sample, b, a)
                merged |= {a, b}
                match = float(matches[first[i], second[i]])
                actions.append(Action("merge", b, a, sample.size, match))
                logger.info("Merge: label %d gave its %d rows to label %d (reference match %.3f).",
                            b, sample.size, a, match)
        if len(actions) == done:
            break
    if unplaced:
        warnings.warn(f"refine_catalog left {unplaced} rows of split clusters where they were because no label absent "
                      f"from their samples was empty or matched them; run the catalog with a larger max_num_sources "
                      f"(or widen it with expand_chain) so that split clusters have an empty label to go to.",
                      UserWarning, stacklevel=2)
    refined = PosteriorChain(chain, num_labels, catalog.num_params_per_source,
                             trans_dimensional=catalog.trans_dimensional,
                             prob_in_model=find_prob_in_model(chain, num_labels, eps=0), cost_dict={},
                             validate_nan_convention=catalog.validate_nan_convention)
    return MatchResult(refined, tuple(actions), rounds)
