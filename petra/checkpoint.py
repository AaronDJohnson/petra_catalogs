"""Relabeling checkpoints: the run fingerprint, atomic writes with pruning, discovery and verified loads.

A checkpoint is a PosteriorChain Feather file numbered by the iteration that found its labeling, with the run state
under :data:`CHECKPOINT_METADATA_KEY`; a resume refuses a file without it, of another version, or of another run.
"""

import contextlib
import hashlib
import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from petra import __version__
from petra.posterior_chain import PosteriorChain

logger = logging.getLogger(__name__)

CHECKPOINT_TEMPLATE = "posterior_chain_iteration_{iteration:03d}.feather"
_CHECKPOINT_RE = re.compile(r"^posterior_chain_iteration_(\d+)\.feather$")
CHECKPOINT_METADATA_KEY = b"petra_checkpoint"
CHECKPOINT_METADATA_VERSION = 2
_DIGEST_BLOCK_ROWS = 1 << 18


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint read back by :func:`load_checkpoint`.

    Parameters:
        path: the file read.
        file_iteration: the number in its name, the iteration its labeling was found at; 0 for an unnumbered file.
        completed_iterations: iterations the run that wrote it had completed, ``state["iteration"]``.
        posterior_chain: the checkpointed labeling, with its prob_in_model and cost_dict.
        state: the run fingerprint and progress stored with it.
    """

    path: str
    file_iteration: int
    completed_iterations: int
    posterior_chain: PosteriorChain
    state: dict[str, Any]


def run_fingerprint(posterior_chain: PosteriorChain, *, method: str, max_num_sources: int, eps: float,
                    settings: Mapping[str, Any] | None = None,
                    run_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the fingerprint every checkpoint of a run carries and a resume is checked against.

    Parameters:
        posterior_chain: the chain the run was given; shuffling, widening and relabeling leave its digest unchanged.
        method: name of the relabeling method; only that method may resume the run.
        max_num_sources: the run's number of source slots.
        eps: the run's clip bound.
        settings: values that change what a cost means, keyed by the caller's keyword names; compared on resume.
        run_info: settings of the entry point, recorded and never compared.

    Returns:
        the fingerprint, already in its form after a JSON round trip, so it compares equal to one read back.
    """
    fingerprint = {
        "version": CHECKPOINT_METADATA_VERSION,
        "petra_version": __version__,
        "method": method,
        "settings": {str(key): _json_value(value) for key, value in (settings or {}).items()},
        "max_num_sources": int(max_num_sources),
        "eps": float(eps),
        "n_samples": int(posterior_chain.shape[0]),
        "num_params_per_source": int(posterior_chain.num_params_per_source),
        "input_sha256": _chain_digest(posterior_chain.chain),
        "validate_nan_convention": bool(posterior_chain.validate_nan_convention),
        "run_info": {str(key): _json_value(value) for key, value in (run_info or {}).items()},
    }
    return json.loads(json.dumps(fingerprint, allow_nan=False))


def _json_value(value: Any) -> Any:
    """Return a setting's JSON form, the same in every run: a non-finite float as its repr, an object as its type."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    return f"<{type(value).__qualname__}>"


def _chain_digest(chain: np.ndarray) -> str:
    """Return a SHA-256 digest of each sample's sorted present rows, which shuffling, widening and relabeling keep."""
    n_samples, n_slots, n_params = chain.shape
    counts_digest = hashlib.sha256()
    rows_digest = hashlib.sha256()
    # Whole samples per block: both streams are concatenations in sample order, so neither the block size nor the
    # width it depends on can change the digest.
    block = max(1, _DIGEST_BLOCK_ROWS // max(1, n_slots))
    for start in range(0, n_samples, block):
        part = chain[start:start + block]
        rows = part.reshape(-1, n_params) + 0.0          # a copy, with -0.0 read as 0.0
        is_nan: np.ndarray = np.isnan(rows)
        rows[is_nan] = np.nan                            # one NaN bit pattern
        present = np.logical_not(np.all(is_nan, axis=1))
        sample = np.repeat(np.arange(part.shape[0]), n_slots)
        order = np.lexsort(tuple(rows[:, j] for j in range(n_params - 1, -1, -1)) + (sample,))
        order = order[present[order]]
        counts_digest.update(np.bincount(sample[present], minlength=part.shape[0]).astype("<i8").tobytes())
        rows_digest.update(np.ascontiguousarray(rows[order], dtype="<f8").tobytes())
    header = f"petra-chain-digest-v1 n_samples={n_samples} n_params={n_params}\n".encode("ascii")
    return hashlib.sha256(header + counts_digest.digest() + rows_digest.digest()).hexdigest()


def _numbered(checkpoint_dir: str) -> list[tuple[int, str]]:
    """Return a directory's checkpoint files as sorted ``(number, filename)`` pairs; none if it does not exist."""
    if not os.path.isdir(checkpoint_dir):
        return []
    found = []
    for filename in os.listdir(checkpoint_dir):
        match = _CHECKPOINT_RE.match(filename)
        if match is not None:
            found.append((int(match.group(1)), filename))
    return sorted(found)


def prepare_checkpoint_dir(checkpoint_dir: str | os.PathLike[str], resumed: Checkpoint | None = None) -> None:
    """Create and check a run's checkpoint directory before any work is done; nothing is ever deleted.

    Parameters:
        checkpoint_dir: directory the run will write its checkpoints to; created if missing.
        resumed: the checkpoint a resumed run continues from, or None for a fresh run.

    Raises:
        ValueError: a fresh run's directory holds checkpoints; a resume's own directory holds one numbered above the
            resumed file or another with its number; any other directory given to a resume holds checkpoints.
        OSError: the directory cannot be created, or (PermissionError) is not writable.
    """
    checkpoint_dir = os.fspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    probe = os.path.join(checkpoint_dir, f".petra_write_test_{uuid.uuid4().hex}.tmp")
    try:
        with open(probe, "wb"):
            pass
        os.remove(probe)
    except OSError as exc:
        raise PermissionError(f"checkpoint_dir {checkpoint_dir!r} is not writable: {exc}") from exc

    existing = _numbered(checkpoint_dir)
    if resumed is None:
        if existing:
            names = ", ".join(name for _, name in existing)
            raise ValueError(
                f"checkpoint_dir {checkpoint_dir!r} already holds checkpoints of an earlier run "
                f"({names}). Pass resume_from={checkpoint_dir!r} to continue that run, or give "
                f"this run an empty directory: petra never deletes checkpoints, and a later "
                f"resume from this directory could pick up the other run's files."
            )
        return
    resume_path, resume_number = resumed.path, resumed.file_iteration
    resumed_file = os.path.realpath(resume_path)
    if existing and os.path.realpath(checkpoint_dir) != os.path.dirname(resumed_file):
        names = ", ".join(name for _, name in existing)
        raise ValueError(
            f"checkpoint_dir {checkpoint_dir!r} is not the directory of the checkpoint being "
            f"resumed ({resume_path!r}), and it already holds checkpoints of another run "
            f"({names}). This run could overwrite them, keep_checkpoints would delete them, and a "
            f"later resume from that directory could mix the two runs. Write this run's "
            f"checkpoints to an empty directory, or to the resumed checkpoint's own directory."
        )
    # Above the resumed number a later directory resume would pick the file up; at it, this run's stop record would
    # overwrite it, unless it is the very file being resumed, which this run owns.
    in_the_way = [name for number, name in existing
                  if number > resume_number
                  or (number == resume_number
                      and os.path.realpath(os.path.join(checkpoint_dir, name)) != resumed_file)]
    if in_the_way:
        raise ValueError(
            f"checkpoint_dir {checkpoint_dir!r} holds checkpoints of another run numbered at or "
            f"above the one being resumed ({resume_path!r}): {', '.join(in_the_way)}. This run "
            f"could overwrite them, and a later resume from that directory could pick them up "
            f"instead of this run's. Resume from the directory itself to continue from its newest "
            f"checkpoint, or write this run's checkpoints to an empty directory."
        )


def write_checkpoint(posterior_chain: PosteriorChain, checkpoint_dir: str | os.PathLike[str], iteration: int,
                     state: Mapping[str, Any], *, keep: int | None = None) -> str:
    """Write a labeling and its run state as a checkpoint, atomically, then prune the directory's older checkpoints.

    Parameters:
        posterior_chain: the labeling to save, with the prob_in_model and cost_dict a resume needs.
        checkpoint_dir: the run's directory; created if missing.
        iteration: iteration at which the labeling was found, the number in the file name.
        state: run fingerprint and progress, stored as strict JSON under CHECKPOINT_METADATA_KEY.
        keep: if given, only this many highest-numbered checkpoints of the directory are kept after the write.

    Returns:
        the path written.

    Raises:
        ValueError: a cost_dict entry or a state value cannot be stored as strict JSON; nothing is written.
    """
    checkpoint_dir = os.fspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = CHECKPOINT_TEMPLATE.format(iteration=iteration)
    filepath = os.path.join(checkpoint_dir, filename)

    table = posterior_chain.to_arrow_table()
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           CHECKPOINT_METADATA_KEY: json.dumps(dict(state), allow_nan=False).encode()})

    # A name the template never matches, renamed into place: a job killed mid-write leaves the previous file intact.
    temporary = os.path.join(checkpoint_dir, f".{filename}.{uuid.uuid4().hex}.tmp")
    try:
        feather.write_feather(table, temporary)
        os.replace(temporary, filepath)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temporary)
        raise
    logger.info("  Checkpoint saved: %s", filepath)
    if keep is not None:
        for _, name in _numbered(checkpoint_dir)[:-keep]:
            # Another process may have removed it already; that is the outcome wanted.
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(checkpoint_dir, name))
            logger.info("  Checkpoint pruned: %s", os.path.join(checkpoint_dir, name))
    return filepath


def find_latest_checkpoint(checkpoint_dir: str | os.PathLike[str]) -> tuple[str, int] | None:
    """Locate the highest-numbered checkpoint of a directory without opening any file.

    Parameters:
        checkpoint_dir: a directory a run wrote checkpoints to.

    Returns:
        its path and number, the iteration its labeling was found at, or None if the directory is missing or holds
        none; the temporary files a write goes through never match.
    """
    checkpoint_dir = os.fspath(checkpoint_dir)
    found = _numbered(checkpoint_dir)
    if not found:
        return None
    iteration, filename = found[-1]
    return os.path.join(checkpoint_dir, filename), iteration


def load_checkpoint(resume_from: str | os.PathLike[str], *,
                    fingerprint: Mapping[str, Any] | None = None) -> Checkpoint:
    """Read a checkpoint file, or a directory's newest one, and check it against the run that resumes it.

    Parameters:
        resume_from: a checkpoint file, or a directory of them.
        fingerprint: the resuming run's :func:`run_fingerprint`; None only reads.

    Returns:
        the checkpoint, its chain read with the validate_nan_convention its run stored.

    Raises:
        FileNotFoundError: resume_from is neither a file nor a directory holding a checkpoint.
        ValueError: the file cannot be read (the message names an older one to resume from), is not a petra
            checkpoint, holds run state of another layout version, or was written by another run: other samples,
            shape, width, method, settings or eps.
    """
    resume_from = os.fspath(resume_from)
    if os.path.isdir(resume_from):
        latest = find_latest_checkpoint(resume_from)
        if latest is None:
            raise FileNotFoundError(f"No relabeling checkpoint found in {resume_from!r}.")
        path, file_iteration = latest
    elif os.path.isfile(resume_from):
        path = resume_from
        match = _CHECKPOINT_RE.match(os.path.basename(resume_from))
        file_iteration = int(match.group(1)) if match is not None else 0
    else:
        raise FileNotFoundError(f"No such checkpoint file or directory: {resume_from!r}.")

    try:
        table = feather.read_table(path)
    except (pa.ArrowException, OSError) as exc:
        directory = os.path.dirname(path) or "."
        older = [name for number, name in _numbered(directory) if number < file_iteration]
        advice = (f"Resume from an older checkpoint instead: resume_from={os.path.join(directory, older[-1])!r}."
                  if older else "There is no older checkpoint beside it to resume from.")
        raise ValueError(
            f"Checkpoint {path!r} cannot be read ({type(exc).__name__}: {exc}). A job "
            f"killed while writing it can leave it truncated; petra 1.1.0 and earlier wrote "
            f"checkpoints in place. {advice}"
        ) from exc

    raw_state = (table.schema.metadata or {}).get(CHECKPOINT_METADATA_KEY)
    if raw_state is None:
        raise ValueError(f"Checkpoint {path!r} is not a petra checkpoint: it carries no run state. "
                         f"Start a fresh run in an empty directory.")
    try:
        state = json.loads(raw_state.decode("utf-8"))
    except ValueError as exc:       # bad UTF-8 or bad JSON; both are ValueErrors
        raise ValueError(f"Checkpoint {path!r} holds unreadable run metadata: {exc}.") from exc
    version = state.get("version") if isinstance(state, dict) else None
    if version != CHECKPOINT_METADATA_VERSION:
        raise ValueError(
            f"Checkpoint {path!r} holds run metadata of layout version {version!r}, but this "
            f"petra ({__version__}) reads version {CHECKPOINT_METADATA_VERSION}. Resume it "
            f"with the petra version that wrote it."
        )
    posterior_chain = PosteriorChain.from_arrow_table(table, path, bool(state["validate_nan_convention"]))

    if fingerprint is not None:
        n_samples, n_params = posterior_chain.shape[0], posterior_chain.num_params_per_source
        if (n_samples, n_params) != (fingerprint["n_samples"], fingerprint["num_params_per_source"]):
            raise ValueError(
                f"Checkpoint {path!r} holds {n_samples} samples of {n_params} parameters per source, "
                f"but the chain passed to resume it has {fingerprint['n_samples']} samples of "
                f"{fingerprint['num_params_per_source']} parameters per source, so it belongs to a "
                f"different run. Pass the chain that run was started with, or start a fresh run in "
                f"an empty directory."
            )
        if posterior_chain.num_sources != fingerprint["max_num_sources"]:
            raise ValueError(
                f"Checkpoint {path!r} holds a chain {posterior_chain.num_sources} source slots wide, but this "
                f"run asks for max_num_sources={fingerprint['max_num_sources']}. A run can only be "
                f"resumed at the width it was started with: the cost stored with the checkpoint is "
                f"not comparable at another. Resume with max_num_sources={posterior_chain.num_sources}, or "
                f"start a fresh run in an empty directory."
            )
        if state["method"] != fingerprint["method"]:
            raise ValueError(
                f"Checkpoint {path!r} was written by a different method ({state['method']!r}) than this run "
                f"uses ({fingerprint['method']!r}). Costs from different methods are not comparable, so a run "
                f"can only be resumed by the method that wrote it."
            )
        if state["settings"] != fingerprint["settings"]:
            raise ValueError(
                f"Checkpoint {path!r} was written with other settings ({state['settings']!r}) than this run "
                f"uses ({fingerprint['settings']!r}). Costs from different settings are not comparable; "
                f"resume with the settings the run started with."
            )
        if state["eps"] != fingerprint["eps"]:
            raise ValueError(
                f"Checkpoint {path!r} was written with eps={state['eps']!r}, but this run uses "
                f"eps={fingerprint['eps']!r}. eps sets the scale of every inclusion term in the cost, "
                f"so the stored cost is not comparable; resume with eps={state['eps']!r}."
            )
        if state["input_sha256"] != fingerprint["input_sha256"]:
            raise ValueError(
                f"Checkpoint {path!r} was written for a different input chain: the SHA-256 digest of "
                f"its samples is {state['input_sha256']}, but that of the chain passed to resume it is "
                f"{fingerprint['input_sha256']}. Pass the chain that run was started with, or start a "
                f"fresh run in an empty directory."
            )
    return Checkpoint(path, file_iteration, int(state["iteration"]), posterior_chain, state)
