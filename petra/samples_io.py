"""Readers that turn sampler output on disk into a :class:`~petra.posterior_chain.PosteriorChain`.

:func:`load_samples` picks the reader: a directory is UCBMCMC output, a text file is fixed-num-sources (4 trailing
metadata columns) or product-space (5). Text is UTF-8 with an optional byte-order mark; fill_value must be NaN.
"""

import logging
import os
import re
import warnings
from typing import Any

import numpy as np

from petra.posterior_chain import PosteriorChain
from petra.utils import check_int

logger = logging.getLogger(__name__)

_UCBMCMC_FILE = re.compile(r"dimension_chain\.dat\.([0-9]+)")
_UCBMCMC_PREFIX = "dimension_chain.dat."


def _check_fill_value(fill_value: Any) -> None:
    """Require the NaN padding understood by every source-presence calculation."""
    if not isinstance(fill_value, (float, np.floating)) or not np.isnan(fill_value):
        raise ValueError("fill_value must be NaN; only NaN marks absent sources in a PosteriorChain.")


def _read_rows(path: str, burn: int = 0, thin: int = 1, *, allow_empty: bool = False) -> np.ndarray:
    """Parse a whitespace-delimited UTF-8 text file into rows and return ``rows[burn::thin]``, refusing none."""
    with open(path, encoding="utf-8-sig") as handle, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loadtxt: input contained no data", category=UserWarning)
        rows = np.loadtxt(handle, ndmin=2)
    if allow_empty:
        return rows
    if not rows.shape[0]:
        raise ValueError(f"{path!r} contains no data rows.")
    kept = rows[burn::thin]
    if kept.shape[0] == 0:
        raise ValueError(
            f"No samples left in {path!r} after burn={burn} and thin={thin}: the file has "
            f"{rows.shape[0]} samples, so burn must be smaller than {rows.shape[0]}."
        )
    return kept


def _count_data_columns(path: str) -> int:
    """Count the fields of the first data row of a text file, reading nothing past it."""
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.split("#", 1)[0].strip()
            if not stripped or stripped.startswith("#"):
                continue
            return len(stripped.split())
    raise ValueError(f"{path!r} contains no data rows.")


def load_samples(path: str | os.PathLike[str],
                 num_params_per_source: int,
                 fill_value: float = np.nan,
                 burn: int = 0,
                 thin: int = 1,
                 remove_low_numbers: bool = False) -> PosteriorChain:
    """Load a chain, choosing the reader from the path: a directory, or a text file's first data row.

    Parameters:
        path: chain text file or UCBMCMC directory.
        num_params_per_source: parameters per source P.
        fill_value: padding for absent sources; must be NaN.
        burn: samples to discard, an integer >= 0; per file for a UCBMCMC directory.
        thin: keep every thin-th sample, an integer >= 1; per file for a UCBMCMC directory.
        remove_low_numbers: skip UCBMCMC files with at most 8 samples; ignored for a text file.

    Returns:
        the loaded chain.

    Raises:
        ValueError: an argument is invalid, the file fits neither or both text layouts, or the reader refuses it.
        FileNotFoundError: path does not exist.
    """
    path = os.fspath(path)
    num_params_per_source = check_int("num_params_per_source", num_params_per_source)
    _check_fill_value(fill_value)
    burn, thin = check_int("burn", burn, 0), check_int("thin", thin, 1)

    if os.path.isdir(path):
        return load_samples_ucbmcmc(path,
                                    num_params_per_source,
                                    fill_value=fill_value,
                                    burn=burn,
                                    thin=thin,
                                    remove_low_numbers=remove_low_numbers)

    n_cols = _count_data_columns(path)

    # With one parameter per source every wide file fits both layouts, and the values cannot tell them apart.
    is_product_space = n_cols > 5 and (n_cols - 5) % num_params_per_source == 0
    is_fixed_num_sources = n_cols > 4 and (n_cols - 4) % num_params_per_source == 0
    if is_product_space and is_fixed_num_sources:
        raise ValueError(
            f"The layout of {path!r} is ambiguous for num_params_per_source="
            f"{num_params_per_source}; call load_samples_fixed_num_sources or "
            "load_samples_product_space explicitly."
        )

    if is_product_space:
        return load_samples_product_space(path,
                                          num_params_per_source,
                                          fill_value=fill_value,
                                          burn=burn,
                                          thin=thin)

    if is_fixed_num_sources:
        return load_samples_fixed_num_sources(path,
                                              num_params_per_source,
                                              burn=burn,
                                              thin=thin)

    fixed_fits = [k for k in range(1, n_cols - 3) if (n_cols - 4) % k == 0]
    product_fits = [k for k in range(1, n_cols - 4) if (n_cols - 5) % k == 0]
    raise ValueError(
        f"Cannot infer loader for {path!r}: its first data row has {n_cols} columns, which fits "
        f"neither layout for num_params_per_source={num_params_per_source}. A fixed-num-sources "
        "file ends in 4 metadata columns and a product-space file in 5, and the columns before "
        "them must be a positive multiple of num_params_per_source. For "
        f"{n_cols} columns that holds for num_params_per_source in {fixed_fits or 'none'} "
        f"(fixed-num-sources) and {product_fits or 'none'} (product-space); a value that fits "
        "both layouts is ambiguous and needs the explicit reader. A UCBMCMC "
        "dimension_chain.dat.<n> file is read through its directory: pass the directory instead."
    )


def load_samples_product_space(filepath: str | os.PathLike[str],
                               num_params_per_source: int,
                               fill_value: float = np.nan,
                               burn: int = 0,
                               thin: int = 1) -> PosteriorChain:
    """Load a product-space text file, whose fifth-from-last column is the number of sources minus one.

    Parameters:
        filepath: chain text file.
        num_params_per_source: parameters per source P.
        fill_value: padding for absent sources; must be NaN.
        burn: samples to discard, an integer >= 0 below the sample count.
        thin: keep every thin-th sample, an integer >= 1.

    Returns:
        a trans-dimensional chain as wide as the largest source count kept after burn and thin.

    Raises:
        ValueError: an argument is invalid, no sample is left, every kept sample has zero sources,
            or the columns or source counts do not describe a product-space layout.
    """
    filepath = os.fspath(filepath)
    num_params_per_source = check_int("num_params_per_source", num_params_per_source)
    _check_fill_value(fill_value)
    burn, thin = check_int("burn", burn, 0), check_int("thin", thin, 1)
    chain = _read_rows(filepath, burn, thin)
    num_data_columns = chain.shape[1] - 5
    if num_data_columns < 0 or num_data_columns % num_params_per_source:
        raise ValueError(
            "Product-space data columns must contain complete source vectors "
            "followed by five metadata columns."
        )
    counts = chain[:, -5] + 1
    available_sources = num_data_columns // num_params_per_source
    if not np.all(
        np.isfinite(counts)
        & (counts == np.floor(counts))
        & (counts >= 0)
        & (counts <= available_sources)
    ):
        raise ValueError(
            "Product-space source counts must be finite integers between zero "
            f"and the {available_sources} sources available in the data columns."
        )
    num_sources_vector = counts.astype(int)
    num_sources = np.max(num_sources_vector)
    if num_sources == 0:
        raise ValueError(
            f"Every sample kept from {filepath!r} after burn={burn} and thin={thin} has zero "
            "sources (the fifth-from-last column, the number of sources minus one, is -1 in "
            "every kept row), so the chain has no source to catalog."
        )

    only_samples_chain = np.full((chain.shape[0], num_sources * num_params_per_source),
                                 fill_value, dtype=float)
    for i in range(chain.shape[0]):
        only_samples_chain[i, : num_sources_vector[i] * num_params_per_source] = chain[
            i, : num_sources_vector[i] * num_params_per_source
        ]
    samples = only_samples_chain.reshape(chain.shape[0], num_sources, num_params_per_source)
    return PosteriorChain(
        samples, num_sources, num_params_per_source, trans_dimensional=True
    )


def load_samples_fixed_num_sources(filepath: str | os.PathLike[str],
                                   num_params_per_source: int,
                                   burn: int = 0,
                                   thin: int = 1) -> PosteriorChain:
    """Load a fixed-num-sources text file: source vectors followed by four metadata columns.

    Parameters:
        filepath: chain text file.
        num_params_per_source: parameters per source P.
        burn: samples to discard, an integer >= 0 below the sample count.
        thin: keep every thin-th sample, an integer >= 1.

    Returns:
        the chain, with trans_dimensional False.

    Raises:
        ValueError: an argument is invalid, no sample is left, or the rows are not whole source vectors plus 4 columns.
    """
    filepath = os.fspath(filepath)
    num_params_per_source = check_int("num_params_per_source", num_params_per_source)
    burn, thin = check_int("burn", burn, 0), check_int("thin", thin, 1)
    chain = _read_rows(filepath, burn, thin)[:, :-4]
    if chain.shape[1] == 0 or chain.shape[1] % num_params_per_source:
        raise ValueError(
            "Fixed-source data columns must contain complete source vectors "
            "followed by four metadata columns."
        )
    num_sources = (chain.shape[1]) // num_params_per_source
    samples = chain.reshape(chain.shape[0], num_sources, num_params_per_source)
    return PosteriorChain(samples, num_sources, num_params_per_source)


def load_samples_ucbmcmc(chain_folder: str | os.PathLike[str],
                         num_params_per_source: int = 8,
                         fill_value: float = np.nan,
                         burn: int = 0,
                         thin: int = 1,
                         remove_low_numbers: bool = False) -> PosteriorChain:
    """Load a UCBMCMC directory of ``dimension_chain.dat.<n>`` files, n consecutive rows per sample.

    The result is ordered by dimension, not by iteration: every n = 1 sample, then every n = 2 sample, and so on.

    Parameters:
        chain_folder: the directory, its name taken literally.
        num_params_per_source: parameters per source P.
        fill_value: padding for absent sources; must be NaN.
        burn: samples to discard from each file separately, which reweights the dimensions; leave it 0.
        thin: keep every thin-th sample of each file, an integer >= 1.
        remove_low_numbers: skip files with at most 8 samples, counted before burn and thin.

    Returns:
        a trans-dimensional chain as wide as the largest n holding samples; burn and thin never narrow it.

    Raises:
        ValueError: an argument is invalid, no file holds samples, burn and thin empty every file,
            or a file is not a whole number of P-column samples.
        FileNotFoundError: chain_folder does not exist.
        NotADirectoryError: chain_folder is not a directory.

    Warns:
        UserWarning: a file holds no samples, burn and thin empty a file, or a file name does not end in a number.
    """
    chain_folder = os.fspath(chain_folder)
    num_params_per_source = check_int("num_params_per_source", num_params_per_source)
    _check_fill_value(fill_value)
    burn, thin = check_int("burn", burn, 0), check_int("thin", thin, 1)

    if not os.path.exists(chain_folder):
        raise FileNotFoundError(f"UCBMCMC chain directory {chain_folder!r} does not exist.")
    if not os.path.isdir(chain_folder):
        raise NotADirectoryError(
            f"{chain_folder!r} is not a directory: load_samples_ucbmcmc reads a directory of "
            "dimension_chain.dat.<n> files."
        )
    files, ignored = [], []
    for name in os.listdir(chain_folder):
        match = _UCBMCMC_FILE.fullmatch(name)
        if match:
            files.append((int(match.group(1)), os.path.join(chain_folder, name)))
        elif name.startswith(_UCBMCMC_PREFIX):
            ignored.append(name)
    if ignored:
        warnings.warn(
            f"Ignoring {len(ignored)} file(s) in {chain_folder!r} whose names start with "
            f"{_UCBMCMC_PREFIX!r} but do not end in a number of sources: {', '.join(sorted(ignored))}.",
            UserWarning,
            stacklevel=2,
        )

    blocks = []
    empty_files = []
    emptied_by_burn = []
    max_sources = 0
    for nsources, filepath in sorted(files):
        if nsources == 0:
            continue

        chain = _read_rows(filepath, allow_empty=True)
        # An empty file parses as shape (0, 1); it is not a column-count error.
        if chain.shape[0] and chain.shape[1] != num_params_per_source:
            raise ValueError(
                f"{filepath!r} has {chain.shape[1]} columns, but "
                f"num_params_per_source={num_params_per_source} was requested."
            )
        # Samples are counted from the parsed array, so trailing newlines and comments cannot skew the count.
        if chain.shape[0] % nsources != 0:
            raise ValueError(
                f"{filepath!r} has {chain.shape[0]} rows, which is not a multiple of the "
                f"{nsources} sources implied by its name."
            )
        nsamples = chain.shape[0] // nsources

        if remove_low_numbers and nsamples <= 8:
            logger.debug("Skipping %s: only %d samples.", filepath, nsamples)
            continue
        if nsamples == 0:
            empty_files.append(filepath)
            continue

        # The width is fixed before burn and thin, so a dimension the sampler visited keeps its slots.
        max_sources = max(max_sources, nsources)
        block = chain.reshape(nsamples, nsources, num_params_per_source)[burn::thin]
        if block.shape[0] == 0:
            emptied_by_burn.append((filepath, nsamples))
            continue
        blocks.append((block, nsources))

    if empty_files:
        warnings.warn(
            f"Skipped {len(empty_files)} UCBMCMC dimension file(s) in {chain_folder!r} that hold "
            f"no samples: {', '.join(os.path.basename(path) for path in empty_files)}. The "
            "sampler never recorded a sample in those dimensions, so they add no source slots.",
            UserWarning,
            stacklevel=2,
        )
    if emptied_by_burn:
        lost = ", ".join(f"{os.path.basename(path)} (all {nsamples} samples)"
                         for path, nsamples in emptied_by_burn)
        if not blocks:
            raise ValueError(
                f"No samples left in {chain_folder!r} after burn={burn} and thin={thin}, which "
                f"are applied to each dimension file separately: {lost}. Load UCBMCMC output "
                "with burn=0."
            )
        warnings.warn(
            f"burn={burn} and thin={thin} are applied to each UCBMCMC dimension file in "
            f"{chain_folder!r} separately, and left no samples in {lost}. Those dimensions are "
            "missing from the chain although the sampler visited them, which reweights the "
            "posterior on the number of sources; their source slots are kept, empty. Load "
            "UCBMCMC output with burn=0.",
            UserWarning,
            stacklevel=2,
        )
    if not blocks:
        raise ValueError(
            f"No usable chain files found in {chain_folder!r}: it needs a dimension_chain.dat.<n> "
            f"file with n >= 1 that holds samples (remove_low_numbers={remove_low_numbers}, "
            f"burn={burn}, thin={thin})."
        )

    samples = np.full((sum(block.shape[0] for block, _ in blocks), max_sources, num_params_per_source),
                      fill_value, dtype=float)
    current_total = 0
    for block, nsources in blocks:
        samples[current_total: current_total + block.shape[0], :nsources, :] = block
        current_total += block.shape[0]

    return PosteriorChain(
        samples,
        max_sources,
        trans_dimensional=True,
        num_params_per_source=num_params_per_source,
    )
