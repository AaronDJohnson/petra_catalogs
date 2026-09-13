"""
Readers that turn sampler output on disk into a :class:`~petra.posterior_chain.PosteriorChain`.

:func:`load_samples` is the entry point: give it a path and the number of
parameters per source and it picks the right reader.  The three readers behind
it can also be called directly when the layout is already known:

:func:`load_samples_product_space`
    One text file per run, with five trailing metadata columns whose fifth-from-last
    holds the number of sources minus one in that sample.
:func:`load_samples_fixed_num_sources`
    One text file per run, four trailing metadata columns, a constant number of
    sources.
:func:`load_samples_ucbmcmc`
    A *directory* of ``dimension_chain.dat.<n>`` files, one per source count,
    each storing ``n`` consecutive rows per sample.

Conventions
-----------
All three return a chain shaped ``(n_samples, n_sources, n_params_per_source)``
in which ``NaN`` means "this source is absent from this sample"; see
:mod:`petra.posterior_chain`. The `fill_value` argument must be NaN; another
sentinel would be interpreted as a real source by the catalog methods. Chains that can
vary in dimension are flagged ``trans_dimensional=True``.

`burn` and `thin` mean the same thing everywhere and are honoured by every
branch: discard the first `burn` samples, then keep every `thin`-th one.  For a
directory of UCBMCMC files they are applied to each file *before* the files are
concatenated, so every file's own burn-in is discarded rather than only the
first one's.

Nothing here parses a file twice: :func:`load_samples` reads only the first data
row to choose a reader, and the reader it picks does the single full parse.
Diagnostics go to ``logging.getLogger("petra.samples_io")``.
"""

import glob
import os

import numpy as np

from petra.utils import get_logger, sort_by_number
from petra.posterior_chain import PosteriorChain

logger = get_logger(__name__)


def _validate_num_params_per_source(num_params_per_source: int) -> int:
    """Return a normalized positive parameter count or raise a clear error."""
    if (
        not isinstance(num_params_per_source, (int, np.integer))
        or isinstance(num_params_per_source, bool)
        or num_params_per_source < 1
    ):
        raise ValueError(
            "num_params_per_source must be a positive integer, "
            f"got {num_params_per_source!r}."
        )
    return int(num_params_per_source)


def _validate_fill_value(fill_value: float) -> None:
    """Require the NaN padding understood by every source-presence calculation."""
    if not isinstance(fill_value, (float, np.floating)) or not np.isnan(fill_value):
        raise ValueError("fill_value must be NaN; only NaN marks absent sources in a PosteriorChain.")


def _count_data_columns(filepath: str) -> int:
    """
    Count the columns of the first data row of a whitespace-delimited text file.

    Only the first non-empty, non-comment line is read, so this is cheap even
    for multi-gigabyte chains.

    Parameters
    ----------
    filepath : str
        Path to the text file.

    Returns
    -------
    int
        Number of whitespace-separated fields in the first data row.

    Raises
    ------
    ValueError
        If the file contains no data rows.
    """
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.split("#", 1)[0].strip()
            if not stripped or stripped.startswith("#"):
                continue
            return len(stripped.split())
    raise ValueError(f"{filepath!r} contains no data rows.")


def load_samples(path: str,
                 num_params_per_source: int,
                 fill_value: float = np.nan,
                 burn: int = 0,
                 thin: int = 1,
                 remove_low_numbers: bool = False) -> PosteriorChain:
    """
    Unified loader for posterior sample chains.

    Parameters
    ----------
    path : str
        Path to a chain file or directory of UCBMCMC outputs.
    num_params_per_source : int
        Number of parameters per source.
    fill_value : float, optional
        Padding for absent sources. Must be NaN (the default).
    burn : int, optional
        Number of initial samples to discard (default 0).
    thin : int, optional
        Interval for thinning; keep every `thin`-th sample (default 1).
    remove_low_numbers : bool, optional
        If True, skip UCBMCMC files with <= 8 samples (default False).

    Returns
    -------
    PosteriorChain
        Loaded and reshaped posterior samples.

    Raises
    ------
    ValueError
        If `num_params_per_source` is invalid, if `fill_value` is not NaN, if the number of columns in
        `path` matches neither supported layout, or if it matches both layouts
        and an explicit loader is required.

    Examples
    --------
    Load a fixed-num-sources chain, discarding the first five samples and
    keeping every second one::

        pc = load_samples('chains/fixed_chain.dat', num_params_per_source=4,
                          burn=5, thin=2)
        pc.chain.shape  # (ceil((n_samples - 5) / 2), n_sources, 4)
    """
    num_params_per_source = _validate_num_params_per_source(num_params_per_source)
    _validate_fill_value(fill_value)

    if os.path.isdir(path):
        return load_samples_ucbmcmc(path,
                                    num_params_per_source,
                                    fill_value=fill_value,
                                    burn=burn,
                                    thin=thin,
                                    remove_low_numbers=remove_low_numbers)

    # file case: only the first data row is read here; the delegate parses the file.
    n_cols = _count_data_columns(path)

    # Product-space files have 5 metadata columns and fixed chains have 4.
    # With one parameter per source, every sufficiently wide file satisfies
    # both arithmetic checks.  Its values cannot reliably disambiguate the
    # layouts because metadata and source values are both ordinary numbers.
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

    raise ValueError(
        f"Cannot infer loader for {path!r} with shape (n_cols={n_cols})"
    )


def load_samples_product_space(filepath: str,
                               num_params_per_source: int,
                               fill_value: float = np.nan,
                               burn: int = 0,
                               thin: int = 1) -> PosteriorChain:
    """
    Load a product-space chain from a text file.

    Parameters
    ----------
    filepath : str
        Path to the product-space chain file.
    num_params_per_source : int
        Number of parameters per source.
    fill_value : float, optional
        Padding for absent sources. Must be NaN (the default).
    burn : int, optional
        Number of initial samples to discard (default 0).
    thin : int, optional
        Interval for thinning; keep every `thin`-th sample (default 1).

    Returns
    -------
    PosteriorChain
        PosteriorChain with shape (n_kept_samples, max_num_sources, num_params_per_source)
        and `trans_dimensional=True`. `max_num_sources` is the largest number of
        sources among the samples that survive `burn` and `thin`.

    Raises
    ------
    ValueError
        If `num_params_per_source` is not a positive integer, if `fill_value`
        is not NaN, if no samples
        are left after `burn` and `thin`, or if the data columns or zero-based
        source counts do not describe a valid product-space layout.

    Examples
    --------
    Load a product-space chain, keeping every third sample after the first ten::

        pc = load_samples_product_space('chain.ps.txt', num_params_per_source=3,
                                        burn=10, thin=3)
        pc.chain.shape  # (n_kept, max_num_sources, 3)
    """
    num_params_per_source = _validate_num_params_per_source(num_params_per_source)
    _validate_fill_value(fill_value)
    with open(filepath, "r") as f:
        chain = np.loadtxt(f, ndmin=2)
    chain = chain[burn::thin]
    if chain.shape[0] == 0:
        raise ValueError(
            f"No samples left in {filepath!r} after burn={burn} and thin={thin}."
        )
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


def load_samples_fixed_num_sources(filepath: str,
                                   num_params_per_source: int,
                                   burn: int = 0,
                                   thin: int = 1) -> PosteriorChain:
    """
    Load a fixed-num-sources chain from a text file.

    Parameters
    ----------
    filepath : str
        Path to the fixed-num-sources chain file.
    num_params_per_source : int
        Number of parameters per source.
    burn : int, optional
        Number of initial samples to discard (default 0).
    thin : int, optional
        Interval for thinning; keep every `thin`-th sample (default 1).

    Returns
    -------
    PosteriorChain
        PosteriorChain with shape (n_kept_samples, num_sources, num_params_per_source).

    Raises
    ------
    ValueError
        If `num_params_per_source` is not a positive integer or data rows do
        not contain complete source vectors and four metadata columns.

    Examples
    --------
    Load a fixed-num-sources chain, discarding the first two samples and
    keeping every third one::

        pc = load_samples_fixed_num_sources('chain.txt', num_params_per_source=2,
                                            burn=2, thin=3)
        pc.chain.shape  # (ceil((n_samples - 2) / 3), num_sources, 2)
    """
    num_params_per_source = _validate_num_params_per_source(num_params_per_source)
    with open(filepath, "r") as f:
        chain = np.loadtxt(f, ndmin=2)
    chain = chain[burn::thin, :-4]  # remove metadata columns
    if chain.shape[1] == 0 or chain.shape[1] % num_params_per_source:
        raise ValueError(
            "Fixed-source data columns must contain complete source vectors "
            "followed by four metadata columns."
        )
    num_sources = (chain.shape[1]) // num_params_per_source
    samples = chain.reshape(chain.shape[0], num_sources, num_params_per_source)
    return PosteriorChain(samples, num_sources, num_params_per_source)


def load_samples_ucbmcmc(chain_folder: str,
                         num_params_per_source: int = 8,
                         fill_value: float = np.nan,
                         burn: int = 0,
                         thin: int = 1,
                         remove_low_numbers: bool = False) -> PosteriorChain:
    """
    Load multiple UCBMCMC chain files from a directory.

    Each ``dimension_chain.dat.<n>`` file holds the samples drawn with exactly
    `n` sources, written as `n` consecutive rows per sample. `burn` and `thin`
    are applied to every file individually, before the files are concatenated,
    so that each file's own burn-in is discarded.

    Parameters
    ----------
    chain_folder : str
        Directory containing `dimension_chain.dat.*` files.
    num_params_per_source : int, optional
        Number of parameters per source (default 8, the UCBMCMC model size).
    fill_value : float, optional
        Padding for absent sources. Must be NaN (the default).
    burn : int, optional
        Number of initial samples to discard from each file (default 0).
    thin : int, optional
        Interval for thinning; keep every `thin`-th sample of each file (default 1).
    remove_low_numbers : bool, optional
        If True, exclude files with <= 8 samples (default False).

    Returns
    -------
    PosteriorChain
        PosteriorChain with shape (n_total_samples, max_sources, num_params_per_source)
        and `trans_dimensional=True`.

    Raises
    ------
    ValueError
        If `num_params_per_source` is not a positive integer, if `fill_value`
        is not NaN, if no usable
        files are found, or if a file's contents are not a whole number of
        samples of `num_params_per_source` columns.

    Examples
    --------
    Load a directory of UCBMCMC chains, discarding the first five samples of
    each file and keeping every second one::

        pc = load_samples_ucbmcmc('ucbmcmc_chains/', burn=5, thin=2)
        pc.chain.shape  # (total_samples_after_burn_and_thin, max_sources, 8)
    """

    num_params_per_source = _validate_num_params_per_source(num_params_per_source)
    _validate_fill_value(fill_value)

    # Find and sort all matching filepaths.
    filepaths = sort_by_number(
        glob.glob(os.path.join(chain_folder, "dimension_chain.dat.*"))
    )

    # Load each file, apply burn/thin to it individually, and keep the blocks.
    blocks = []
    for filepath in filepaths:
        # Determine the number of sources from the filename.
        nsources = int(filepath.split(".")[-1])
        if nsources == 0:
            continue  # Skip files with no sources

        chain = np.loadtxt(filepath, ndmin=2)
        if chain.shape[1] != num_params_per_source:
            raise ValueError(
                f"{filepath!r} has {chain.shape[1]} columns, but "
                f"num_params_per_source={num_params_per_source} was requested."
            )
        # The number of samples comes from the parsed array, not from a raw line
        # count, so trailing newlines and comment lines cannot corrupt it.
        if chain.shape[0] % nsources != 0:
            raise ValueError(
                f"{filepath!r} has {chain.shape[0]} rows, which is not a multiple of the "
                f"{nsources} sources implied by its name."
            )
        nsamples = chain.shape[0] // nsources

        # Only accept files with more than 8 samples.
        if remove_low_numbers and nsamples <= 8:
            logger.debug("Skipping %s: only %d samples.", filepath, nsamples)
            continue

        block = chain.reshape(nsamples, nsources, num_params_per_source)[burn::thin]
        if block.shape[0] == 0:
            logger.debug("Skipping %s: no samples left after burn=%d, thin=%d.",
                         filepath, burn, thin)
            continue
        blocks.append((block, nsources))

    if not blocks:
        raise ValueError(
            f"No usable chain files found in {chain_folder!r} "
            f"(remove_low_numbers={remove_low_numbers}, burn={burn}, thin={thin})."
        )

    # Compute the total number of samples and maximum number of sources among valid files.
    total_samples = sum(block.shape[0] for block, _ in blocks)
    max_sources = max(nsources for _, nsources in blocks)

    # Initialize the array to hold the samples.
    samples = np.full((total_samples, max_sources, num_params_per_source), fill_value, dtype=float)

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
