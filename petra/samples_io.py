import numpy as np
import os
import glob
from petra.utils import sort_by_number
from petra.posterior_chain import PosteriorChain


def load_samples(path,
                 num_params_per_source,
                 fill_value=np.nan,
                 burn=0,
                 thin=1,
                 remove_low_numbers=False):
    """
    Unified loader for posterior sample chains.

    Parameters
    ----------
    path : str
        Path to a chain file or directory of UCBMCMC outputs.
    num_params_per_source : int
        Number of parameters per source.
    fill_value : scalar, optional
        Value to use for padding missing entries (default np.nan).
    burn : int, optional
        Number of initial samples to discard (default 0).
    thin : int, optional
        Interval for thinning; keep every `thin`-th sample (default 1).
    remove_low_numbers : bool, optional
        If True, skip UCBMCMC files with ≤ 8 samples (default False).

    Returns
    -------
    PosteriorChain
        Loaded and reshaped posterior samples.

    Examples
    --------
    >>> from petra.samples_io import load_samples
    >>> pc = load_samples('chains/fixed_chain.dat', num_params_per_source=4, burn=5, thin=2)
    >>> pc.chain.shape
    (ceil((n_samples-5)/2), n_sources, 4)
    """
    if os.path.isdir(path):
        return load_samples_ucbmcmc(path,
                                    fill_value=fill_value,
                                    burn=burn,
                                    thin=thin,
                                    remove_low_numbers=remove_low_numbers)

    # file case
    chain = np.loadtxt(path)
    n_cols = chain.shape[1]

    # product‐space files have 5 metadata cols at end
    if n_cols > 5 and (n_cols - 5) % num_params_per_source == 0:
        return load_samples_product_space(path,
                                          num_params_per_source,
                                          fill_value=fill_value)

    # fixed‐num‐sources files have 4 metadata cols at end
    if n_cols > 4 and (n_cols - 4) % num_params_per_source == 0:
        return load_samples_fixed_num_sources(path,
                                              num_params_per_source,
                                              burn=burn,
                                              thin=thin)

    raise ValueError(
        f"Cannot infer loader for {path!r} with shape (n_cols={n_cols})"
    )


def load_samples_product_space(filepath, num_params_per_source, fill_value=np.nan):
    """
    Load a product-space chain from a text file.

    Parameters
    ----------
    filepath : str
        Path to the product-space chain file.
    num_params_per_source : int
        Number of parameters per source.
    fill_value : scalar, optional
        Value to use for padding missing entries (default np.nan).

    Returns
    -------
    PosteriorChain
        PosteriorChain with shape (n_samples, max_num_sources, num_params_per_source)
        and `trans_dimensional=True`.

    Examples
    --------
    >>> from petra.samples_io import load_samples_product_space
    >>> pc = load_samples_product_space('chain.ps.txt', num_params_per_source=3)
    >>> pc.chain.shape
    (n_samples, max_num_sources, 3)
    """
    with open(filepath, "r") as f:
        chain = np.loadtxt(f)
    num_sources_vector = np.rint(chain[:, -5] + 1).astype(int)
    num_sources = np.max(num_sources_vector)

    only_samples_chain = (
        np.zeros((chain.shape[0], num_sources * num_params_per_source)) + fill_value
    )
    for i in range(chain.shape[0]):
        only_samples_chain[i, : num_sources_vector[i] * num_params_per_source] = chain[
            i, : num_sources_vector[i] * num_params_per_source
        ]
    samples = only_samples_chain.reshape(-1, num_sources, num_params_per_source)
    return PosteriorChain(
        samples, num_sources, num_params_per_source, trans_dimensional=True
    )


def load_samples_fixed_num_sources(filepath, num_params_per_source, burn=0, thin=1):
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
        PosteriorChain with shape (n_samples, num_sources, num_params_per_source).

    Examples
    --------
    >>> from petra.samples_io import load_samples_fixed_num_sources
    >>> pc = load_samples_fixed_num_sources('chain.txt', num_params_per_source=2, burn=2, thin=3)
    >>> pc.chain.shape
    (ceil((n_samples-2)/3), num_sources, 2)
    """
    with open(filepath, "r") as f:
        chain = np.loadtxt(f)
    chain = chain[burn::thin, :-4]  # remove metadata columns
    num_sources = (chain.shape[1]) // num_params_per_source
    samples = chain.reshape(-1, num_sources, num_params_per_source)
    return PosteriorChain(samples, num_sources, num_params_per_source)


def load_samples_ucbmcmc(chain_folder, fill_value=np.nan, burn=0, thin=1, remove_low_numbers=False):
    """
    Load multiple UCBMCMC chain files from a directory.

    Parameters
    ----------
    chain_folder : str
        Directory containing `dimension_chain.dat.*` files.
    fill_value : scalar, optional
        Value to use for padding missing entries (default np.nan).
    burn : int, optional
        Number of initial samples to discard after loading (default 0).
    thin : int, optional
        Interval for thinning; keep every `thin`-th sample (default 1).
    remove_low_numbers : bool, optional
        If True, exclude files with ≤ 8 samples (default False).

    Returns
    -------
    PosteriorChain
        PosteriorChain with shape (n_total_samples, max_sources, num_params_per_source)
        and `trans_dimensional=True`.

    Examples
    --------
    >>> from petra.samples_io import load_samples_ucbmcmc
    >>> pc = load_samples_ucbmcmc('ucbmcmc_chains/', burn=5, thin=2)
    >>> pc.chain.shape
    (total_samples_after_burn_and_thin, max_sources, num_params_per_source)
    """

    # Find and sort all matching filepaths.
    filepaths = sort_by_number(
        glob.glob(os.path.join(chain_folder, "dimension_chain.dat.*"))
    )

    # Build a list of valid files that have more samples than the model's 8 dimensions.
    valid_files = []
    for filepath in filepaths:
        # Determine the number of sources from the filename.
        nsources = int(filepath.split(".")[-1])
        if nsources == 0:
            continue  # Skip files with no sources

        # Count the total number of lines in the file.
        with open(filepath, "r") as f:
            total_lines = sum(1 for line in f)
        # Calculate the number of samples by dividing by the number of sources.
        # (Assumes that the file's total line count is an integer multiple of nsources.)
        nsamples = total_lines // nsources

        # Only accept files with more than 8 samples.
        if not remove_low_numbers:
            valid_files.append((filepath, nsamples, nsources))
        elif remove_low_numbers and nsamples > 8:
            valid_files.append((filepath, nsamples, nsources))

    if not valid_files:
        raise ValueError("No files with more than 8 samples found.")

    # Compute the total number of samples and maximum number of sources among valid files.
    total_samples = sum(nsamples for (_, nsamples, _) in valid_files)
    max_sources = max(nsources for (_, _, nsources) in valid_files)

    # Initialize the array to hold the samples.
    samples = np.full((total_samples, max_sources, 8), fill_value)

    current_total = 0
    # Loop over valid files and load their data.
    for filepath, nsamples, nsources in valid_files:
        # Load the chain data from the file.
        chain = np.loadtxt(filepath)
        # Reshape and place the data into the samples array.
        samples[current_total: current_total + nsamples, :nsources, :] = chain.reshape((nsamples, nsources, 8))
        current_total += nsamples

    return PosteriorChain(
        samples[burn::thin],
        max_sources,
        trans_dimensional=True,
        num_params_per_source=8,
    )
