import numpy as np
import os
import glob
from petra.utils import sort_by_number
from petra.posterior_chain import PosteriorChain


def load_samples_product_space(filepath, num_params_per_source, fill_value=np.nan):
    """
    Load chains from a file and reshape them into a 3D array
    with dimensions (num_samples, num_sources, num_params)

    Assumes the chain is organized as (param1_source1, param2_source1, ..., paramN_source1, paramN_source2, ...)
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
    Load chains from a file and reshape them into a 3D array
    with dimensions (num_samples, num_sources, num_params)

    Assumes the chain is organized as (param1_source1, param2_source1, ..., paramN_source1, paramN_source2, ...)
    """
    with open(filepath, "r") as f:
        chain = np.loadtxt(f)
    chain = chain[burn::thin, :-4]  # remove metadata columns
    num_sources = (chain.shape[1]) // num_params_per_source
    samples = chain.reshape(-1, num_sources, num_params_per_source)
    return PosteriorChain(samples, num_sources, num_params_per_source)


def load_samples_ucbmcmc(chain_folder, fill_value=np.nan, burn=0, thin=1, remove_low_numbers=False):
    """
    Load chains from files and reshape them into a 3D array
    with dimensions (num_samples, num_sources, num_params).

    Only files with more than 8 samples (i.e. more samples than the number
    of dimensions in the model) are loaded.

    Assumes the chain is organized as:
      (param1_source1, param2_source1, ..., 
       param1_source2, param2_source2, ..., 
       ...,
       paramN_source1, paramN_source2, ...)
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
