import numpy as np
from dataclasses import dataclass
import pandas as pd


@dataclass
class PosteriorChain:
    """
    A dataclass to store the chains, number of sources, and number of parameters per source.

    Parameters
    ----------
    chain : ndarray, shape (num_samples, num_sources, num_params_per_source)
        The chain of samples.
    num_sources : int
        The number of sources.
    num_params_per_source : int
        The number of parameters per source.
    trans_dimensional : bool, optional
        Whether the chain has variable number of sources. Default is False.
    prob_in_model : ndarray, optional
        Probability of a source being in the model.
    cost_dict : dict, optional
        A dictionary of total cost for the labeling.

    Attributes
    ----------
    chain : ndarray
        Reshaped chain after initialization.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.posterior_chain import PosteriorChain
    >>> arr = np.random.randn(50, 2, 4)
    >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=4)
    >>> pc.chain.shape
    (50, 2, 4)
    """

    chain: np.ndarray
    num_sources: int
    num_params_per_source: int
    trans_dimensional: bool = False
    prob_in_model: np.ndarray = None
    cost_dict: dict = None

    def __post_init__(self):
        if self.cost_dict is None:
            self.cost_dict = {}
        self.chain = self.chain.reshape(-1, self.num_sources, self.num_params_per_source)

    def __repr__(self):
        return self.chain.view().__repr__()

    @property
    def shape(self):
        return self.chain.shape

    def __getitem__(self, index):
        return self.chain[index]

    def __setitem__(self, index, value):
        self.chain[index] = value

    def get_chain(self, burn=0, thin=1):
        """
        Retrieve a sub-chain by discarding initial samples and thinning.

        Parameters
        ----------
        burn : int, default 0
            Number of initial samples to discard.
        thin : int, default 1
            Keep every `thin`-th sample.

        Returns
        -------
        ndarray
            Array of shape (ceil((n_samples - burn) / thin), n_sources, n_params_per_source).

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.arange(60).reshape(20, 3, 1)
        >>> pc = PosteriorChain(arr, num_sources=3, num_params_per_source=1)
        >>> pc.get_chain(burn=5, thin=2).shape
        (8, 3, 1)
        """
        return self.chain[burn::thin]

    def expand_chain(self, max_num_sources):
        """
        Expand the chain to a larger number of sources, padding with NaNs.

        Parameters
        ----------
        max_num_sources : int
            Desired number of sources after expansion.

        Returns
        -------
        PosteriorChain
            New instance with `chain.shape == (n_samples, max_num_sources, n_params_per_source)`.

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.random.randn(10, 2, 3)
        >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=3)
        >>> pc2 = pc.expand_chain(4)
        >>> pc2.chain.shape
        (10, 4, 3)
        """
        expanded_chain = np.zeros((self.chain.shape[0], max_num_sources, self.num_params_per_source)) + np.nan
        expanded_chain[:, :self.num_sources, :] = self.chain
        return PosteriorChain(expanded_chain, max_num_sources, self.num_params_per_source, True, self.prob_in_model, self.cost_dict)

    def randomize_entries(self, seed=None):
        """
        Shuffle the entries (second dimension) of each sample without repetition.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible shuffling.

        Returns
        -------
        PosteriorChain
            New instance with entries randomized per sample.

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.arange(12).reshape(3, 2, 2)
        >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=2)
        >>> pc2 = pc.randomize_entries(seed=0)
        >>> pc2.chain.shape
        (3, 2, 2)
        """
        # Shuffle along axis=1 (the second dimension) for each index in the first dimension
        rng = np.random.default_rng(seed=seed)

        # Make a copy of the chain to avoid modifying the original
        chain = self.chain.copy()

        for i in range(self.chain.shape[0]):
            rng.shuffle(chain[i])
        return PosteriorChain(chain, self.num_sources, self.num_params_per_source, self.trans_dimensional, self.prob_in_model, self.cost_dict)

    def to_feather(self, feather_filepath):
        """
        Save the PosteriorChain to a Feather file including chain data and metadata.

        Parameters
        ----------
        feather_filepath : str
            Path to the output Feather file.

        Returns
        -------
        None

        Examples
        --------
        >>> import numpy as np
        >>> from petra.posterior_chain import PosteriorChain
        >>> arr = np.random.randn(5, 2, 3)
        >>> pc = PosteriorChain(arr, num_sources=2, num_params_per_source=3)
        >>> pc.to_feather('test_pc.feather')
        >>> import os
        >>> os.path.exists('test_pc.feather')
        True
        """
        df = pd.DataFrame(self.chain.reshape(-1, self.chain.shape[1] * self.chain.shape[2]))
        df['num_sources'] = self.chain.shape[1]
        df['num_params_per_source'] = self.chain.shape[2]
        df['transdimensional'] = int(self.trans_dimensional)
        df.to_feather(feather_filepath)

    @staticmethod
    def read_feather(feather_filepath):
        """
        Load a PosteriorChain from a Feather file with stored chain data and metadata.

        Parameters
        ----------
        feather_filepath : str
            Path to the Feather file produced by `to_feather`.

        Returns
        -------
        PosteriorChain
            A new PosteriorChain instance reconstructed from the file.

        Examples
        --------
        >>> from petra.posterior_chain import PosteriorChain
        >>> # assume 'test_pc.feather' exists from to_feather()
        >>> pc2 = PosteriorChain.read_feather('test_pc.feather')
        >>> isinstance(pc2, PosteriorChain)
        True
        >>> pc2.chain.shape
        (5, 2, 3)
        """
        df = pd.read_feather(feather_filepath)
        num_sources = df['num_sources'][0]
        num_params_per_source = df['num_params_per_source'][0]
        transdimensional = bool(df['transdimensional'][0])
        # remove these three parameters from the DataFrame
        df.drop(columns=['num_sources', 'num_params_per_source', 'transdimensional'], inplace=True)
        # reshape the DataFrame to the original shape
        chain = df.values.reshape(-1, num_sources, num_params_per_source)
        return PosteriorChain(chain, num_sources, num_params_per_source, transdimensional)
    