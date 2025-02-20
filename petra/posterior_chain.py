import numpy as np
from dataclasses import dataclass


@dataclass
class PosteriorChain:
    """
    A dataclass to store the chains, number of sources, and number of parameters per source

    Parameters
    ----------
    chain: np.ndarray
        The chain of samples
    num_sources: int
        The number of sources
    num_params_per_source: int
        The number of parameters per source
    trans_dimensional: bool
        Whether the chain has variable number of sources
    prob_in_model: ndarray
        Probability of a source being in the model
    cost_dict: dict
        A dictionary of total cost for the labeling
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
        return self.chain[burn::thin]

    def expand_chain(self, max_num_sources):
        """
        Expand the second dimension of the chain to the maximum number of sources.
        """
        expanded_chain = np.zeros((self.chain.shape[0], max_num_sources, self.num_params_per_source)) + np.nan
        expanded_chain[:, :self.num_sources, :] = self.chain
        return PosteriorChain(expanded_chain, max_num_sources, self.num_params_per_source, True, self.prob_in_model, self.cost_dict)
    
    def randomize_entries(self, seed=None):
        """
        Randomize the entries (second dimension) of the chain without repetition.
        """
        # Shuffle along axis=1 (the second dimension) for each index in the first dimension
        if seed is not None:
            np.random.seed(seed)
        for i in range(self.chain.shape[0]):
            np.random.shuffle(self.chain[i])
