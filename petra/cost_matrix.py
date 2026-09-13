"""
Build the per-sample assignment matrix that drives relabeling.

Third of the four stages (:mod:`~petra.parametric_fits` ->
:mod:`~petra.aux_distributions` -> :mod:`~petra.cost_matrix` ->
:mod:`~petra.relabel`).  Given one posterior sample and the auxiliary
distributions fitted to the current labeling, this module scores every
(label, slot) pairing; :mod:`petra.relabel` then hands the result to
``scipy.optimize.linear_sum_assignment``.

Conventions
-----------
* The matrix has shape ``(num_distributions, n_sources)`` and is indexed
  ``[label, slot]``: **rows are labels** (fitted distributions), **columns are
  slots** (positions in the sample).  Getting that orientation wrong is not
  loud -- it produces a plausible matrix with the spike term applied along the
  wrong axis.
* Entries are **rewards, not costs**, despite the name: entry ``(i, j)`` is
  ``log(p_i) + log f_i(x_j)``, the log joint probability that label *i* is in
  the model *and* generated the data in slot *j*.  The assignment is therefore
  **maximized**, and larger is better throughout.
* The model is spike-and-slab.  A slot where the source is absent is detected
  from the original sample and its entry is replaced by ``log(1 - p_i)``: the
  probability that label *i* is out of the model, which is the correct
  alternative hypothesis for a missing source.  It depends on the *row* (the
  label), never on the column, precisely so that the column does not become
  row-constant -- a row-constant column drops out of the assignment objective
  entirely and the spike half of the model would be inert.
* ``num_distributions`` need not equal ``n_sources``; the matrix is rectangular
  whenever the catalog has more slots than fitted labels.
"""

import numpy as np
from typing import Callable, List
from functools import partial

from petra.utils import source_present


def create_compute_cost_matrix(aux_distribution: Callable, single_parameter: int | None = None) -> Callable:
    """
    Create a cost-matrix computation function for a given auxiliary distribution.

    Parameters
    ----------
    aux_distribution : callable
        Function to compute log-pdf values for the auxiliary distribution.
    single_parameter : int, optional
        Fix an index if `aux_distribution` is single-parameter.

    Returns
    -------
    compute_cost_matrix : callable
        A function with signature
            (sample, aux_parameters, prob_in_model, num_distributions) -> cost_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from petra.aux_distributions import mv_normal_aux_distribution
    >>> from petra.cost_matrix import create_compute_cost_matrix
    >>> compute_cost = create_compute_cost_matrix(mv_normal_aux_distribution)
    >>> # sample: 4 sources x 3 params each
    >>> sample = np.random.randn(4, 3)
    >>> means = [np.zeros(3), np.ones(3)]
    >>> covs = [np.eye(3), 2*np.eye(3)]
    >>> prob = np.array([0.6, 0.4])
    >>> cost = compute_cost(sample, (means, covs), prob, num_distributions=2)
    >>> cost.shape
    (2, 4)
    """
    if single_parameter is not None:
        aux_distribution = partial(aux_distribution, single_parameter=single_parameter)

    def compute_cost_matrix(sample: np.ndarray, aux_parameters: List[np.ndarray],
                            prob_in_model: np.ndarray, num_distributions: int) -> np.ndarray:
        """
        Compute the cost matrix for one sample and a set of distributions.

        Parameters
        ----------
        sample : ndarray, shape (n_sources, n_params_per_source)
            A single sample of parameter values for each source.
        aux_parameters : tuple of lists of ndarray
            Auxiliary distribution parameters,
            e.g., (means, cov_matrices) for multivariate normals.
        prob_in_model : ndarray, shape (num_distributions,)
            Probability that each distribution is active.
        num_distributions : int
            Number of distributions to include.

        Returns
        -------
        cost_matrix : ndarray, shape (num_distributions, n_sources)
            Cost where entry (i, j) = log(prob_in_model[i]) + logpdf_i(sample[j]).

        Examples
        --------
        >>> from petra.aux_distributions import mv_normal_aux_distribution
        >>> compute_cost = create_compute_cost_matrix(mv_normal_aux_distribution)
        >>> sample = np.random.randn(5, 4)
        >>> means = [np.zeros(4), np.ones(4)]
        >>> covs = [np.eye(4), 1.5*np.eye(4)]
        >>> prob = np.array([0.7, 0.3])
        >>> cost = compute_cost(sample, (means, covs), prob, num_distributions=2)
        >>> cost.shape
        (2, 5)
        """
        # Precompute logarithms.
        with np.errstate(divide='ignore'):  # avoid divide by zero warnings, they are expected when we don't clip prob_in_model
            log_prob = np.log(prob_in_model)  # shape: (num_distributions,)
            # Indexed by distribution, i.e. along the ROW axis, to match `log_prob` below.
            # Broadcasting it along the column axis instead would fill an absent slot with
            # log(1 - p_j) of the *slot* rather than log(1 - p_i) of the *label*, which makes
            # the column row-constant and drops it out of the assignment objective entirely.
            log_prob_not = np.log1p(-prob_in_model)[:, np.newaxis]  # shape: (num_distributions, 1)

        # Vectorize over distribution indices.
        distribution_indices = np.arange(num_distributions)
        # Assume that aux_distribution is vectorized so that it returns an array of shape (num_distributions, num_sources)
        cost_matrix = aux_distribution(sample, aux_parameters, distribution_indices)
        cost_matrix = log_prob[:, np.newaxis] + cost_matrix

        # Absence is a property of the original sample, not of the evaluated
        # density.  Flow evaluators deliberately replace non-finite outputs by
        # a finite floor, so looking for NaNs here would lose the missing-source
        # sentinel.  Conversely, a present point whose flow returns NaN should
        # keep that finite floor rather than be mistaken for an absent source.
        present_slots = source_present(sample)[np.newaxis, :]
        cost_matrix = np.where(present_slots, cost_matrix, log_prob_not)

        return cost_matrix

    return compute_cost_matrix
