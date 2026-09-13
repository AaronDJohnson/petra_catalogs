"""
Fit simple parametric distributions to each source slot of a posterior chain.

This is the first of the four stages of a relabeling pass
(:mod:`~petra.parametric_fits` -> :mod:`~petra.aux_distributions` ->
:mod:`~petra.cost_matrix` -> :mod:`~petra.relabel`).  A *fit function* here turns
a chain into the parameters of one auxiliary distribution per source label; the
matching evaluator in :mod:`petra.aux_distributions` turns those parameters back
into log-densities.

Conventions
-----------
* A chain is an ndarray of shape ``(n_samples, n_sources, n_params_per_source)``.
  Axis 1 is the *slot* (equivalently the current label) and axis 2 is the
  parameter vector of the source sitting in that slot.
* ``NaN`` means **absent**: the source was not in the model for that sample.  A
  slot is fitted from its non-NaN samples only, and a slot with too few of them
  falls back to a fit pooled over the whole chain, so that every label always has
  a proper distribution to be scored against.
* Every fit function has the signature ``(chain, max_num_sources) -> (a, b)``
  where ``a`` holds the location parameters and ``b`` the scale parameters, one
  entry per source label.  :func:`create_parametric_fit` adapts fit functions
  that need an extra fixed parameter index to that common signature.

Degeneracy
----------
Both fits are regularized, because an unregularized scale estimated from a
handful of samples is the package's main source of silent corruption.  A
rank-deficient covariance is *not* rejected by ``np.linalg.inv``, and a zero
standard deviation produces a NaN log-density that
:mod:`petra.cost_matrix` then replaces by a constant, leaving that label
indifferent between every slot.  :data:`COVARIANCE_RIDGE` sets the size of the
guard in both cases, relative to the scale of the data: it is *added* to a
fitted covariance (:func:`_regularize_covariance`) and used as a *floor* on a
fitted standard deviation (:func:`_regularize_std`), so that a healthy
univariate fit comes back unchanged.
"""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd

from petra.posterior_chain import PosteriorChain
from petra.utils import find_prob_in_model, get_logger

logger = get_logger(__name__)

#: Whatever a fit function hands to its paired auxiliary distribution.  The two
#: fits here return ``(locations, scales)``, but the seam is deliberately open:
#: :mod:`petra.bayesian_gaussian` passes a list of posterior-parameter dicts and
#: the flow modules pass a list of fitted distributions.
AuxParameters = Any

#: The ``(locations, scales)`` pair returned by the two normal fits in this
#: module.  For :func:`mv_normal_fit` that is ``(means, cov_matrices)`` with
#: shapes ``(n, d)`` and ``(n, d, d)``; for
#: :func:`uni_normal_fit_single_parameter` it is ``(means, stds)``, both of
#: shape ``(n,)``.
FitParameters = Tuple[np.ndarray, np.ndarray]

#: Signature every fit function is adapted to by :func:`create_parametric_fit`.
FitFunction = Callable[[np.ndarray, int], AuxParameters]

# Relative ridge added to every fitted covariance. Large enough to keep a Cholesky
# factorization well defined when a parameter is (near-)constant across the retained
# samples, small enough to leave the science untouched.
COVARIANCE_RIDGE = 1e-8

# Minimum retained samples for a univariate fit. Two would suffice for a variance, but a
# handful of points gives a std that is not pure noise.
MIN_UNIVARIATE_SAMPLES = 8


def _regularize_covariance(cov: np.ndarray,
                           fallback_variances: np.ndarray | None = None) -> np.ndarray:
    """
    Add a small relative ridge to a covariance matrix.

    An empirical covariance estimated from ``n <= d`` samples is rank deficient, and
    ``np.linalg.inv`` does not raise on such a matrix - it returns entries of order
    ``1/eps``, which then propagate into finite but meaningless cost-matrix values.
    Each diagonal ridge is scaled by that parameter's own variance. Averaging
    variances across parameters mixes their units: a broad coordinate can then
    overwhelm a narrow one and erase the information separating sources.

    Parameters
    ----------
    cov : ndarray, shape (d, d)
        Empirical covariance matrix.
    fallback_variances : ndarray, shape (d,), optional
        Pooled variance of each parameter, used when a source has no variance
        in that coordinate. This keeps constant sources sensitive to their
        separation in the units of the data.

    Returns
    -------
    cov : ndarray, shape (d, d)
        Covariance with ``COVARIANCE_RIDGE * diag(cov)`` added to the diagonal,
        equivalent to adding a ridge to the correlation matrix and transforming
        back to the original units. A parameter with no positive finite variance
        uses its pooled variance when supplied. Unit variance is the final
        fallback when neither covariance can establish that parameter's scale.
    """
    cov = np.asarray(cov, dtype=float)
    variances = np.diag(cov)
    fallback = np.ones_like(variances) if fallback_variances is None else np.asarray(fallback_variances)
    fallback = np.where(np.isfinite(fallback) & (fallback > 0.0), fallback, 1.0)
    scales = np.where(np.isfinite(variances) & (variances > 0.0), variances, fallback)
    return cov + COVARIANCE_RIDGE * np.diag(scales)


def _regularize_std(std: float, scale: float) -> float:
    """
    Floor a standard deviation with the univariate form of the covariance ridge.

    This is the one-dimensional sibling of :func:`_regularize_covariance`.  The ridge
    there raises a zero variance to ``COVARIANCE_RIDGE * scale``, i.e. a standard
    deviation to ``sqrt(COVARIANCE_RIDGE) * sqrt(scale)``; the floor applied here is
    the same fraction of the spread of the data.  A slot whose retained samples are
    all identical therefore gets a narrow-but-finite normal instead of a degenerate
    one.  Without it,
    :func:`petra.aux_distributions.uni_normal_aux_distribution_single_parameter`
    divides by zero, the whole row of the cost matrix becomes NaN, and
    :mod:`petra.cost_matrix` replaces it by a single constant -- a row-constant cost
    row is invariant under ``scipy.optimize.linear_sum_assignment``, so that label is
    then assigned arbitrarily and silently.

    A floor rather than an additive ridge, so that a healthy fit is returned bit for
    bit unchanged: the pooled `scale` can be orders of magnitude larger than the
    spread of one well-localized source, and adding to that source's variance would
    perturb it far more than the constant suggests.

    Parameters
    ----------
    std : float
        Empirical standard deviation, possibly zero or NaN.
    scale : float
        Characteristic spread of the parameter, used to make the floor invariant to
        the units of the parameter.  Non-finite or non-positive values are replaced
        by ``1.0``.

    Returns
    -------
    std : float
        ``max(std, sqrt(COVARIANCE_RIDGE) * scale)``, always strictly positive.

    Examples
    --------
    >>> round(_regularize_std(0.0, 2.0), 8)   # floor is 1e-4 of the scale
    0.0002
    >>> _regularize_std(1.0, 1.0)             # a healthy std is untouched exactly
    1.0
    """
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    if not np.isfinite(std):
        std = 0.0
    return float(max(std, np.sqrt(COVARIANCE_RIDGE) * scale))


def create_parametric_fit(fit_function: Callable[..., AuxParameters],
                          single_parameter: Optional[int] = None) -> FitFunction:
    """
    Wrap a fitting function in a unified interface.

    Parameters
    ----------
    fit_function : callable
        A function with signature
        ``(chain, max_num_sources[, fit_parameter])`` that returns fit parameters.
    single_parameter : int, optional
        If provided, fixes the parameter index for single-parameter fit functions.

    Returns
    -------
    parametric_fit : callable
        A function with signature ``(chain, max_num_sources)``
        that calls `fit_function` and returns its output.

    Examples
    --------
    >>> from petra.parametric_fits import create_parametric_fit, uni_normal_fit_single_parameter
    >>> import numpy as np
    >>> fit = create_parametric_fit(uni_normal_fit_single_parameter, single_parameter=2)
    >>> chain = np.random.randn(100, 3, 5)
    >>> means, stds = fit(chain, max_num_sources=3)
    >>> means.shape, stds.shape
    ((3,), (3,))
    """
    if single_parameter is not None:
        fit_function = partial(fit_function, fit_parameter=single_parameter)

    def parametric_fit(chain: np.ndarray, max_num_sources: int) -> AuxParameters:
        """
        Fit a parametric distribution to the chain of samples.

        Parameters
        ----------
        chain : ndarray, shape (n_samples, n_sources, n_params_per_source)
            Posterior samples; NaN marks an absent source.
        max_num_sources : int
            Number of source labels to fit.

        Returns
        -------
        fit : tuple of ndarray
            The ``(locations, scales)`` returned by the wrapped fit function.
        """
        return fit_function(chain, max_num_sources)

    return parametric_fit


def mv_normal_fit(chain: np.ndarray, max_num_sources: int) -> FitParameters:
    """
    Fit a multivariate normal distribution to each entry.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Posterior samples.
    max_num_sources : int
        Number of sources to fit.

    Returns
    -------
    means : ndarray, shape (max_num_sources, n_params)
        Mean vectors for each source.
    cov_matrices : ndarray, shape (max_num_sources, n_params, n_params)
        Covariance matrices for each source, each regularized by
        :func:`_regularize_covariance`.

    Examples
    --------
    >>> from petra.parametric_fits import mv_normal_fit
    >>> import numpy as np
    >>> chain = np.random.randn(500, 4, 2)
    >>> means, covs = mv_normal_fit(chain, max_num_sources=4)
    >>> means.shape, covs.shape
    ((4, 2), (4, 2, 2))
    """
    num_params = chain.shape[2]
    # A d-dimensional covariance needs more than d samples to have full rank; require a
    # couple to spare. The old threshold was a bare 8, independent of dimension, so an
    # 8-parameter source with exactly 8 retained samples passed the guard and produced a
    # rank-deficient covariance.
    min_samples = num_params + 2

    means = []
    cov_matrices = []
    pooled_variances = None
    for source in range(max_num_sources):
        sample_i = chain[:, source, :]  # shape: (num_samples, num_params)
        valid = ~np.isnan(sample_i).any(axis=1)
        valid_samples = sample_i[valid]
        if valid_samples.shape[0] < min_samples:
            logger.info('Fewer than %d values in source index %d (%d found, %d '
                        'parameters). Appending normal distribution fit to all entries.',
                        min_samples, source, valid_samples.shape[0], num_params)
            df_all = pd.DataFrame(chain.reshape(-1, num_params)).dropna()
            means.append(np.array(df_all.mean()))
            cov_matrices.append(_regularize_covariance(np.array(df_all.cov())))
            continue
        df = pd.DataFrame(valid_samples)
        means.append(np.array(df.mean()))
        covariance = np.array(df.cov())
        if pooled_variances is None and np.any(np.diag(covariance) <= 0):
            # Infer the scale from the same coordinate across all sources. A
            # constant source can still be separated from other sources by its
            # mean; an absolute ridge would erase that separation in small units.
            pooled = pd.DataFrame(chain.reshape(-1, num_params)).dropna()
            pooled_variances = np.array(pooled.var())
        cov_matrices.append(_regularize_covariance(covariance, pooled_variances))
    return np.array(means), np.array(cov_matrices)


def uni_normal_fit_single_parameter(chain: np.ndarray, max_num_sources: int,
                                    fit_parameter: int) -> FitParameters:
    """
    Fit a univariate normal distribution to one parameter for each source.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Posterior samples.
    max_num_sources : int
        Number of sources to fit.
    fit_parameter : int
        Index of the parameter to fit.

    Returns
    -------
    means : ndarray, shape (max_num_sources,)
        Means for each source.
    stds : ndarray, shape (max_num_sources,)
        Standard deviations for each source.  Strictly positive: a slot whose
        retained samples are constant is floored by :func:`_regularize_std`
        rather than returned as zero.

    Notes
    -----
    The ridge is scaled by the spread of the *same* parameter pooled over the whole
    chain, since a degenerate slot has no spread of its own to scale it by.

    Examples
    --------
    >>> from petra.parametric_fits import uni_normal_fit_single_parameter
    >>> chain = np.random.randn(200, 3, 5)
    >>> means, stds = uni_normal_fit_single_parameter(chain, max_num_sources=3, fit_parameter=1)
    >>> means.shape, stds.shape
    ((3,), (3,))

    A slot that never varies still gets a usable (narrow, but finite) normal:

    >>> chain = np.zeros((50, 2, 1))
    >>> chain[:, 0, 0] = np.linspace(0.0, 1.0, 50)
    >>> chain[:, 1, 0] = 3.0                       # constant across every sample
    >>> means, stds = uni_normal_fit_single_parameter(chain, 2, 0)
    >>> bool(np.all(stds > 0.0))
    True
    """
    # Pool the SAME parameter across every source and sample. Flattening the whole chain
    # instead would mix unrelated parameter dimensions (frequency, amplitude,
    # inclination, ...) into one meaningless marginal.
    pooled = chain[:, :, fit_parameter].reshape(-1)
    pooled = pooled[~np.isnan(pooled)]
    # Characteristic spread of this parameter, used only to make the ridge below
    # invariant to the parameter's units.
    pooled_scale = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0

    means = []
    stds = []
    for source in range(max_num_sources):
        sample_i = chain[:, source, fit_parameter]  # shape: (num_samples,)
        valid_samples = sample_i[~np.isnan(sample_i)]
        if valid_samples.shape[0] < MIN_UNIVARIATE_SAMPLES:
            logger.info('Fewer than %d values in source index %d. Appending normal '
                        'distribution fit to all entries.',
                        MIN_UNIVARIATE_SAMPLES, source)
            means.append(float(np.mean(pooled)))
            raw_std = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0
        else:
            # ddof=1 throughout, to match the sample covariance used by `mv_normal_fit`.
            means.append(float(np.mean(valid_samples)))
            raw_std = float(np.std(valid_samples, ddof=1))
        if not raw_std > 0.0:
            logger.warning(
                'Source index %d has a degenerate spread in parameter %d '
                '(std=%r from %d retained samples); flooring it so the label keeps a '
                'proper distribution. A zero std would make its whole cost-matrix row '
                'NaN, and the label would then be assigned arbitrarily.',
                source, fit_parameter, raw_std, valid_samples.shape[0],
            )
        stds.append(_regularize_std(raw_std, pooled_scale))

    return np.array(means), np.array(stds)


def update_parametric_fit_and_prob_in_model(
    posterior_chain: PosteriorChain,
    max_num_sources: int,
    parametric_fit_function: Callable[[np.ndarray, int], AuxParameters],
    eps: float = 1e-2,
) -> Tuple[AuxParameters, np.ndarray]:
    """
    Compute parametric fit and inclusion probabilities.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Object with a ``.get_chain()`` method returning an ndarray.
    max_num_sources : int
        Number of sources to consider.
    parametric_fit_function : callable
        Function ``(chain, max_num_sources) -> aux_params``.
    eps : float, optional
        Clip bound keeping every inclusion probability inside ``[eps, 1 - eps]``,
        so that ``log(p)`` and ``log(1 - p)`` stay finite (default 1e-2).

    Returns
    -------
    aux_params : tuple
        Output of `parametric_fit_function`, e.g. ``(means, covs)`` or
        ``(means, stds)``.
    prob_in_model : ndarray, shape (max_num_sources,)
        Inclusion probabilities.

    Examples
    --------
    >>> from petra.parametric_fits import update_parametric_fit_and_prob_in_model, mv_normal_fit
    >>> from petra.posterior_chain import PosteriorChain
    >>> pc = PosteriorChain(np.random.randn(150, 4, 2), 4, 2, True, None, {})
    >>> aux, probs = update_parametric_fit_and_prob_in_model(pc, 4, mv_normal_fit)
    >>> aux[0].shape, probs.shape
    ((4, 2), (4,))
    """
    aux_params = parametric_fit_function(posterior_chain.get_chain(), max_num_sources)
    prob_in_model = find_prob_in_model(posterior_chain.get_chain(), max_num_sources, eps=eps)
    return aux_params, prob_in_model
