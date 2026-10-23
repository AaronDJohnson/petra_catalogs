"""Per-slot normal fits, their log-densities and the per-sample reward matrix of a relabeling pass.

A fit maps a chain (N, K, P), absent rows all-NaN, to opaque parameters that a log-prob scores sample rows against;
`compute_cost_matrix` turns both into the (K, S) rewards that ``linear_sum_assignment`` maximizes.
"""

import functools
import logging
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from petra.utils import source_present

logger = logging.getLogger(__name__)

AuxParameters = Any

COVARIANCE_RIDGE = 1e-8
MIN_UNIVARIATE_SAMPLES = 8
# 64 ulps of a parameter's largest magnitude: far below any posterior width, above a frozen column's round-off.
SPREAD_RTOL = 64 * float(np.finfo(np.float64).eps)


class _PooledFit(NamedTuple):
    """Moments pooled over every slot: fallback mean, regularized covariance, per-parameter variance (0 if constant)."""

    mean: np.ndarray
    covariance: np.ndarray
    variances: np.ndarray


def _sample_moments(samples: np.ndarray,
                    frame: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the pandas mean and covariance of NaN-free rows, exact for degenerate parameters, and that mask."""
    num_params = samples.shape[1]
    if samples.shape[0] == 0:
        return (np.full(num_params, np.nan), np.full((num_params, num_params), np.nan),
                np.zeros(num_params, dtype=bool))
    frame = pd.DataFrame(samples) if frame is None else frame
    mean = np.array(frame.mean())
    covariance = np.array(frame.cov())
    # A column frozen at 0.3 has a round-off variance near 1e-30, not 0; anything that small counts as exactly 0,
    # so it gets the pooled-variance ridge instead of a spike of round-off width.
    constant = np.ptp(samples, axis=0) == 0
    tolerance = (SPREAD_RTOL * np.max(np.abs(samples), axis=0)) ** 2
    degenerate = constant | ~(np.diag(covariance) > tolerance)
    if degenerate.any():
        mean = np.where(constant, samples[0], mean)
        covariance[degenerate, :] = 0.0
        covariance[:, degenerate] = 0.0
    return mean, covariance, degenerate


def _pool(chain: np.ndarray) -> _PooledFit:
    """Fit one normal to the present rows of every slot of `chain`."""
    frame = pd.DataFrame(chain.reshape(-1, chain.shape[2])).dropna()
    mean, covariance, degenerate = _sample_moments(frame.to_numpy(), frame)
    variances = np.where(degenerate, 0.0, np.array(frame.var()))
    return _PooledFit(mean, regularize_covariance(covariance), variances)


def regularize_covariance(cov: np.ndarray, fallback_variances: np.ndarray | None = None) -> np.ndarray:
    """Add ``COVARIANCE_RIDGE`` times each parameter's own variance to the diagonal of a covariance.

    Parameters:
        cov: (d, d) covariance.
        fallback_variances: (d,) scale for a parameter whose variance is not positive and finite; else unit scale.

    Returns:
        (d, d) float64 covariance with the ridge added.
    """
    cov = np.asarray(cov, dtype=float)
    variances = np.diag(cov)
    fallback = np.ones_like(variances) if fallback_variances is None else np.asarray(fallback_variances)
    fallback = np.where(np.isfinite(fallback) & (fallback > 0.0), fallback, 1.0)
    scales = np.where(np.isfinite(variances) & (variances > 0.0), variances, fallback)
    return cov + COVARIANCE_RIDGE * np.diag(scales)


def _fit_slot(samples: np.ndarray, source: int, pool: Callable[[], _PooledFit],
              num_samples: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit one slot of `mv_normal_fit` from its present rows, or return None so it takes the pooled fit."""
    num_rows, num_params = samples.shape
    # A d-parameter covariance needs d + 1 points in general position for full rank; require one to spare.
    min_samples = num_params + 2
    if num_rows < min_samples:
        logger.info('Fewer than %d values in source index %d (%d found, %d '
                    'parameters). Appending normal distribution fit to all entries.',
                    min_samples, source, num_rows, num_params)
        return None

    mean, covariance, degenerate = _sample_moments(samples)

    # Kish effective size over repeated rows: a stuck sampler's copies add no spread. A hash (+0.0 folds -0.0)
    # can merge distinct rows but never split equal ones, so a count that comes out short is redone exactly.
    hashes = pd.util.hash_pandas_object(pd.DataFrame(samples + 0.0), index=False)
    counts = hashes.value_counts(sort=False).to_numpy()
    effective = num_rows ** 2 / float(np.sum(np.square(counts, dtype=float)))
    if effective < min_samples:
        counts = np.unique(samples, axis=0, return_counts=True)[1]
        effective = num_rows ** 2 / float(np.sum(np.square(counts, dtype=float)))
    # A stuck stretch keeps its mean and borrows the pooled spread; a slot holding its few states in most samples
    # is a source held fixed, and the pooled (between-source) spread would make it absorb its neighbours.
    if effective < min_samples and 2 * num_rows <= num_samples:
        warnings.warn(
            f"mv_normal_fit: source slot {source} has {num_rows} retained samples, but "
            f"they repeat so much that they count as only {effective:.1f} independent "
            f"ones, fewer than the {min_samples} a {num_params}-parameter covariance "
            f"needs; the slot keeps its own mean and is given the covariance pooled over "
            f"the whole chain. Repeated rows usually mean the sampler was stuck, and a "
            f"covariance built from them would push every other draw of that source out "
            f"of this slot.",
            UserWarning, stacklevel=3,
        )
        return mean, pool().covariance

    # A constant parameter's ridge is scaled by its pooled variance, so the source keeps its separation in those units.
    ridge_fallback = pool().variances if degenerate.any() else None
    return mean, regularize_covariance(covariance, ridge_fallback)


def mv_normal_fit(chain: np.ndarray, max_num_sources: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a multivariate normal to the present rows of each of the first `max_num_sources` slots.

    Parameters:
        chain: (N, S, P) samples; a row with a NaN is absent.
        max_num_sources: number of leading slots to fit, at most S.

    Returns:
        (means, covariances) of shapes (max_num_sources, P) and (max_num_sources, P, P), each covariance with the
        ridge of `regularize_covariance`; a slot with fewer than P + 2 rows gets the pooled fit.

    Warns:
        UserWarning: a slot of repeated rows present in at most half the samples borrows the pooled covariance.
    """
    pool = functools.lru_cache(maxsize=None)(partial(_pool, chain))

    means = []
    cov_matrices = []
    for source in range(max_num_sources):
        sample_i = chain[:, source, :]
        valid = ~np.isnan(sample_i).any(axis=1)
        fit = _fit_slot(sample_i[valid], source, pool, chain.shape[0])
        if fit is None:
            fit = pool().mean, pool().covariance
        means.append(fit[0])
        cov_matrices.append(fit[1])
    return np.array(means), np.array(cov_matrices)


def _mean_and_std(values: np.ndarray) -> tuple[float, float]:
    """Return the mean and ddof=1 std of NaN-free values; the std is 0.0 for no spread or a round-off spread."""
    if values.size == 0:
        return float("nan"), 0.0
    if np.ptp(values) == 0:
        return float(values[0]), 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if not std > SPREAD_RTOL * float(np.max(np.abs(values))):
        std = 0.0
    return mean, std


def uni_normal_fit(chain: np.ndarray, max_num_sources: int, param_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a univariate normal to parameter `param_index` of each of the first `max_num_sources` slots.

    Parameters:
        chain: (N, S, P) samples; NaN values are left out.
        max_num_sources: number of leading slots to fit, at most S.
        param_index: parameter to fit.

    Returns:
        (means, stds), each (max_num_sources,), stds with ddof=1; a slot below ``MIN_UNIVARIATE_SAMPLES`` values
        takes the pooled parameter, and a zero std becomes ``sqrt(COVARIANCE_RIDGE)`` times the pooled std (or 1).
    """
    # Pool the same parameter over every slot; flattening the chain would mix unrelated parameters.
    pooled = chain[:, :, param_index].reshape(-1)
    pooled = pooled[~np.isnan(pooled)]
    pooled_mean, pooled_std = _mean_and_std(pooled)
    # Only a std that is zero or not finite is replaced: a floor at the pooled (between-source) spread widened a
    # narrow source tens of times and let it absorb a neighbour's samples.
    scale = pooled_std if np.isfinite(pooled_std) and pooled_std > 0.0 else 1.0
    replacement = float(np.sqrt(COVARIANCE_RIDGE) * scale)

    means = []
    stds = []
    for source in range(max_num_sources):
        sample_i = chain[:, source, param_index]
        valid_samples = sample_i[~np.isnan(sample_i)]
        if valid_samples.shape[0] < MIN_UNIVARIATE_SAMPLES:
            logger.info('Fewer than %d values in source index %d. Appending normal '
                        'distribution fit to all entries.',
                        MIN_UNIVARIATE_SAMPLES, source)
            mean, raw_std = pooled_mean, pooled_std
        else:
            mean, raw_std = _mean_and_std(valid_samples)
        if not raw_std > 0.0:
            logger.warning(
                'Source index %d has a degenerate spread in parameter %d '
                '(std=%r from %d retained samples); replacing it so the label keeps a '
                'proper distribution. A zero std would make its whole cost-matrix row '
                'NaN, and the label would then be assigned arbitrarily.',
                source, param_index, raw_std, valid_samples.shape[0],
            )
        means.append(mean)
        stds.append(float(raw_std) if np.isfinite(raw_std) and raw_std > 0.0 else replacement)

    return np.array(means), np.array(stds)


class _MvNormalFactors(NamedTuple):
    """Read-only, sample-independent part of `mv_normal_log_prob`: selected means, ``inv(L).T`` and log-determinants."""

    means: np.ndarray
    inverse_cholesky_t: np.ndarray
    logdet: np.ndarray


_last_mv_normal_factors: tuple[np.ndarray, np.ndarray, np.ndarray, _MvNormalFactors] | None = None


def _factorize_mv_normal(means: np.ndarray, cov_matrices: np.ndarray,
                         source_indices: np.ndarray) -> _MvNormalFactors:
    """Return the means, transposed inverse Cholesky factors and log-determinants of the selected fits."""
    means = means[source_indices]
    cov_matrices = cov_matrices[source_indices]

    # Cholesky, not inv: inv and slogdet return finite nonsense for a rank-deficient covariance instead of raising.
    try:
        chol = np.linalg.cholesky(cov_matrices)
    except np.linalg.LinAlgError:
        bad = []
        for i, matrix in enumerate(cov_matrices):
            try:
                np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:
                bad.append(int(source_indices[i]))
        raise np.linalg.LinAlgError(
            f"Covariance matrices for source indices {bad} are not positive definite, so the "
            f"multivariate normal log-pdf is undefined. This usually means the source was fit "
            f"from too few retained samples, or a parameter is constant across them. Check the "
            f"fit that produced these covariances (see petra.fits.mv_normal_fit, "
            f"which regularizes with COVARIANCE_RIDGE)."
        )

    logdet = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    identity = np.eye(chol.shape[-1])
    inverse_cholesky_t = np.empty(chol.shape, dtype=np.result_type(chol, float))
    for i, factor in enumerate(chol):
        inverse_cholesky_t[i] = solve_triangular(factor, identity, lower=True, check_finite=False).T

    for array in (means, inverse_cholesky_t, logdet):
        array.setflags(write=False)
    return _MvNormalFactors(means, inverse_cholesky_t, logdet)


def _mv_normal_factors(means: np.ndarray, cov_matrices: np.ndarray,
                       source_indices: np.ndarray) -> _MvNormalFactors:
    """Return `_factorize_mv_normal` of a fit, reused while the inputs keep the same shape, dtype and bytes."""
    global _last_mv_normal_factors
    memo = _last_mv_normal_factors  # read once: another thread may replace it
    if memo is not None and all(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
                                for a, b in zip(memo[:3], (means, cov_matrices, source_indices))):
        return memo[3]
    factors = _factorize_mv_normal(means, cov_matrices, source_indices)
    # Set only after a successful factorization, from private copies, so a failed or later edited fit never poisons it.
    _last_mv_normal_factors = (means.copy(), cov_matrices.copy(), source_indices.copy(), factors)
    return factors


def mv_normal_log_prob(sample: np.ndarray, aux_parameters: tuple[np.ndarray, np.ndarray],
                       source_index: int | Sequence[int] | np.ndarray) -> np.ndarray:
    """Return the multivariate normal log-density of every row of `sample` under each selected fit.

    Parameters:
        sample: (S, P) rows; a row with a NaN gives NaN.
        aux_parameters: (means, covariances) as `mv_normal_fit` returns them; lists of arrays work too.
        source_index: one fit index or several.

    Returns:
        (L, S) log-densities for L selected fits, or (S,) when L is 1.

    Raises:
        numpy.linalg.LinAlgError: a selected covariance is not positive definite.
    """
    means, cov_matrices = aux_parameters
    source_indices = np.atleast_1d(source_index)
    factors = _mv_normal_factors(np.asarray(means), np.asarray(cov_matrices), source_indices)

    d = sample.shape[1]
    diff = sample[np.newaxis, :, :] - factors.means[:, np.newaxis, :]
    whitened = np.matmul(diff, factors.inverse_cholesky_t)
    mahal = np.einsum("nsd,nsd->ns", whitened, whitened)
    logpdf = -0.5 * (d * np.log(2 * np.pi) + factors.logdet[:, None] + mahal)
    if logpdf.shape[0] == 1:
        return logpdf[0]
    return logpdf


def uni_normal_log_prob(sample: np.ndarray, aux_parameters: tuple[np.ndarray, np.ndarray],
                        source_index: int | Sequence[int] | np.ndarray, *, param_index: int) -> np.ndarray:
    """Return the univariate normal log-density of parameter `param_index` of every row under each selected fit.

    Parameters:
        sample: (S, P) rows.
        aux_parameters: (means, stds) as `uni_normal_fit` returns them.
        source_index: one fit index or several.
        param_index: parameter to score.

    Returns:
        (L, S) log-densities for L selected fits, or (S,) when L is 1.

    Raises:
        ValueError: a selected std is not strictly positive and finite.
    """
    values = sample[:, param_index]
    means, stds = aux_parameters
    source_indices = np.atleast_1d(source_index)
    means = np.array(means)[source_indices]
    stds = np.array(stds)[source_indices]

    # Refuse a zero or non-finite std by label here rather than let a NaN row reach the assignment.
    bad_scale = ~(stds > 0.0) | ~np.isfinite(stds)
    if np.any(bad_scale):
        bad = [int(i) for i in np.atleast_1d(source_indices)[bad_scale]]
        raise ValueError(
            f"Standard deviations for source indices {bad} are not strictly positive "
            f"and finite, so the univariate normal log-pdf is undefined. This usually "
            f"means the parameter is constant across the samples retained for that "
            f"source. Check the fit that produced these scales (see "
            f"petra.fits.uni_normal_fit, which floors them "
            f"with COVARIANCE_RIDGE)."
        )

    logpdf = (
        -0.5 * np.log(2 * np.pi) - np.log(stds)[:, None] - 0.5 * (((values - means[:, None]) / stds[:, None]) ** 2)
    )
    if logpdf.shape[0] == 1:
        return logpdf[0]
    return logpdf


def compute_cost_matrix(sample: np.ndarray, aux_parameters: Any, prob_in_model: np.ndarray, num_distributions: int,
                        *, log_prob: Callable[..., np.ndarray]) -> np.ndarray:
    """Return one sample's reward matrix, indexed [label, slot], for ``linear_sum_assignment(maximize=True)``.

    Parameters:
        sample: (S, P) rows; a row with a NaN is absent.
        aux_parameters: the fit's output, passed to `log_prob` unchanged.
        prob_in_model: (K,) inclusion probabilities, normally clipped to [eps, 1 - eps].
        num_distributions: number of labels K.
        log_prob: ``log_prob(sample, aux_parameters, labels)`` giving (K, S) log-densities.

    Returns:
        (K, S) array: ``log p_i + log_prob_i(x_j)`` for a present row j, ``log(1 - p_i)`` for an absent one.
    """
    with np.errstate(divide='ignore'):  # p of 0 or 1 is expected when prob_in_model is not clipped
        log_p = np.log(prob_in_model)
        # Indexed by label (row): log(1 - p) of the slot would make an absent column constant, inert in the assignment.
        log_not_p = np.log1p(-prob_in_model)[:, np.newaxis]

    matrix = log_prob(sample, aux_parameters, np.arange(num_distributions))
    matrix = log_p[:, np.newaxis] + matrix
    # Absence is read from the sample, not from NaN densities: flow evaluators floor non-finite outputs.
    return np.where(source_present(sample)[np.newaxis, :], matrix, log_not_p)
