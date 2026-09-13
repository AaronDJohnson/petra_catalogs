"""Bayesian Gaussian relabeling using conjugate Normal-Inverse-Wishart updates.

Uses the posterior predictive (multivariate t-distribution) as the auxiliary
distribution for Hungarian assignment. Sources with few data points get
naturally broader predictive distributions due to the prior, regularizing
the cost matrix without requiring MCMC sampling.

The method is expressed as an ordinary ``(fit, aux_distribution)`` pair --
:func:`niw_fit` and :func:`niw_aux_distribution` -- so it plugs into the same
machinery every other relabeler uses: :func:`petra.parametric_fits.create_parametric_fit`,
:func:`petra.cost_matrix.create_compute_cost_matrix` and
:func:`petra.relabel.run_relabeling_loop`.  Nothing about the outer loop is
local to this module any more: it used to be copied out because it kept the
best labeling seen and reverted to it when the cost increased, and the shared
loop now does that too.

Settings objects
----------------
:func:`make_catalog_bayesian_gaussian` takes the optional pre-relabeling as a
:class:`petra.options.Initialization` rather than as two loose keywords, which is
what the other ``make_catalog_*`` entry points do.  Both of its fields are still
accepted as plain keywords -- ``init_num_iterations=20`` and
``initialization=Initialization(num_iterations=20)`` are the same call -- but the
object *and* one of its fields in the same call is a :class:`TypeError`, never a
merge.  ``kappa0`` and ``nu0`` stay named: they are the method itself rather than
a concern separable from it.
"""

from functools import partial
from typing import Any, Sequence

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import gammaln

from petra.cost_matrix import create_compute_cost_matrix
from petra.initialization import relabel_univariate_normal, run_initialization_passes
from petra.make_catalog import relabel_mv_normal
from petra.options import Initialization
from petra.parametric_fits import _regularize_covariance, create_parametric_fit
from petra.posterior_chain import PosteriorChain
from petra.relabel import _validate_relabel_settings, prepare_chain, run_relabeling_loop
from petra.utils import get_logger, resolve_entry_point_kwargs

logger = get_logger(__name__)

#: Default prior concentration on the source location, ``kappa0``.  It is the
#: number of pseudo-observations the prior mean is worth, so a value well below
#: one makes the location prior almost uninformative.
DEFAULT_KAPPA0 = 0.01


def compute_niw_prior(chain: np.ndarray,
                      max_num_sources: int,
                      *,
                      kappa0: float = DEFAULT_KAPPA0,
                      nu0: int | None = None) -> dict:
    """
    Compute weakly informative Normal-Inverse-Wishart prior from pooled data.

    Prior specification:
        mu | Sigma ~ N(m0, Sigma / kappa0)
        Sigma ~ Inv-Wishart(nu0, Psi0)

    ``Psi0`` is always chosen so that the prior mean of ``Sigma`` is the pooled
    covariance of the whole chain; `kappa0` and `nu0` say how strongly that
    prior is believed.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Posterior chain with NaN for absent sources.
    max_num_sources : int
        Number of source slots.
    kappa0 : float, default `DEFAULT_KAPPA0`
        Prior concentration on the location, in pseudo-observations.  Must be
        strictly positive and finite. Larger values pull every source's predictive mean
        towards the pooled mean.
    nu0 : int, optional
        Prior degrees of freedom of the Inverse-Wishart.  Defaults to
        ``n_params + 2``, the smallest value for which the prior mean of
        ``Sigma`` exists; must be finite and greater than ``n_params + 1``. Larger values
        make the per-source covariances shrink harder towards the pooled one.

    Returns
    -------
    prior : dict
        Keys: m0 (D,), kappa0 (float), nu0 (int), Psi0 (D, D).

    Raises
    ------
    ValueError
        If either hyperparameter is not finite, `kappa0` is not strictly
        positive, or `nu0` is not greater than
        ``n_params + 1`` -- in which case ``Psi0`` would not be positive
        definite and the prior mean of ``Sigma`` would not exist -- or if the
        pooled chain contains fewer than two complete, finite source vectors.

    Notes
    -----
    ``Psi0`` carries the same relative ridge as every other covariance in
    petra (:data:`petra.parametric_fits.COVARIANCE_RIDGE`).  Without it a
    single constant parameter column makes the pooled covariance singular, and
    a singular ``Psi0`` propagates into every ``Psi_n``; the whole method then
    silently degenerates into an arbitrary tie-break rather than failing.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chain = rng.normal(size=(200, 2, 3))
    >>> prior = compute_niw_prior(chain, 2)
    >>> sorted(prior)
    ['Psi0', 'kappa0', 'm0', 'nu0']
    >>> prior['nu0'], prior['Psi0'].shape
    (5, (3, 3))
    >>> constant = np.zeros((200, 2, 3))          # every column constant
    >>> np.linalg.cholesky(compute_niw_prior(constant, 2)['Psi0']).shape
    (3, 3)

    A stronger prior is reachable from the entry point rather than baked in:

    >>> strong = compute_niw_prior(chain, 2, kappa0=5.0, nu0=50)
    >>> strong['kappa0'], strong['nu0']
    (5.0, 50)
    >>> compute_niw_prior(chain, 2, nu0=4)
    Traceback (most recent call last):
        ...
    ValueError: nu0 must be greater than n_params + 1 = 4, got 4.
    """
    D = chain.shape[2]
    if nu0 is None:
        nu0 = D + 2
    if not np.isfinite(kappa0):
        raise ValueError(f"kappa0 must be finite, got {kappa0}.")
    if not np.isfinite(nu0):
        raise ValueError(f"nu0 must be finite, got {nu0}.")
    if not kappa0 > 0:
        raise ValueError(f"kappa0 must be strictly positive, got {kappa0}.")
    if not nu0 > D + 1:
        raise ValueError(f"nu0 must be greater than n_params + 1 = {D + 1}, got {nu0}.")

    valid_data = _pooled_finite_source_vectors(chain, max_num_sources)

    m0 = valid_data.mean(axis=0)
    pooled_cov = np.cov(valid_data, rowvar=False)
    if pooled_cov.ndim == 0:
        pooled_cov = pooled_cov.reshape(1, 1)

    # Prior mean of Sigma = Psi0 / (nu0 - D - 1) = pooled_cov
    Psi0 = _regularize_covariance(pooled_cov * (nu0 - D - 1))

    return {"m0": m0, "kappa0": kappa0, "nu0": nu0, "Psi0": Psi0}


def _pooled_finite_source_vectors(chain: np.ndarray, max_num_sources: int) -> np.ndarray:
    """Validate the pooled observations needed to estimate an NIW prior."""
    all_data = chain[:, :max_num_sources, :].reshape(-1, chain.shape[2])
    valid_data = all_data[np.isfinite(all_data).all(axis=1)]
    if valid_data.shape[0] < 2:
        raise ValueError(
            "compute_niw_prior requires at least two complete, finite source vectors; "
            f"found {valid_data.shape[0]}."
        )
    return valid_data


def fit_source_niw(source_data: np.ndarray, prior: dict) -> dict:
    """
    Conjugate Normal-Inverse-Wishart posterior update for one source.

    Parameters
    ----------
    source_data : ndarray, shape (n_valid, D)
        Valid (non-NaN) observations for this source.
    prior : dict
        NIW prior params: m0, kappa0, nu0, Psi0.

    Returns
    -------
    posterior : dict
        Keys: m_n, kappa_n, nu_n, Psi_n.

    Examples
    --------
    >>> import numpy as np
    >>> prior = {'m0': np.zeros(1), 'kappa0': 0.01, 'nu0': 3,
    ...          'Psi0': np.eye(1)}
    >>> posterior = fit_source_niw(np.array([[1.0], [3.0]]), prior)
    >>> posterior['nu_n'], round(float(posterior['m_n'][0]), 6)
    (5, 1.99005)
    >>> fit_source_niw(np.empty((0, 1)), prior)['nu_n']    # no data: the prior
    3
    """
    m0 = prior["m0"]
    kappa0 = prior["kappa0"]
    nu0 = prior["nu0"]
    Psi0 = prior["Psi0"]
    n = source_data.shape[0]

    if n == 0:
        return {
            "m_n": m0.copy(), "kappa_n": kappa0,
            "nu_n": nu0, "Psi_n": Psi0.copy(),
        }

    x_bar = source_data.mean(axis=0)
    S = (source_data - x_bar).T @ (source_data - x_bar)

    kappa_n = kappa0 + n
    m_n = (kappa0 * m0 + n * x_bar) / kappa_n
    nu_n = nu0 + n
    diff = x_bar - m0
    Psi_n = Psi0 + S + (kappa0 * n / kappa_n) * np.outer(diff, diff)

    return {"m_n": m_n, "kappa_n": kappa_n, "nu_n": nu_n, "Psi_n": Psi_n}


def fit_all_sources_niw(chain: np.ndarray, max_num_sources: int, prior: dict) -> list[dict]:
    """
    Fit NIW posteriors for all source slots.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Posterior chain.
    max_num_sources : int
        Number of source slots to fit.
    prior : dict
        NIW prior params.

    Returns
    -------
    all_posteriors : list of dict
        Length max_num_sources, each with NIW posterior params.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chain = rng.normal(size=(50, 3, 2))
    >>> posteriors = fit_all_sources_niw(chain, 3, compute_niw_prior(chain, 3))
    >>> len(posteriors), posteriors[0]['nu_n']
    (3, 54)
    """
    all_posteriors = []
    for source_idx in range(max_num_sources):
        source_data = chain[:, source_idx, :]
        valid = ~np.isnan(source_data).any(axis=1)
        valid_data = source_data[valid]
        all_posteriors.append(fit_source_niw(valid_data, prior))
    return all_posteriors


def multivariate_t_logpdf(x: np.ndarray, df: float, loc: np.ndarray,
                          scale_inv: np.ndarray, log_scale_det: float,
                          D: int) -> np.ndarray:
    """
    Log-pdf of the multivariate t-distribution.

    Parameters
    ----------
    x : ndarray, shape (D,) or (n, D)
        Points to evaluate.
    df : float
        Degrees of freedom.
    loc : ndarray, shape (D,)
        Location parameter.
    scale_inv : ndarray, shape (D, D)
        Inverse of the scale matrix.
    log_scale_det : float
        Log determinant of the scale matrix.
    D : int
        Dimensionality.

    Returns
    -------
    logpdf : float or ndarray, shape (n,)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import multivariate_t
    >>> scale = np.array([[2.0, 0.3], [0.3, 1.0]])
    >>> loc, df = np.array([0.5, -0.5]), 7.0
    >>> x = np.array([[0.0, 0.0], [1.0, 2.0]])
    >>> ours = multivariate_t_logpdf(x, df, loc, np.linalg.inv(scale),
    ...                              float(np.linalg.slogdet(scale)[1]), 2)
    >>> bool(np.allclose(ours, multivariate_t(loc, scale, df).logpdf(x)))
    True
    """
    diff = x - loc
    if diff.ndim == 1:
        mahal = diff @ scale_inv @ diff
    else:
        mahal = np.einsum('ni,ij,nj->n', diff, scale_inv, diff)

    return (
        gammaln((df + D) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * D * np.log(df * np.pi)
        - 0.5 * log_scale_det
        - 0.5 * (df + D) * np.log1p(mahal / df)
    )


def precompute_t_params(all_posteriors: list[dict], D: int) -> list[dict | None]:
    """
    Precompute multivariate t parameters for each source slot.

    The posterior predictive is a multivariate t with:
        df = nu_n - D + 1
        location = m_n
        scale = Psi_n * (kappa_n + 1) / (kappa_n * df)

    Parameters
    ----------
    all_posteriors : list of dict
        NIW posterior params for each source.
    D : int
        Dimensionality.

    Returns
    -------
    t_params : list of (dict or None)
        Each dict has keys: df, loc, scale_inv, log_scale_det.
        None if the source has degenerate parameters.

    Raises
    ------
    numpy.linalg.LinAlgError
        If *every* source came back degenerate.  That used to pass silently:
        each cost-matrix row became the constant ``log(1 - prob_in_model[i])``,
        the assignment tie-broke arbitrarily, the cost stopped changing and the
        loop reported convergence -- a no-op dressed up as a result.

    Notes
    -----
    The scale matrix is factorized with :func:`numpy.linalg.cholesky` rather
    than tested with :func:`numpy.linalg.slogdet`.  ``slogdet`` returns a
    positive sign and a finite log-determinant for a numerically singular
    matrix, so the guard it was used for never fired; Cholesky fails on exactly
    the matrices whose inverse is meaningless.  The inverse itself is then
    obtained from that factorization.

    Examples
    --------
    >>> import numpy as np
    >>> prior = {'m0': np.zeros(1), 'kappa0': 0.01, 'nu0': 3, 'Psi0': np.eye(1)}
    >>> t_params = precompute_t_params([fit_source_niw(np.array([[1.0], [3.0]]), prior)], 1)
    >>> sorted(t_params[0])
    ['df', 'loc', 'log_scale_det', 'scale_inv']
    >>> t_params[0]['df']
    5
    >>> precompute_t_params([{'m_n': np.zeros(1), 'kappa_n': 1.0, 'nu_n': 3,
    ...                       'Psi_n': np.zeros((1, 1))}], 1)
    Traceback (most recent call last):
        ...
    numpy.linalg.LinAlgError: The posterior predictive of every one of the 1 source slots is degenerate, so the assignment cost matrix would carry no information about the data. This usually means a parameter is constant across the whole chain; drop it, or widen the prior.
    """
    t_params: list[dict | None] = []
    for source_idx, posterior in enumerate(all_posteriors):
        df = posterior["nu_n"] - D + 1
        if df <= 0:
            logger.warning("Source %d: %d degrees of freedom, falling back to "
                           "the inclusion probability alone.", source_idx, df)
            t_params.append(None)
            continue

        scale = (posterior["Psi_n"] * (posterior["kappa_n"] + 1)
                 / (posterior["kappa_n"] * df))
        try:
            chol = np.linalg.cholesky(scale)
        except np.linalg.LinAlgError:
            logger.warning("Source %d: the posterior predictive scale matrix is not "
                           "positive definite, falling back to the inclusion "
                           "probability alone.", source_idx)
            t_params.append(None)
            continue

        t_params.append({
            "df": df,
            "loc": posterior["m_n"],
            "scale_inv": cho_solve((chol, True), np.eye(D)),
            "log_scale_det": 2.0 * float(np.sum(np.log(np.diagonal(chol)))),
        })

    if t_params and all(tp is None for tp in t_params):
        raise np.linalg.LinAlgError(
            f"The posterior predictive of every one of the {len(t_params)} source slots "
            f"is degenerate, so the assignment cost matrix would carry no information "
            f"about the data. This usually means a parameter is constant across the "
            f"whole chain; drop it, or widen the prior."
        )

    return t_params


def niw_fit(chain: np.ndarray,
            max_num_sources: int,
            *,
            kappa0: float = DEFAULT_KAPPA0,
            nu0: int | None = None) -> list[dict | None]:
    """
    Fit the Bayesian Gaussian auxiliary distributions of a chain.

    This is a :func:`petra.parametric_fits.create_parametric_fit` compatible fit
    function: pooled NIW prior, conjugate per-source update, then the
    multivariate t posterior predictive of each source.

    Parameters
    ----------
    chain : ndarray, shape (n_samples, n_sources, n_params)
        Posterior chain with NaN for absent sources.
    max_num_sources : int
        Number of source slots to fit.
    kappa0 : float, default `DEFAULT_KAPPA0`
        Prior concentration on the location, see :func:`compute_niw_prior`.
    nu0 : int, optional
        Prior degrees of freedom, see :func:`compute_niw_prior`.  Defaults to
        ``n_params + 2``.

    Returns
    -------
    t_params : list of (dict or None), length max_num_sources
        Posterior predictive parameters, as returned by
        :func:`precompute_t_params`.

    Notes
    -----
    `kappa0` and `nu0` are keyword-only, so ``functools.partial`` can bind them
    and the result still matches the two-positional-argument fit-function
    protocol that :func:`petra.parametric_fits.create_parametric_fit` expects.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chain = rng.normal(size=(100, 2, 2)) + np.array([[0.0, 0.0], [8.0, 8.0]])
    >>> t_params = niw_fit(chain, 2)
    >>> [tp['df'] for tp in t_params]
    [103, 103]
    >>> bool(np.allclose([tp['loc'] for tp in t_params], [[0, 0], [8, 8]], atol=0.3))
    True
    >>> [tp['df'] for tp in niw_fit(chain, 2, nu0=20)]      # a stronger prior
    [119, 119]
    """
    prior = compute_niw_prior(chain, max_num_sources, kappa0=kappa0, nu0=nu0)
    all_posteriors = fit_all_sources_niw(chain, max_num_sources, prior)
    return precompute_t_params(all_posteriors, chain.shape[2])


def niw_aux_distribution(sample: np.ndarray,
                         aux_parameters: list[dict | None],
                         source_index: int | Sequence[int] | np.ndarray) -> np.ndarray:
    """
    Posterior predictive log-pdf of each source slot, for one sample.

    Parameters
    ----------
    sample : ndarray, shape (n_sources, n_params)
        One posterior sample; NaN rows mark absent sources.
    aux_parameters : list of (dict or None)
        Output of :func:`niw_fit`.
    source_index : int or array-like
        Index or indices of the fitted distributions to evaluate.

    Returns
    -------
    logpdf : ndarray
        Shape ``(n_sources,)`` for a scalar `source_index`, otherwise
        ``(len(source_index), n_sources)``. Absent sources are NaN, and
        :func:`petra.cost_matrix.create_compute_cost_matrix` uses
        ``log(1 - prob_in_model[i])`` for them. A degenerate predictive marked
        ``None`` contributes zero for present sources, so its assignment uses
        the inclusion probability alone as requested by
        :func:`precompute_t_params`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chain = rng.normal(size=(100, 2, 2)) + np.array([[0.0, 0.0], [8.0, 8.0]])
    >>> t_params = niw_fit(chain, 2)
    >>> logpdf = niw_aux_distribution(np.array([[0.0, 0.0], [np.nan, np.nan]]),
    ...                               t_params, [0, 1])
    >>> logpdf.shape
    (2, 2)
    >>> bool(logpdf[0, 0] > logpdf[1, 0])   # slot 0 belongs to source 0
    True
    >>> np.isnan(logpdf[:, 1]).tolist()     # slot 1 is absent from this sample
    [True, True]
    """
    source_indices = np.atleast_1d(source_index)
    n_sources, D = sample.shape

    logpdf = np.full((len(source_indices), n_sources), np.nan)
    valid_mask = ~np.isnan(sample).any(axis=1)

    if valid_mask.any():
        valid_data = sample[valid_mask]
        for row, distribution_index in enumerate(source_indices):
            t_param = aux_parameters[distribution_index]
            if t_param is None:
                logpdf[row, valid_mask] = 0.0
                continue
            logpdf[row, valid_mask] = multivariate_t_logpdf(
                valid_data, t_param["df"], t_param["loc"],
                t_param["scale_inv"], t_param["log_scale_det"], D,
            )

    if logpdf.shape[0] == 1:
        return logpdf[0]
    return logpdf


def bayesian_relabel_loop(posterior_chain: PosteriorChain,
                          max_num_sources: int | None = None,
                          num_iterations: int = 10,
                          eps: float = 1e-6,
                          *,
                          kappa0: float = DEFAULT_KAPPA0,
                          nu0: int | None = None,
                          checkpoint_dir: str | None = None,
                          resume_from: str | None = None,
                          progress: bool = True) -> PosteriorChain:
    """
    Main Bayesian relabeling loop: NIW fit -> cost matrix -> Hungarian -> repeat.

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Chain to relabel; widened to `max_num_sources` if necessary.
    max_num_sources : int, optional
        Number of source slots (defaults to the chain's ``num_sources``).
    num_iterations : int, default 10
        Maximum number of relabeling iterations.
    eps : float, default 1e-6
        Clip bound on the inclusion probabilities used *inside* the cost matrix,
        keeping them inside ``[eps, 1 - eps]`` so the logarithms there stay
        finite.  It does not reach the returned ``prob_in_model``; see Notes.
    kappa0 : float, default `DEFAULT_KAPPA0`
        Prior concentration on the source locations, see
        :func:`compute_niw_prior`.
    nu0 : int, optional
        Prior degrees of freedom of the Inverse-Wishart, see
        :func:`compute_niw_prior`.  Defaults to ``n_params + 2``.
    checkpoint_dir : str, optional
        Directory to save the best chain available after every iteration.
    resume_from : str, optional
        Checkpoint file, or directory of checkpoints, to resume from.  See
        :func:`petra.relabel.load_checkpoint`.
    progress : bool, default True
        Show a tqdm progress bar over the samples of each iteration.

    Returns
    -------
    result : PosteriorChain
        The cheapest labeling seen during the run, with that cost recorded in
        ``cost_dict[max_num_sources]`` and with ``prob_in_model`` recomputed
        from the returned samples *without* clipping, so that
        ``find_prob_in_model(result.get_chain(), max_num_sources, eps=0)``
        reproduces it exactly.

    Raises
    ------
    ValueError
        If `num_iterations` is less than 1, `max_num_sources` is smaller than
        the number of entries in the chain, or the NIW prior settings are out
        of range (see :func:`compute_niw_prior`).

    Notes
    -----
    This is :func:`petra.relabel.run_relabeling_loop` with the NIW fit and the
    posterior-predictive cost matrix bound to it.  The loop used to be spelled
    out here because it keeps the cheapest labeling seen and reverts to it as
    soon as the cost goes back up; the shared loop does that for every
    relabeler now, so the returned ``cost_dict[max_num_sources]`` is both the
    best and the final cost.

    `eps` governs only the clipping applied inside the cost matrix during
    iteration, where a probability of exactly 0 or 1 would make a ``log`` term
    infinite.  Only the returned array is unclipped.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.posterior_chain import PosteriorChain
    >>> from petra.utils import find_prob_in_model
    >>> rng = np.random.default_rng(0)
    >>> chain = rng.normal(size=(30, 2, 1)) + np.array([[0.0], [6.0]])
    >>> pc = PosteriorChain(chain, 2, 1, trans_dimensional=True)
    >>> result = bayesian_relabel_loop(pc, 2, num_iterations=5, progress=False)
    >>> sorted(result.cost_dict)
    [2]
    >>> result.prob_in_model
    array([1., 1.])
    >>> bool(np.array_equal(result.prob_in_model,
    ...                     find_prob_in_model(result.get_chain(), 2, eps=0)))
    True
    """
    # Name the method in the log: the shared loop reports iteration counts and
    # costs but has no idea which relabeler is producing them.
    logger.info("Bayesian Gaussian relabeling (conjugate NIW): at most %d iterations, "
                "%d parameters per source.",
                num_iterations, posterior_chain.num_params_per_source)

    return run_relabeling_loop(
        posterior_chain,
        create_parametric_fit(partial(niw_fit, kappa0=kappa0, nu0=nu0)),
        create_compute_cost_matrix(niw_aux_distribution),
        max_num_sources=max_num_sources,
        num_iterations=num_iterations,
        eps=eps,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        progress=progress,
    )


def make_catalog_bayesian_gaussian(posterior_chain: PosteriorChain,
                                   max_num_sources: int,
                                   *,
                                   num_iterations: int = 10,
                                   initialization_param_index: int | None = 0,
                                   shuffle_seed: int | None = None,
                                   rng_seed: int = 999,
                                   checkpoint_dir: str | None = None,
                                   resume_from: str | None = None,
                                   progress: bool = True,
                                   eps: float = 1e-6,
                                   kappa0: float = DEFAULT_KAPPA0,
                                   nu0: int | None = None,
                                   initialization: Initialization | None = None,
                                   **flat_keywords: Any) -> PosteriorChain:
    """
    Build a catalog using Bayesian Gaussian relabeling with conjugate NIW.

    Uses the Normal-Inverse-Wishart conjugate posterior to compute
    multivariate t posterior predictive distributions for each source.
    Sources with few data points get naturally broader predictive
    distributions, regularizing the assignment cost matrix.

    Workflow:
        1. Optionally shuffle entries (for reproducibility)
        2. Expand chain to max_num_sources
        3. Optionally initialize with univariate normal relabeling
        4. Optionally initialize with MV normal relabeling
        5. Run Bayesian NIW relabeling loop

    Parameters
    ----------
    posterior_chain : PosteriorChain
        Input posterior chain.
    max_num_sources : int
        Target number of source slots in the catalog.
    num_iterations : int, default 10
        Maximum number of relabeling iterations.  One iteration recomputes the
        conjugate Normal-Inverse-Wishart posterior of every source slot in
        closed form, precomputes each slot's multivariate-t posterior
        predictive, and solves one Hungarian assignment per sample against those
        predictives.  There is no training here either, so an iteration costs
        about what one of :func:`~petra.make_catalog.make_catalog_mv_normal`'s
        does; the default is twenty times smaller because the regularized
        predictive settles in a handful of iterations, not because an iteration
        is dearer.  The loop stops early as soon as an iteration fails to lower
        the assignment cost, so this is a ceiling, not a schedule.
    initialization_param_index : int or None, default 0
        Parameter index for the univariate initialization.  ``None`` skips it,
        and is the only thing that does: ``initialization.with_mv_normal``
        switches the multivariate pass alone (see Notes).  It names a column of
        *your* chain -- which parameter separates the sources -- rather than
        tuning the initializer, which is why it is here rather than on
        `initialization`.
    shuffle_seed : int, optional
        Seed for random shuffling of chain entries (for reproducibility).
    rng_seed : int, default 999
        Unused: the NIW relabeling is deterministic.  Accepted so that every
        ``make_catalog_*`` entry point takes the same keyword arguments.
    checkpoint_dir : str, optional
        Directory to save the best chain available after every iteration of the
        NIW loop.
    resume_from : str, optional
        Checkpoint file, or directory of checkpoints, to resume the NIW loop
        from.  The initialization steps are skipped when resuming, since the
        checkpoint already reflects them.
    progress : bool, default True
        Show a tqdm progress bar over the samples of each iteration.
    eps : float, default 1e-6
        Clip bound on the inclusion probabilities used *inside* the cost matrix.
        It does not reach the returned ``prob_in_model``; see Notes.
    kappa0 : float, default `DEFAULT_KAPPA0`
        Prior concentration on the source locations, in pseudo-observations.
        See :func:`compute_niw_prior`.
    nu0 : int, optional
        Prior degrees of freedom of the Inverse-Wishart on the source
        covariances.  Defaults to ``n_params + 2``, the weakest proper choice.
        See :func:`compute_niw_prior`.
    initialization : Initialization, optional
        The pre-relabeling passes -- ``with_mv_normal`` switches the
        multivariate one and ``num_iterations`` bounds each pass that runs.
        The univariate pass is switched by `initialization_param_index`
        instead, not by anything on this object.  ``None`` means
        ``Initialization()``.

    Returns
    -------
    result : PosteriorChain
        Relabeled chain with ``cost_dict`` and with ``prob_in_model`` recomputed
        from the returned samples *without* clipping, so that
        ``find_prob_in_model(result.get_chain(), max_num_sources, eps=0)``
        reproduces it exactly.

    Other Parameters
    ----------------
    init_with_mv_normal, init_num_iterations
        The fields of `initialization`, still accepted as flat keywords with
        unchanged meaning and no warning.  They keep the same spelling on all
        three ``make_catalog_*`` entry points, so a call can still be moved
        between methods without being rewritten.  ``init_num_iterations``
        governs the univariate initialization as well as the multivariate one.
    n_phases : int, optional
        Deprecated alias for `num_iterations`.
    mv_normal_init : bool, optional
        Deprecated alias for `init_with_mv_normal`.
    mv_normal_init_iterations : int, optional
        Deprecated alias for `init_num_iterations`.  Note that the replacement
        also governs the univariate initialization, which used to have its own
        budget.

    Raises
    ------
    ValueError
        If `max_num_sources` is smaller than the number of entries in the
        chain, if `initialization_param_index` is neither ``None`` nor a column
        of the chain, if the NIW prior settings are out of range (see
        :func:`compute_niw_prior`), or if a flat keyword carries a value its
        options class rejects.
    TypeError
        If an unknown keyword argument is supplied, or if an options object is
        passed alongside a flat keyword that would overwrite one of its fields.

    Notes
    -----
    `eps` governs only the clipping applied inside the cost matrix during
    iteration, where a probability of exactly 0 or 1 would make a ``log`` term
    infinite.  Only the returned array is unclipped; see
    :func:`petra.relabel.run_relabeling_loop`.

    Every argument after `max_num_sources` is keyword-only, and the defaults of
    ``initialization.with_mv_normal`` (was False) and
    `initialization_param_index` (was None) follow the shared ``make_catalog_*``
    contract rather than this module's earlier ones.  Pass them explicitly to
    reproduce a run made with an older version.

    An options object and one of its flat fields is a :class:`TypeError`, never
    a merge -- ``Initialization(num_iterations=20)`` alongside
    ``init_with_mv_normal=False`` would otherwise raise the question of which
    defaults the object contributes, and the answer would have to be remembered
    rather than read.

    The two initialization passes are switched independently:
    `initialization_param_index` selects the univariate one and
    ``initialization.with_mv_normal`` the multivariate one.  That is what this
    entry point has always done; in 1.1
    :func:`petra.copula_flows.make_catalog_copula_flows` was brought into line
    with it and now share
    :func:`petra.initialization.run_initialization_passes`, so the same pair of
    keywords initializes the same way whichever method is called.

    Examples
    --------
    >>> import numpy as np
    >>> from petra.options import Initialization
    >>> from petra.posterior_chain import PosteriorChain
    >>> rng = np.random.default_rng(0)
    >>> chain = rng.normal(size=(30, 3, 1)) + np.array([[0.0], [6.0], [12.0]])
    >>> pc = PosteriorChain(chain, 3, 1, trans_dimensional=True)
    >>> flat = make_catalog_bayesian_gaussian(
    ...     pc, 3, num_iterations=5, init_num_iterations=5, progress=False)
    >>> flat.chain.shape
    (30, 3, 1)
    >>> flat.prob_in_model
    array([1., 1., 1.])

    Spelling the same call with the settings object is the same run:

    >>> grouped = make_catalog_bayesian_gaussian(
    ...     pc, 3, num_iterations=5,
    ...     initialization=Initialization(num_iterations=5), progress=False)
    >>> bool(np.array_equal(flat.get_chain(), grouped.get_chain()))
    True
    """
    # Use the shared resolver rather than per-keyword `deprecated_keyword` calls:
    # the latter is handed its alias table one pair at a time, so this entry point
    # used to reject `n_phases` -- the alias for `num_iterations`, which it does
    # have -- while the other `make_catalog_*` accepted it.
    resolved, options = resolve_entry_point_kwargs(
        make_catalog_bayesian_gaussian, flat_keywords,
        options={"initialization": initialization},
        current={"num_iterations": num_iterations},
    )
    # `num_iterations` is the only named keyword an alias can resolve onto here --
    # the two initialization aliases land on `options` instead -- and it still has
    # to be read back out, or the "warn about an alias, then ignore it" bug is
    # reinstated for `n_phases`.
    num_iterations = resolved.get("num_iterations", num_iterations)
    _validate_relabel_settings(num_iterations, eps)
    init: Initialization = options["initialization"]

    # shuffle the entries and make sure the chain has the right shape
    posterior_chain = prepare_chain(posterior_chain, max_num_sources, shuffle_seed=shuffle_seed)

    # Reject insufficient observations before Gaussian initialization tries to
    # estimate a covariance. Resumed runs validate the loaded checkpoint below.
    if resume_from is None:
        _pooled_finite_source_vectors(posterior_chain.get_chain(), max_num_sources)

    # Share initialization with the copula method: one keyword, one pass.
    posterior_chain = run_initialization_passes(
        posterior_chain,
        max_num_sources,
        univariate_relabeler=relabel_univariate_normal,
        mv_normal_relabeler=relabel_mv_normal,
        with_mv_normal=init.with_mv_normal,
        param_index=initialization_param_index,
        num_iterations=init.num_iterations,
        resume_from=resume_from,
        progress=progress,
    )

    return bayesian_relabel_loop(
        posterior_chain, max_num_sources,
        num_iterations=num_iterations,
        eps=eps,
        kappa0=kappa0,
        nu0=nu0,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        progress=progress,
    )
