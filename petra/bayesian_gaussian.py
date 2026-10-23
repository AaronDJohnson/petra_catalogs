"""Bayesian Gaussian relabeling: a conjugate Normal-Inverse-Wishart fit per slot, scored by its predictive t.

A sparse slot gets a broader predictive through the prior. :func:`niw_fit` and :func:`niw_aux_distribution` are an
ordinary fit and log-prob for :func:`petra.relabel.run_relabeling_loop`.
"""

import logging
from functools import partial
from typing import Sequence

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import gammaln

from petra.fits import compute_cost_matrix, regularize_covariance
from petra.initialization import relabel_univariate_normal, run_initialization_passes
from petra.make_catalog import relabel_mv_normal
from petra.posterior_chain import PosteriorChain
from petra.relabel import prepare_chain, run_relabeling_loop
from petra.utils import check_eps, check_int

logger = logging.getLogger(__name__)

DEFAULT_KAPPA0 = 0.01


def compute_niw_prior(chain: np.ndarray,
                      max_num_sources: int,
                      *,
                      kappa0: float = DEFAULT_KAPPA0,
                      nu0: int | None = None) -> dict:
    """Return the prior ``mu | Sigma ~ N(m0, Sigma / kappa0)``, ``Sigma ~ Inv-Wishart(nu0, Psi0)``, E[Sigma] pooled.

    Parameters:
        chain: (N, S, P) samples; a row with a NaN is absent.
        max_num_sources: number of leading slots pooled.
        kappa0: prior concentration on the location, in pseudo-observations; finite and > 0.
        nu0: prior degrees of freedom, finite and > P + 1; None means P + 2.

    Returns:
        dict with m0 (P,), kappa0, nu0 and Psi0 (P, P), Psi0 ridged like every petra covariance.

    Raises:
        ValueError: kappa0 or nu0 out of range, or fewer than two complete, finite source vectors.
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

    # The ridge keeps Psi0 positive definite when a column is constant; a singular Psi0 makes every Psi_n singular.
    Psi0 = regularize_covariance(pooled_cov * (nu0 - D - 1))

    return {"m0": m0, "kappa0": kappa0, "nu0": nu0, "Psi0": Psi0}


def _pooled_finite_source_vectors(chain: np.ndarray, max_num_sources: int) -> np.ndarray:
    """Return the complete, finite rows of the leading slots, refusing fewer than two."""
    all_data = chain[:, :max_num_sources, :].reshape(-1, chain.shape[2])
    valid_data = all_data[np.isfinite(all_data).all(axis=1)]
    if valid_data.shape[0] < 2:
        raise ValueError(
            "compute_niw_prior requires at least two complete, finite source vectors; "
            f"found {valid_data.shape[0]}."
        )
    return valid_data


def fit_source_niw(source_data: np.ndarray, prior: dict) -> dict:
    """Return the conjugate NIW posterior of one source; with no data it is the prior.

    Parameters:
        source_data: (n, P) present rows of the source.
        prior: NIW prior with keys m0, kappa0, nu0 and Psi0.

    Returns:
        dict with m_n, kappa_n, nu_n and Psi_n.
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
    """Return the NIW posterior of each of the first `max_num_sources` slots, from its rows without a NaN.

    Parameters:
        chain: (N, S, P) samples.
        max_num_sources: number of slots to fit.
        prior: NIW prior, as from compute_niw_prior.

    Returns:
        max_num_sources posteriors, as from fit_source_niw.
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
    """Return the multivariate-t log-density at `x`.

    Parameters:
        x: (D,) point or (n, D) points.
        df: degrees of freedom.
        loc: (D,) location.
        scale_inv: (D, D) inverse of the scale matrix.
        log_scale_det: log-determinant of the scale matrix.
        D: dimension.

    Returns:
        a scalar for a 1-D `x`, else an (n,) array.
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
    """Return each slot's predictive t: df = nu_n - D + 1, loc = m_n, scale = Psi_n (kappa_n + 1) / (kappa_n df).

    Parameters:
        all_posteriors: NIW posteriors, as from fit_all_sources_niw.
        D: dimension.

    Returns:
        per slot a dict with df, loc, scale_inv and log_scale_det, or None for a degenerate slot (``df <= 0`` or a
        scale that is not positive definite), which is then scored by its inclusion probability alone.

    Raises:
        numpy.linalg.LinAlgError: every slot is degenerate, so the cost matrix would carry no information.
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
        # Cholesky, not slogdet: slogdet passes a numerically singular scale whose inverse is meaningless.
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
    """Fit every slot's posterior predictive: the pooled NIW prior, a conjugate update, then a multivariate t.

    Parameters:
        chain: (N, S, P) samples; a row with a NaN is absent.
        max_num_sources: number of slots to fit.
        kappa0: prior concentration on the location, see compute_niw_prior.
        nu0: prior degrees of freedom, see compute_niw_prior.

    Returns:
        max_num_sources predictives, as from precompute_t_params.
    """
    prior = compute_niw_prior(chain, max_num_sources, kappa0=kappa0, nu0=nu0)
    all_posteriors = fit_all_sources_niw(chain, max_num_sources, prior)
    return precompute_t_params(all_posteriors, chain.shape[2])


def niw_aux_distribution(sample: np.ndarray,
                         aux_parameters: list[dict | None],
                         source_index: int | Sequence[int] | np.ndarray) -> np.ndarray:
    """Return the posterior-predictive log-density each requested slot gives every row of one sample.

    Parameters:
        sample: (S, P) rows; a row with a NaN is absent.
        aux_parameters: predictives, as from niw_fit.
        source_index: label or labels to evaluate.

    Returns:
        (L, S) array, or (S,) for one label: NaN at an absent row, 0 at a present row of a degenerate label.
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
    """Relabel with the NIW fit and its posterior-predictive cost until an iteration fails to lower the cost.

    Parameters:
        posterior_chain: chain to relabel; widened to max_num_sources if needed.
        max_num_sources: number of slots; None means the chain's num_sources.
        num_iterations: iteration ceiling, at least 1.
        eps: clip bound on prob_in_model inside the cost matrix only.
        kappa0: prior concentration on the locations, see compute_niw_prior.
        nu0: prior degrees of freedom, see compute_niw_prior.
        checkpoint_dir: directory for a checkpoint of the best labeling after each iteration; None for none.
        resume_from: checkpoint file, or directory of them, of a run with the same chain and settings.
        progress: show a progress bar over the samples of each iteration.

    Returns:
        the cheapest labeling seen, its cost in ``cost_dict[max_num_sources]`` and its prob_in_model unclipped.

    Raises:
        ValueError: a setting out of range, or a checkpoint written by another run or with another kappa0 or nu0.
    """
    logger.info("Bayesian Gaussian relabeling (conjugate NIW): at most %d iterations, "
                "%d parameters per source.",
                num_iterations, posterior_chain.num_params_per_source)

    return run_relabeling_loop(
        posterior_chain,
        partial(niw_fit, kappa0=kappa0, nu0=nu0),
        partial(compute_cost_matrix, log_prob=niw_aux_distribution),
        max_num_sources=max_num_sources,
        num_iterations=num_iterations,
        eps=eps,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        progress=progress,
        method="niw",
        settings={"kappa0": kappa0, "nu0": nu0},
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
                                   init_with_mv_normal: bool = True,
                                   init_num_iterations: int = 200) -> PosteriorChain:
    """Build a catalog with Bayesian Gaussian (conjugate NIW) relabeling after the shared initialization passes.

    Parameters:
        posterior_chain: input chain.
        max_num_sources: number of slots in the catalog, at least the chain's num_sources.
        num_iterations: ceiling on the NIW iterations.
        initialization_param_index: parameter for the univariate initialization pass; None skips that pass.
        shuffle_seed: seed for shuffling the entries of each sample; None keeps their order.
        rng_seed: unused (the NIW relabeling is deterministic); accepted so every make_catalog_* takes it.
        checkpoint_dir: directory for checkpoints of the NIW loop; None for none.
        resume_from: checkpoint of the NIW loop to resume; the initialization passes are then skipped.
        progress: show progress bars.
        eps: clip bound on prob_in_model inside the cost matrix only.
        kappa0: prior concentration on the locations, see compute_niw_prior.
        nu0: prior degrees of freedom on the covariances, see compute_niw_prior.
        init_with_mv_normal: run the multivariate normal initialization pass.
        init_num_iterations: iteration ceiling of each initialization pass.

    Returns:
        the relabeled chain, with its cost in cost_dict and its prob_in_model unclipped.

    Raises:
        ValueError: a setting out of range, initialization_param_index not None or a column of the chain, or fewer
            than two complete, finite source vectors.
    """
    check_int("init_num_iterations", init_num_iterations)
    check_int("num_iterations", num_iterations)
    check_eps(eps)

    posterior_chain = prepare_chain(posterior_chain, max_num_sources, shuffle_seed=shuffle_seed)

    # Refuse too few observations before the Gaussian initialization estimates a covariance; a resume checks later.
    if resume_from is None:
        _pooled_finite_source_vectors(posterior_chain.get_chain(), max_num_sources)

    posterior_chain = run_initialization_passes(
        posterior_chain,
        max_num_sources,
        univariate_relabeler=relabel_univariate_normal,
        mv_normal_relabeler=relabel_mv_normal,
        with_mv_normal=init_with_mv_normal,
        param_index=initialization_param_index,
        num_iterations=init_num_iterations,
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
