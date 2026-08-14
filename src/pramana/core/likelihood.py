"""Gaussian likelihood for SN Ia Hubble-diagram fitting, with analytic
marginalization over the absolute-magnitude/H0 offset.

This is the standard trick in SN cosmology (e.g. Conley et al. 2011,
Betoule et al. 2014, and Pantheon+ shape-only analyses): rather than
fitting an unconstrained nuisance offset A as a free MCMC parameter, it is
marginalized out in closed form, which is faster and avoids a poorly-mixing
extra dimension in the chain.
"""
import numpy as np


def log_likelihood(
    theta: np.ndarray,
    z: np.ndarray,
    mb_obs: np.ndarray,
    cov_inv: np.ndarray,
    model_func,
    param_names: list[str],
) -> float:
    """Marginalized Gaussian log-likelihood (analytic M_B/H0 offset)."""
    params = dict(zip(param_names, theta))
    mu_model = model_func(z, **params)
    delta = mb_obs - mu_model

    ones = np.ones_like(delta)
    A = delta @ cov_inv @ delta
    B = ones @ cov_inv @ delta
    C = ones @ cov_inv @ ones

    chi2 = A - (B**2) / C
    return -0.5 * chi2


def log_prior(
    theta: np.ndarray,
    param_names: list[str],
    priors: dict[str, tuple[float, float]],
) -> float:
    """Flat priors within bounds."""
    for val, name in zip(theta, param_names):
        lo, hi = priors[name]
        if not (lo <= val <= hi):
            return -np.inf
    return 0.0


def log_probability(
    theta: np.ndarray,
    z: np.ndarray,
    mb_obs: np.ndarray,
    cov_inv: np.ndarray,
    model_func,
    param_names: list[str],
    priors: dict[str, tuple[float, float]],
) -> float:
    """Log-posterior = log-prior + log-likelihood (for emcee)."""
    lp = log_prior(theta, param_names, priors)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, z, mb_obs, cov_inv, model_func, param_names)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll