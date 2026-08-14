"""Profile likelihood: frequentist alternative/cross-check to Bayesian
posteriors. For each value of a parameter of interest, maximize (not
marginalize) over the remaining ("nuisance") parameters. Confidence
intervals come from Wilks' theorem (delta chi2 thresholds), not credible
regions from a posterior — useful precisely when prior choice is
contested (e.g. the wa upper/lower bound genuinely changes the CPL
Bayesian posterior; profiling sidesteps that by construction) or as a
sanity check that MCMC/nested-sampling credible intervals aren't being
driven by prior volume effects rather than the data.
"""
import numpy as np
from scipy.optimize import minimize


def profile_scan(
    neg_log_likelihood,
    param_names: list[str],
    param_of_interest: str,
    scan_values: np.ndarray,
    fixed_bounds: dict[str, tuple[float, float]],
    x0: np.ndarray | None = None,
    args: tuple = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan `param_of_interest` over `scan_values`; at each point, minimize
    neg_log_likelihood over every OTHER parameter (profiling them out).

    neg_log_likelihood(theta, *args) -> -log L (a minimizer target, so
    negative of the log_likelihood functions elsewhere in this suite).
    fixed_bounds: {param_name: (lo, hi)} for every parameter (used both as
    optimizer bounds and, implicitly, as the profiling range for nuisance
    params).
    """
    idx = param_names.index(param_of_interest)
    other_idx = [i for i in range(len(param_names)) if i != idx]
    other_names = [param_names[i] for i in other_idx]

    if x0 is None:
        x0 = np.array([np.mean(fixed_bounds[p]) for p in param_names])

    profile_nll = []
    best_fits = []

    for val in scan_values:
        def wrapped_nll(theta_other, val=val):
            theta_full = np.empty(len(param_names))
            theta_full[idx] = val
            theta_full[other_idx] = theta_other
            return neg_log_likelihood(theta_full, *args)

        if not other_idx:
            # Nothing to profile over (single-parameter model) — just evaluate.
            theta_full = np.array([val])
            profile_nll.append(neg_log_likelihood(theta_full, *args))
            best_fits.append(np.array([]))
            continue

        x0_other = x0[other_idx]
        bounds_other = [fixed_bounds[p] for p in other_names]
        result = minimize(wrapped_nll, x0_other, bounds=bounds_other, method="L-BFGS-B")

        profile_nll.append(result.fun)
        best_fits.append(result.x)
        x0 = np.empty(len(param_names))
        x0[idx] = val
        x0[other_idx] = result.x  # warm-start the next scan point

    profile_nll = np.array(profile_nll)
    return scan_values, profile_nll, np.array(best_fits)


def confidence_interval_from_profile(
    scan_values: np.ndarray,
    profile_nll: np.ndarray,
    cl: float = 0.6827,
) -> tuple[float, float, float]:
    """Delta chi2 = 2*(profile_nll - min(profile_nll)) threshold from
    Wilks' theorem, 1 dof: 1.0 for 68.3%, 3.84 for 95%, 9.0 for 99.7%.
    Returns (best_fit_value, lower_bound, upper_bound)."""
    from scipy.stats import chi2

    delta_chi2 = 2 * (profile_nll - profile_nll.min())
    threshold = chi2.ppf(cl, df=1)

    best_idx = np.argmin(profile_nll)
    best_val = scan_values[best_idx]

    inside = scan_values[delta_chi2 <= threshold]
    return best_val, inside.min(), inside.max()


def global_best_fit(
    neg_log_likelihood,
    param_names: list[str],
    bounds: dict[str, tuple[float, float]],
    x0: np.ndarray | None = None,
    args: tuple = (),
) -> tuple[dict[str, float], float]:
    """Full (non-profiled) maximum-likelihood point estimate — the
    frequentist "best fit," useful as an x0 seed for profile_scan or as a
    quick point estimate before committing to a full MCMC/nested run."""
    if x0 is None:
        x0 = np.array([np.mean(bounds[p]) for p in param_names])
    bounds_list = [bounds[p] for p in param_names]
    result = minimize(neg_log_likelihood, x0, args=args, bounds=bounds_list, method="L-BFGS-B")
    return dict(zip(param_names, result.x)), result.fun