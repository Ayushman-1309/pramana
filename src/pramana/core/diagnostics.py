"""Convergence diagnostics for emcee chains.

Rule of thumb used throughout: trust a chain once its length exceeds
~50x the integrated autocorrelation time (tau). Acceptance fraction
outside roughly 0.2-0.5 usually means the proposal step size (walker
initialization spread) needs adjusting.
"""
import numpy as np


def summarize(
    sampler: "emcee.EnsembleSampler",
    param_names: list[str],
    burn_in_frac: float = 0.3,
    target_tau_multiple: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Print convergence summary and return flat_chain, tau."""
    nsteps = sampler.get_chain().shape[0]
    burn_in = int(nsteps * burn_in_frac)

    try:
        tau = sampler.get_autocorr_time(discard=burn_in, quiet=True)
    except Exception:
        tau = np.full(len(param_names), np.nan)

    acceptance = np.mean(sampler.acceptance_fraction)
    flat_chain = sampler.get_chain(discard=burn_in, flat=True)

    print(f"Acceptance fraction (mean): {acceptance:.3f}  (healthy range ~0.2-0.5)")
    converged = True
    for i, name in enumerate(param_names):
        med = np.median(flat_chain[:, i])
        lo, hi = np.percentile(flat_chain[:, i], [16, 84])
        this_tau = tau[i] if np.isfinite(tau[i]) else np.nan
        n_tau = (nsteps - burn_in) / this_tau if np.isfinite(this_tau) and this_tau > 0 else np.nan
        print(
            f"  {name}: {med:.4f}  +{hi - med:.4f}/-{med - lo:.4f}   "
            f"tau={this_tau:.1f}  (chain = {n_tau:.1f}x tau)"
        )
        if not np.isfinite(n_tau) or n_tau < target_tau_multiple:
            converged = False

    if not converged:
        print(
            f"WARNING: chain is shorter than {target_tau_multiple}x the autocorrelation "
            "time for at least one parameter. Increase --nsteps before trusting these results."
        )

    return flat_chain, tau


def gelman_rubin(chains: list[np.ndarray]) -> np.ndarray:
    """R-hat across independent chains. chains: list of (nsteps, ndim) arrays
    from separate emcee runs (different seeds/starting points). Want R-hat < 1.01.
    """
    m = len(chains)
    n = min(c.shape[0] for c in chains)
    stacked = np.array([c[:n] for c in chains])  # (m, n, ndim)

    chain_means = stacked.mean(axis=1)  # (m, ndim)
    grand_mean = chain_means.mean(axis=0)

    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = np.mean(stacked.var(axis=1, ddof=1), axis=0)

    var_hat = (1 - 1 / n) * W + B / n
    r_hat = np.sqrt(var_hat / W)
    return r_hat