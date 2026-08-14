"""Fisher matrix forecasting: fast (no MCMC needed) Gaussian-approximation
parameter uncertainties, given a model and expected data precision. Use
this to ask "how good would a future survey need to be to detect w0 != -1
at 3-sigma" without ever running a real chain — a single Hessian evaluation
replaces thousands of likelihood calls. It's an approximation (assumes a
Gaussian posterior), so always sanity-check against a real MCMC/nested-
sampling run once real data is in hand — this module includes that
cross-check as `compare_to_mcmc`.
"""
import numpy as np


def numerical_jacobian(
    func,
    theta: np.ndarray,
    args: tuple = (),
    step_frac: float = 1e-4,
) -> np.ndarray:
    """d(func)/d(theta_i) via central differences, one column per parameter.
    func(theta, *args) must return a 1D array (e.g. model predictions at
    each data point).
    """
    theta = np.asarray(theta, dtype=float)
    f0 = np.asarray(func(theta, *args))
    n_params = len(theta)
    n_out = len(f0)

    jac = np.zeros((n_out, n_params))
    for i in range(n_params):
        h = max(abs(theta[i]) * step_frac, 1e-8)
        theta_plus, theta_minus = theta.copy(), theta.copy()
        theta_plus[i] += h
        theta_minus[i] -= h
        jac[:, i] = (func(theta_plus, *args) - func(theta_minus, *args)) / (2 * h)
    return jac


def fisher_matrix_gaussian(
    model_func,
    theta_fiducial: np.ndarray,
    cov_inv: np.ndarray,
    args: tuple = (),
) -> np.ndarray:
    """Standard Gaussian-likelihood Fisher matrix: F_ij = J^T Cinv J, where
    J is the Jacobian of model predictions w.r.t. parameters. Valid whenever
    the likelihood is (or is well-approximated by) a Gaussian in the data
    residuals — true for the SN/BAO likelihoods in this suite.
    """
    J = numerical_jacobian(model_func, theta_fiducial, args=args)
    return J.T @ cov_inv @ J


def forecast_errors(
    fisher: np.ndarray,
    param_names: list[str],
) -> tuple[dict[str, float], np.ndarray]:
    """1-sigma marginalized errors = sqrt(diag(F^-1)) — NOT 1/sqrt(diag(F)),
    which would ignore parameter degeneracies and understate the true error
    whenever parameters are correlated (nearly always, in cosmology).
    """
    cov = np.linalg.inv(fisher)
    errs = np.sqrt(np.diag(cov))
    print("Fisher-forecast 1-sigma marginalized errors:")
    for name, err in zip(param_names, errs):
        print(f"  sigma({name}) = {err:.4g}")
    return dict(zip(param_names, errs)), cov


def figure_of_merit(fisher: np.ndarray, i: int, j: int) -> float:
    """Dark Energy Task Force FoM convention for a 2-param sub-block
    (typically w0, wa): FoM = 1 / sqrt(det(Cov_2x2)) = pi / (area of the
    95% confidence ellipse). Higher = tighter joint constraint."""
    cov = np.linalg.inv(fisher)
    sub_cov = cov[np.ix_([i, j], [i, j])]
    return 1.0 / np.sqrt(np.linalg.det(sub_cov))


def fisher_ellipse(
    cov: np.ndarray,
    i: int,
    j: int,
    center: tuple[float, float],
    n_sigma: int = 1,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) points tracing the n_sigma confidence ellipse for params i, j
    — for overplotting directly on top of an MCMC/nested-sampling corner
    plot to visually check Fisher vs full-posterior agreement."""
    sub_cov = cov[np.ix_([i, j], [i, j])]
    eigvals, eigvecs = np.linalg.eigh(sub_cov)

    # Delta-chi2 for n_sigma in 2D (Wilks' theorem, 2 dof)
    from scipy.stats import chi2
    scale = np.sqrt(chi2.ppf(2 * (0.6827 if n_sigma == 1 else 0.9545 if n_sigma == 2 else 0.9973), df=2))

    t = np.linspace(0, 2 * np.pi, n_points)
    circle = np.array([np.cos(t), np.sin(t)])
    ellipse = eigvecs @ (np.sqrt(eigvals)[:, None] * circle) * scale
    return center[0] + ellipse[0], center[1] + ellipse[1]


def compare_to_mcmc(
    fisher_errs: dict[str, float],
    mcmc_flat_chain: np.ndarray,
    param_names: list[str],
) -> None:
    """Sanity check: Fisher forecast errors vs actual MCMC posterior
    std devs. Large disagreement flags a non-Gaussian posterior (common
    near prior boundaries or in strongly degenerate directions like
    w0-wa) — trust the MCMC in that case, Fisher is a linear approximation.
    """
    print(f"{'param':<8} {'Fisher sigma':>14} {'MCMC sigma':>14} {'ratio':>8}")
    for i, name in enumerate(param_names):
        f_err = fisher_errs[name]
        mcmc_err = np.std(mcmc_flat_chain[:, i])
        print(f"{name:<8} {f_err:>14.4g} {mcmc_err:>14.4g} {f_err / mcmc_err:>8.2f}")