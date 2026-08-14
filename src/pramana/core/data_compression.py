"""Optimal data compression (MOPED — Heavens, Jimenez & Lahav 2000):
compress an N-point data vector into exactly P numbers (P = number of
parameters), constructed so those P numbers are provably lossless for
Fisher information — no parameter constraint is lost, but every
likelihood evaluation drops from an N x N covariance solve to a P x P
(in fact diagonal, unit-covariance) one. For P << N (e.g. compressing a
full CMB C_ell spectrum with N~thousands down to P~6 cosmological
parameters), this is the difference between an MCMC being feasible
overnight vs. not at all.

The compression vectors b_i are built from the Fisher-matrix machinery in
fisher_forecast.py — literally the same Jacobian, repurposed.

IMPORTANT — validated against a RAW Gaussian likelihood, not a
marginalized one: this module compresses chi2 = (data-model)^T Cinv
(data-model) directly. The SN likelihood elsewhere in this suite
(likelihood.py) analytically marginalizes over the M_B/H0 offset, which
is a DIFFERENT function of theta — compressing with this module but
validating against likelihood.py's marginalized likelihood will show large,
spurious discrepancies that look like a bug but aren't (caught during
development: discrepancies of ~10^3-10^4 that vanished once tested against
the matching raw likelihood instead). To use MOPED with the marginalized
SN likelihood, include the offset as an explicit parameter in the model
passed to moped_vectors (compress jointly over Om/w0/wa AND the offset)
rather than marginalizing before compressing.
"""
import numpy as np


def moped_vectors(
    model_func,
    theta_fiducial: np.ndarray,
    cov_inv: np.ndarray,
    args: tuple = (),
) -> np.ndarray:
    """MOPED compression vectors b_1...b_P (Heavens, Jimenez & Lahav 2000,
    eq. 15-16). Recursive construction:

        b_m = [Cinv @ mu_,m - sum_{q<m} (mu_,m . b_q) b_q]
              / sqrt(mu_,m . Cinv @ mu_,m - sum_{q<m} (mu_,m . b_q)^2)

    where mu_,i = d(model)/d(theta_i) (column i of the Jacobian). By
    construction each b_m satisfies b_m . mu_,q = delta_mq (orthonormal
    w.r.t. the OTHER parameters' gradients) — this is what makes the P
    compressed numbers y_m = b_m . data individually sensitive to exactly
    one parameter, with unit variance and zero cross-covariance.

    Returns B, shape (P, N): row m is b_m.
    """
    from pramana.core.fisher_forecast import numerical_jacobian

    J = numerical_jacobian(model_func, theta_fiducial, args=args)  # (N, P)
    n_params = J.shape[1]
    N = J.shape[0]

    B = np.zeros((n_params, N))
    Cinv_J = cov_inv @ J  # (N, P): Cinv @ mu_,i for each i, precomputed once

    for m in range(n_params):
        numerator = Cinv_J[:, m].copy()
        denom_sq = J[:, m] @ Cinv_J[:, m]  # mu_,m . Cinv . mu_,m
        for q in range(m):
            proj_coeff = J[:, m] @ B[q]  # mu_,m . b_q
            numerator = numerator - proj_coeff * B[q]
            denom_sq = denom_sq - proj_coeff**2
        B[m] = numerator / np.sqrt(denom_sq)

    return B


def compress(B: np.ndarray, data_vector: np.ndarray) -> np.ndarray:
    """Apply the MOPED compression: N numbers -> P numbers."""
    return B @ data_vector


def compressed_log_likelihood(
    theta: np.ndarray,
    B: np.ndarray,
    y_data_compressed: np.ndarray,
    model_func,
    args: tuple = (),
) -> float:
    """Log-likelihood using ONLY the P compressed numbers instead of the
    full N-point data vector + N x N covariance solve. The MOPED
    compressed variables have unit variance and (to the extent the
    linear/Gaussian approximation at theta_fiducial holds) zero
    cross-covariance, so the compressed likelihood is just a sum of
    squares — no matrix inversion needed at all, at any theta."""
    y_model_compressed = B @ model_func(theta, *args)
    delta = y_data_compressed - y_model_compressed
    return -0.5 * np.sum(delta**2)


def compare_compressed_vs_full(
    model_func,
    theta_fiducial: np.ndarray,
    cov_inv: np.ndarray,
    data_vector: np.ndarray,
    theta_test_points: np.ndarray,
    full_loglike_func,
    args: tuple = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical validation: evaluate both the full and MOPED-compressed
    likelihood at several test points and confirm the RELATIVE
    log-likelihoods agree (the compressed likelihood's absolute
    normalization differs from the full one by construction — only the
    shape, i.e. differences between points, is what matters for
    posteriors/MCMC)."""
    B = moped_vectors(model_func, theta_fiducial, cov_inv, args=args)
    y_data_compressed = B @ data_vector

    full_vals, compressed_vals = [], []
    for theta in theta_test_points:
        full_vals.append(full_loglike_func(theta))
        compressed_vals.append(compressed_log_likelihood(theta, B, y_data_compressed, model_func, args=args))

    full_vals = np.array(full_vals)
    compressed_vals = np.array(compressed_vals)

    full_diff = full_vals - full_vals[0]
    compressed_diff = compressed_vals - compressed_vals[0]

    max_discrepancy = np.max(np.abs(full_diff - compressed_diff))
    print(f"Max discrepancy in relative log-likelihood (full vs MOPED-compressed): "
          f"{max_discrepancy:.4f}  (should be << 1)")
    return B, full_diff, compressed_diff