"""Gaussian process emulation: train a fast surrogate for an expensive
theory function (CAMB CMB spectra costs ~0.1-1s per call; a real ACT-lite
or mflike MCMC needs O(10^5-10^6) evaluations, which is hours-to-days).
Train the GP on a few hundred CAMB evaluations spread across parameter
space, then the emulator answers in ~1ms — the standard trick that makes
CMB MCMC tractable at all (cosmopower, CosmoNet etc. are the same idea
with neural nets instead of GPs; GP is the right choice here since scikit-
learn needs no extra heavy install and the training-set sizes involved,
O(100s-1000s), suit a GP better than a network anyway).
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from scipy.stats import qmc


def latin_hypercube_design(bounds: list[tuple[float, float]], n_samples: int, seed: int = 42) -> np.ndarray:
    """Space-filling parameter grid for training-set generation — far
    better coverage per sample than a random or regular grid, standard
    practice for emulator training sets."""
    d = len(bounds)
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    los = np.array([b[0] for b in bounds])
    his = np.array([b[1] for b in bounds])
    return qmc.scale(unit_samples, los, his)


def train_emulator(
    theta_train: np.ndarray,
    y_train: np.ndarray,
    length_scale: float = 1.0,
    alpha: float = 1e-8,
) -> dict:
    """theta_train: (N, ndim) design points. y_train: (N,) or (N, n_out)
    expensive-function outputs at those points. For multi-output (e.g. a
    full C_ell spectrum), sklearn's GaussianProcessRegressor natively
    handles 2D y — fits one GP per output dimension with shared kernel
    hyperparameters, which is fine for smooth, correlated outputs like a
    power spectrum."""
    theta_train = np.atleast_2d(theta_train)

    # Normalize inputs to unit scale — GP length-scales behave much better
    # when parameters aren't on wildly different scales (e.g. H0~70 vs w~1)
    theta_mean, theta_std = theta_train.mean(axis=0), theta_train.std(axis=0)
    theta_std[theta_std == 0] = 1.0
    theta_norm = (theta_train - theta_mean) / theta_std

    kernel = ConstantKernel(1.0) * RBF(length_scale=[length_scale] * theta_train.shape[1]) \
        + WhiteKernel(noise_level=alpha)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(theta_norm, y_train)

    return {"gp": gp, "theta_mean": theta_mean, "theta_std": theta_std}


def emulate(emulator: dict, theta_query: np.ndarray, return_std: bool = False):
    """Evaluate the trained emulator at new parameter point(s). theta_query
    can be a single point (ndim,) or a batch (M, ndim)."""
    theta_query = np.atleast_2d(theta_query)
    theta_norm = (theta_query - emulator["theta_mean"]) / emulator["theta_std"]
    return emulator["gp"].predict(theta_norm, return_std=return_std)


def validate_emulator(emulator: dict, theta_test: np.ndarray, y_test_true: np.ndarray):
    """Leave-out validation: compare emulator predictions against real
    theory-function evaluations NOT used in training. Always run this
    before trusting an emulator inside a real MCMC — a GP will confidently
    interpolate garbage outside its training-set coverage without this
    check flagging it."""
    y_pred, y_std = emulate(emulator, theta_test, return_std=True)
    residual = y_test_true - y_pred
    rel_err = np.abs(residual) / (np.abs(y_test_true) + 1e-30)

    print(f"Emulator validation on {len(theta_test)} held-out points:")
    print(f"  max relative error: {rel_err.max():.4%}")
    print(f"  mean relative error: {rel_err.mean():.4%}")
    print(f"  mean predicted uncertainty / mean |residual|: "
          f"{np.mean(y_std) / (np.mean(np.abs(residual)) + 1e-30):.2f}  "
          "(should be ~O(1); <<1 means the GP is overconfident)")
    return rel_err