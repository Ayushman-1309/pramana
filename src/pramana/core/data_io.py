"""Loaders for Pantheon+SH0ES data release format.

Expected files (public release: github.com/PantheonPlusSH0ES/DataRelease):
    Pantheon+SH0ES.dat            - whitespace-delimited ASCII, header row
    Pantheon+SH0ES_STAT+SYS.cov   - covariance matrix: first line N, then
                                     N*N values (row-major flattened)

Column names referenced below match the public release. If your working
copy has renamed columns, adjust the constants at the top of load_pantheon.
"""
import numpy as np
import pandas as pd

Z_COL = "zHD"
MAG_COL = "m_b_corr"
CALIB_COL = "IS_CALIBRATOR"


def load_pantheon(
    data_path: str,
    cov_path: str | None = None,
    drop_calibrators: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Load SN redshifts + corrected magnitudes (+ optional covariance).

    drop_calibrators=True excludes the ~40 SNe with independent Cepheid
    distances used for the SH0ES H0 anchor — standard for a cosmology-only
    (shape-only) fit, since those points aren't drawn from the same
    cosmological Hubble-diagram likelihood.
    """
    df = pd.read_csv(data_path, sep=r"\s+")
    keep_idx = np.arange(len(df))
    if drop_calibrators and CALIB_COL in df.columns:
        mask = df[CALIB_COL].values == 0
        df = df[mask].reset_index(drop=True)
        keep_idx = keep_idx[mask]

    z = df[Z_COL].values
    mb = df[MAG_COL].values

    cov = None
    if cov_path is not None:
        cov = load_covariance(cov_path, keep_idx=keep_idx if drop_calibrators else None)

    return z, mb, cov, df


def load_covariance(cov_path: str, keep_idx: np.ndarray | None = None) -> np.ndarray:
    """Load Pantheon+ covariance matrix from file."""
    with open(cov_path) as f:
        n = int(f.readline().strip())
        vals = np.fromstring(f.read(), sep="\n", count=n * n)
    cov = vals.reshape((n, n))
    if keep_idx is not None:
        cov = cov[np.ix_(keep_idx, keep_idx)]
    return cov


def make_synthetic_dataset(
    n: int = 300,
    seed: int = 0,
    Om_true: float = 0.3,
    mb_scatter: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic SN Hubble diagram for smoke-testing the pipeline.

    Not for science use — purely to verify the likelihood/MCMC/plotting
    code runs end-to-end before pointing at real Pantheon+ files.
    """
    from pramana.core.models import distance_modulus_lcdm

    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.01, 1.2, n))
    mu_true = distance_modulus_lcdm(z, Om_true)
    M_true = -19.3
    mb_true = mu_true + M_true
    mb_obs = mb_true + rng.normal(0, mb_scatter, n)
    cov = np.diag(np.full(n, mb_scatter**2))
    return z, mb_obs, cov