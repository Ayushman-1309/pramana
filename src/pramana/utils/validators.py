"""Data format validators for Pantheon+, DESI BAO, ACT data files."""
import numpy as np
import pandas as pd


def validate_pantheon_data(data_path: str) -> dict:
    """Validate Pantheon+ data file format and return info."""
    df = pd.read_csv(data_path, sep=r"\s+", nrows=5)
    required_cols = ["zHD", "m_b_corr", "IS_CALIBRATOR"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")
    return {
        "n_rows": len(pd.read_csv(data_path, sep=r"\s+")),
        "columns": list(df.columns),
        "z_range": (df["zHD"].min(), df["zHD"].max()),
    }


def validate_pantheon_cov(cov_path: str, n_data: int) -> dict:
    """Validate Pantheon+ covariance matrix format."""
    with open(cov_path) as f:
        n = int(f.readline().strip())
        vals = np.fromstring(f.read(), sep="\n", count=n * n)
    cov = vals.reshape((n, n))
    if cov.shape != (n_data, n_data):
        raise ValueError(f"Covariance shape {cov.shape} != data vector length {n_data}")
    # Check symmetry
    if not np.allclose(cov, cov.T):
        raise ValueError("Covariance matrix is not symmetric")
    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() <= 0:
        raise ValueError(f"Covariance matrix not positive definite (min eigval: {eigvals.min()})")
    return {
        "shape": cov.shape,
        "min_eigval": eigvals.min(),
        "max_eigval": eigvals.max(),
        "condition_number": eigvals.max() / eigvals.min(),
    }


def validate_desi_bao_file(filepath: str) -> dict:
    """Validate DESI BAO data file (if using custom file instead of reference table)."""
    data = np.loadtxt(filepath)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("DESI BAO file must have columns: z, observable, error")
    return {
        "n_points": len(data),
        "z_range": (data[:, 0].min(), data[:, 0].max()),
    }


def validate_act_data_dir(data_dir: str) -> dict:
    """Validate ACT DR6 data directory structure."""
    import os
    required = ["act_dr6_lenslike", "act_dr6_cmbonly"]
    missing = [d for d in required if not os.path.isdir(os.path.join(data_dir, d))]
    if missing:
        raise ValueError(f"Missing ACT data subdirectories: {missing}")
    return {"data_dir": data_dir, "subdirs": required}