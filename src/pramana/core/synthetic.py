"""Synthetic data generators for every PRAMANA dataset family.

PRAMANA deliberately ships with NO real observational data files in the
repo. Every probe (SN Ia, BAO, CMB, JWST-era) supports two modes:

1. Manual download — the user fetches the official public release and
   loads it through the Data Hub (download links provided in the UI).
2. Synthetic — these generators produce realistic mock data from the
   fiducial model + the real measurement covariance, so the full pipeline
   can be smoke-tested end-to-end before real data is downloaded.

Nothing here is for science use: the mock spectra/measurements are drawn
from approximate analytic templates, clearly labeled as synthetic.
"""
import numpy as np


def synthetic_desi_bao(
    seed: int = 42,
    Om: float = 0.30,
    H0: float = 70.0,
    rd: float | None = None,
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic DESI DR2-like BAO data vector.

    Uses the real DESI DR2 bin structure (tracers, redshifts, per-bin
    DM/DH/DV observables and their covariance) but draws the central values
    from the fiducial LCDM model + Gaussian noise at the published errors.

    Returns (labels, z_arr, data, cov) in the same format as
    bao_desi.build_data_vector_and_cov().
    """
    from pramana.core.bao_desi import (
        DESI_DR2_BAO_TABLE,
        build_data_vector_and_cov,
        sound_horizon_rd,
        model_predictions,
    )
    from pramana.core.models import e_of_z_lcdm

    rng = np.random.default_rng(seed)
    labels, z_arr, _, cov = build_data_vector_and_cov()

    if rd is None:
        rd = sound_horizon_rd(Om, H0)

    pred = model_predictions(
        z_arr, labels, e_of_z_lcdm, (Om,), H0=H0, rd=rd
    )

    # Add Gaussian noise drawn from the real DESI covariance
    noise = rng.multivariate_normal(np.zeros_like(pred), cov)
    data = pred + noise
    return labels, z_arr, data, cov


def synthetic_highz_sn(
    n: int = 40,
    seed: int = 42,
    Om: float = 0.30,
    H0: float = 70.0,
    zmin: float = 1.0,
    zmax: float = 2.5,
    scatter: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic JWST-era high-z SNe (z > 1), uncorrelated with Pantheon+.

    Returns (z, m_b, sigma_m). Magnitudes use the distance-modulus model
    with a fixed absolute magnitude M = -19.3 (consistent with data_io's
    synthetic generator at low z).
    """
    from pramana.core.models import distance_modulus_lcdm

    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(zmin, zmax, n))
    mu = distance_modulus_lcdm(z, Om, H0=H0)
    mb_true = mu - 19.3
    mb_obs = mb_true + rng.normal(0.0, scatter, n)
    return z, mb_obs, np.full(n, scatter)


def _synthetic_cmb_tt(ell: np.ndarray, A: float = 2800.0) -> np.ndarray:
    """Rough analytic TT spectrum: Sachs-Wolfe plateau + damped acoustic
    peaks. Purely a shape template for pipeline smoke-testing — NOT a
    Boltzmann solution (that requires CAMB, see camb_theory.py)."""
    base = A / (ell + 1.0) ** 1.55
    peaks = [
        (220.0, 1.35, 60.0),
        (540.0, 0.95, 80.0),
        (810.0, 0.70, 90.0),
        (1130.0, 0.50, 110.0),
        (1420.0, 0.36, 120.0),
    ]
    spectrum = base.copy()
    for pos, amp, width in peaks:
        spectrum += A * amp * np.exp(-0.5 * ((ell - pos) / width) ** 2)
    return spectrum


def synthetic_act_cmb(
    seed: int = 42,
    lmax: int = 3000,
    l_kk_max: int = 4000,
) -> dict:
    """Synthetic ACT DR6-like CMB spectra (C_ell, not D_ell, uK^2 units).

    Returns the same dict shape as camb_theory.get_cmb_theory:
        ell, cl_tt, cl_ee, cl_te, cl_bb, ell_kk, cl_kk
    but generated from analytic templates with noise — clearly synthetic.
    """
    from pramana.core.camb_theory import get_cmb_theory

    rng = np.random.default_rng(seed)

    # Prefer CAMB if available; else fall back to the analytic template.
    try:
        theory = get_cmb_theory(H0=67.4, ombh2=0.0224, omch2=0.120, lmax=lmax)
        ell = theory["ell"]
        cl_tt = theory["cl_tt"].copy()
        cl_ee = theory["cl_ee"].copy()
        cl_te = theory["cl_te"].copy()
        cl_bb = theory["cl_bb"].copy()
        ell_kk = theory["ell_kk"]
        cl_kk = theory["cl_kk"].copy()
    except Exception:
        ell = np.arange(2, lmax + 1)
        cl_tt = _synthetic_cmb_tt(ell)
        cl_ee = cl_tt * 0.45 * np.exp(-0.5 * ((ell - 220.0) / 90.0) ** 2)
        cl_te = cl_tt * 0.18 * np.cos(2.0 * np.pi * (ell - 220.0) / 420.0) * np.exp(-(ell / 2000.0) ** 2)
        cl_bb = 1e-4 * cl_tt
        ell_kk = np.arange(2, l_kk_max + 1)
        cl_kk = 1.2e-4 * np.exp(-ell_kk / 800.0) + 1e-6

    # Add observation-like noise (fractional) so the spectra aren't perfect
    n_tt = rng.normal(0.0, 0.02, len(cl_tt)) * np.abs(cl_tt)
    n_ee = rng.normal(0.0, 0.03, len(cl_ee)) * np.abs(cl_ee)
    n_te = rng.normal(0.0, 0.05, len(cl_te)) * np.abs(cl_te)
    n_kk = rng.normal(0.0, 0.05, len(cl_kk)) * np.abs(cl_kk)

    return {
        "ell": ell,
        "cl_tt": cl_tt + n_tt,
        "cl_ee": cl_ee + n_ee,
        "cl_te": cl_te + n_te,
        "cl_bb": cl_bb + np.abs(rng.normal(0, 1e-6, len(cl_bb))),
        "ell_kk": ell_kk,
        "cl_kk": cl_kk + n_kk,
        "synthetic": True,
    }


def synthetic_h0_tables(
    seed: int = 42,
    h0_true: float = 70.0,
    scatter: float = 1.0,
    n: int = 5,
) -> dict:
    """Synthetic H0 tension compilation.

    Returns a dict mapping measurement name -> {"H0", "err", "family"}
    drawn around a chosen central value. Families alternate between
    "early-universe" and "distance-ladder" to preserve the whisker-plot
    color semantics used by jwst_probes.
    """
    rng = np.random.default_rng(seed)
    families = ["early-universe", "distance-ladder", "distance-ladder (JWST)"]
    table = {}
    for i in range(n):
        err = max(0.3, scatter * (0.5 + rng.random()))
        value = h0_true + rng.normal(0.0, scatter * 0.7)
        table[f"Synthetic H0 #{i + 1}"] = {
            "H0": round(float(value), 2),
            "err": round(float(err), 2),
            "family": families[i % len(families)],
        }
    return table


def synthetic_s8_tables(
    seed: int = 42,
    s8_true: float = 0.80,
    scatter: float = 0.02,
    n: int = 4,
) -> dict:
    """Synthetic S8 tension compilation (same structure as H0 tables)."""
    rng = np.random.default_rng(seed)
    families = ["early-universe", "weak-lensing", "weak-lensing"]
    table = {}
    for i in range(n):
        err = max(0.008, scatter * (0.5 + rng.random()))
        value = s8_true + rng.normal(0.0, scatter * 0.6)
        table[f"Synthetic S8 #{i + 1}"] = {
            "S8": round(float(value), 3),
            "err": round(float(err), 3),
            "family": families[i % len(families)],
        }
    return table