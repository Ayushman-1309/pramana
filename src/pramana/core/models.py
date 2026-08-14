"""Cosmological models (LCDM, wCDM, CPL) + MODEL_REGISTRY

Cosmological distance-modulus models for SN Ia Hubble-diagram fitting.

Flat FRW geometry throughout (standard Pantheon+ assumption). Each model
function returns the distance modulus mu(z) = 25 + 5*log10(d_L[Mpc]).

H0 defaults to a fixed fiducial value (70 km/s/Mpc) rather than being a
free parameter. This is intentional: when using the analytic-marginalization
likelihood in likelihood.py (the standard SN-cosmology trick), H0 is
perfectly degenerate with the absolute magnitude M_B and gets absorbed into
the marginalized nuisance offset. Sampling it explicitly alongside that
marginalization would just recover the prior. If you need an actual H0
constraint, you need the full SH0ES Cepheid-calibrator likelihood, not this
marginalized shape-only fit.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

C_LIGHT = 299792.458  # km/s


def e_of_z_lcdm(z: np.ndarray, Om: float) -> np.ndarray:
    """E(z) = H(z)/H0 for flat LCDM."""
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def e_of_z_wcdm(z: np.ndarray, Om: float, w: float) -> np.ndarray:
    """E(z) = H(z)/H0 for flat wCDM (constant w)."""
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om) * (1 + z) ** (3 * (1 + w)))


def e_of_z_cpl(z: np.ndarray, Om: float, w0: float, wa: float) -> np.ndarray:
    """E(z) = H(z)/H0 for flat CPL (w(a) = w0 + wa*(1-a))."""
    de_evol = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om) * de_evol)


# Backward-compatible aliases (original SN-only skill used underscored names)
_e_of_z_lcdm = e_of_z_lcdm
_e_of_z_wcdm = e_of_z_wcdm
_e_of_z_cpl = e_of_z_cpl


def _luminosity_distance(
    z: np.ndarray,
    H0: float,
    e_of_z_func,
    params: tuple,
    z_grid_points: int = 2000,
) -> np.ndarray:
    """Comoving distance integral -> luminosity distance."""
    z = np.atleast_1d(z)
    zmax = max(z.max(), 1e-3)
    zgrid = np.linspace(0, zmax, z_grid_points)
    integrand = 1.0 / e_of_z_func(zgrid, *params)
    comoving_grid = cumulative_trapezoid(integrand, zgrid, initial=0)
    comoving = np.interp(z, zgrid, comoving_grid)
    dc = (C_LIGHT / H0) * comoving
    return (1 + z) * dc


def distance_modulus_lcdm(z: np.ndarray, Om: float, H0: float = 70.0) -> np.ndarray:
    """Distance modulus for flat LCDM."""
    dl = _luminosity_distance(z, H0, e_of_z_lcdm, (Om,))
    return 25 + 5 * np.log10(dl)


def distance_modulus_wcdm(z: np.ndarray, Om: float, w: float, H0: float = 70.0) -> np.ndarray:
    """Distance modulus for flat wCDM."""
    dl = _luminosity_distance(z, H0, e_of_z_wcdm, (Om, w))
    return 25 + 5 * np.log10(dl)


def distance_modulus_cpl(z: np.ndarray, Om: float, w0: float, wa: float, H0: float = 70.0) -> np.ndarray:
    """Distance modulus for flat CPL (w0waCDM)."""
    dl = _luminosity_distance(z, H0, e_of_z_cpl, (Om, w0, wa))
    return 25 + 5 * np.log10(dl)


# Central registry: add a new model here and every script picks it up automatically.
MODEL_REGISTRY = {
    "lcdm": {
        "func": distance_modulus_lcdm,
        "e_of_z": e_of_z_lcdm,
        "params": ["Om"],
        "priors": {"Om": (0.05, 0.6)},
        "labels": {"Om": r"$\Omega_m$"},
    },
    "wcdm": {
        "func": distance_modulus_wcdm,
        "e_of_z": e_of_z_wcdm,
        "params": ["Om", "w"],
        "priors": {"Om": (0.05, 0.6), "w": (-3.0, 0.0)},
        "labels": {"Om": r"$\Omega_m$", "w": r"$w$"},
    },
    "cpl": {
        "func": distance_modulus_cpl,
        "e_of_z": e_of_z_cpl,
        "params": ["Om", "w0", "wa"],
        "priors": {"Om": (0.05, 0.6), "w0": (-3.0, 1.0), "wa": (-3.0, 2.0)},
        "labels": {"Om": r"$\Omega_m$", "w0": r"$w_0$", "wa": r"$w_a$"},
    },
}