"""JWST-era cosmological probes: three genuinely different roles bundled
into one module since none needs the file-size of a full likelihood
package (unlike bao_desi.py or cmb_act.py).

1. High-z SNe extension  — append JWST-discovered SNe to the Pantheon+
   Hubble diagram (generic mechanism; JWST SN discoveries are sparse and
   actively growing, so no fixed catalog is hard-coded).
2. H0 tension compilation — direct distance-ladder measurements,
   including the JWST-specific TRGB/JAGB/Cepheid results.
3. Growth tension (S8) compilation — same whisker-plot pattern applied to
   sigma8/S8, with a caveat about the separate JWST early-massive-galaxy
   question.
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. High-z SNe extension
# ---------------------------------------------------------------------

def append_supernovae(
    z_base: np.ndarray,
    mb_base: np.ndarray,
    cov_base: np.ndarray,
    z_new: np.ndarray,
    mb_new: np.ndarray,
    mb_err_new: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append new (e.g. JWST-discovered high-z) SNe to an existing Pantheon+
    Hubble diagram. New SNe are assumed uncorrelated with the base sample
    and with each other (reasonable default for a handful of individually-
    published high-z discoveries; revisit if a future JWST SN *sample*
    paper publishes its own covariance matrix).
    """
    z_new = np.atleast_1d(z_new)
    mb_new = np.atleast_1d(mb_new)
    mb_err_new = np.atleast_1d(mb_err_new)

    z_out = np.concatenate([z_base, z_new])
    mb_out = np.concatenate([mb_base, mb_new])

    n_base, n_new = len(z_base), len(z_new)
    cov_out = np.zeros((n_base + n_new, n_base + n_new))
    cov_out[:n_base, :n_base] = cov_base
    for i, err in enumerate(mb_err_new):
        cov_out[n_base + i, n_base + i] = err**2

    return z_out, mb_out, cov_out


# ---------------------------------------------------------------------
# 2. H0 tension compilation
# ---------------------------------------------------------------------

H0_MEASUREMENTS = {
    "Planck 2018 (CMB, LambdaCDM)":       {"H0": 67.4,  "err": 0.5,  "family": "early-universe"},
    "DESI DR2 BAO + Planck CMB":          {"H0": 68.17, "err": 0.28, "family": "early-universe"},
    "SH0ES (Riess+2022, Cepheids+SNe)":   {"H0": 73.04, "err": 1.04, "family": "distance-ladder"},
    "SH0ES JWST-extended (Riess+2025)":   {"H0": 72.6,  "err": 2.0,  "family": "distance-ladder (JWST)"},
    "CCHP JWST TRGB+JAGB (Freedman+2025)": {"H0": 69.96, "err": 1.53, "family": "distance-ladder (JWST)"},
}


def h0_tension_sigma(name_a: str, name_b: str, table: dict = H0_MEASUREMENTS) -> float:
    """Tension in sigma between two H0 measurements."""
    a, b = table[name_a], table[name_b]
    diff = abs(a["H0"] - b["H0"])
    sigma = np.sqrt(a["err"] ** 2 + b["err"] ** 2)
    return diff / sigma


def plot_h0_whisker(table: dict = H0_MEASUREMENTS, out_path: str | None = None):
    """Whisker plot of H0 measurements."""
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(table) + 1))
    names = list(table.keys())
    for i, name in enumerate(names):
        d = table[name]
        color = "#1f5fa8" if "early" in d["family"] else "#a83232"
        ax.errorbar(d["H0"], i, xerr=d["err"], fmt="o", color=color, capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax.axvline(H0_MEASUREMENTS["Planck 2018 (CMB, LambdaCDM)"]["H0"], color="#1f5fa8",
               ls="--", alpha=0.3)
    ax.axvline(H0_MEASUREMENTS["SH0ES (Riess+2022, Cepheids+SNe)"]["H0"], color="#a83232",
               ls="--", alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------
# 3. Growth tension (S8) compilation
# ---------------------------------------------------------------------

S8_MEASUREMENTS = {
    "Planck 2018 (CMB, LambdaCDM)": {"S8": 0.830, "err": 0.013, "family": "early-universe"},
    "DES Y3 (cosmic shear)":        {"S8": 0.776, "err": 0.017, "family": "weak-lensing"},
    "KiDS-1000 (cosmic shear)":     {"S8": 0.759, "err": 0.0225, "family": "weak-lensing"},
    "HSC Y1 (cosmic shear)":        {"S8": 0.780, "err": 0.0315, "family": "weak-lensing"},
}
# S8 = sigma8 * sqrt(Om/0.3). KiDS-1000 err symmetrized from +0.024/-0.021;
# HSC Y1 from +0.033/-0.030 — treat as approximate for whisker-plot purposes,
# use the asymmetric intervals directly for a real chi2/tension calculation.


def s8_tension_sigma(name_a: str, name_b: str, table: dict = S8_MEASUREMENTS) -> float:
    """Tension in sigma between two S8 measurements."""
    a, b = table[name_a], table[name_b]
    diff = abs(a["S8"] - b["S8"])
    sigma = np.sqrt(a["err"] ** 2 + b["err"] ** 2)
    return diff / sigma


def plot_s8_whisker(table: dict = S8_MEASUREMENTS, out_path: str | None = None):
    """Whisker plot of S8 measurements."""
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(table) + 1))
    names = list(table.keys())
    for i, name in enumerate(names):
        d = table[name]
        color = "#1f5fa8" if "early" in d["family"] else "#2f9e44"
        ax.errorbar(d["S8"], i, xerr=d["err"], fmt="o", color=color, capsize=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(r"$S_8 = \sigma_8\sqrt{\Omega_m/0.3}$")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# NOTE on JWST + early massive galaxies: this is a qualitatively different
# "tension" than S8 — it's about whether observed high-z (z~7-11) massive
# galaxy number densities exceed LambdaCDM halo-mass-function predictions
# given standard star-formation efficiency assumptions. It is NOT a single
# number you can drop into a whisker plot like H0 or S8: the inference
# depends heavily on stellar-mass-to-halo-mass modeling choices that are
# still actively debated. Flagging this explicitly rather than fabricating
# a "JWST sigma8" data point that doesn't exist in the literature as a
# clean, agreed-upon measurement the way S8 does.