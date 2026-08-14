"""ACT DR6 CMB likelihoods: lensing (act_dr6_lenslike) and primary CMB
(ACT-lite foreground-marginalized, act_dr6_cmbonly via Cobaya+CAMB).

Both wrap the OFFICIAL ACT Collaboration likelihood code rather than
reimplementing it — the lensing normalization/binning and the primary-CMB
foreground marginalization encode instrument-specific corrections that
aren't safe to reconstruct from first principles. This module is the
integration harness; the physics comes from the official packages.

NEITHER likelihood ships with its data (both require a download from
NASA's LAMBDA archive). See the ACT DR6 Integration Reference for exact
download steps to run locally. What's validated here: the API wiring,
parameter flow, and (for the primary-CMB path) that Cobaya + CAMB + the
likelihood class build a working model — everything except the final
"read this data file" step, which needs the real download.
"""
import numpy as np


# ---------------------------------------------------------------------
# Lensing likelihood (act_dr6_lenslike) — standalone, no Cobaya needed
# ---------------------------------------------------------------------

def act_dr6_lensing_loglike(
    cl_kk: np.ndarray,
    ell_kk: np.ndarray,
    cl_tt: np.ndarray,
    cl_ee: np.ndarray,
    cl_te: np.ndarray,
    cl_bb: np.ndarray,
    ell_cmb: np.ndarray,
    variant: str = "act_baseline",
    lens_only: bool = False,
    like_corrections: bool = True,
    data_dict: dict | None = None,
) -> float:
    """Direct wrapper around act_dr6_lenslike.generic_lnlike.

    Pass theory spectra from camb_theory.get_cmb_theory(...) — note ACT
    wants C_ell (not D_ell), uK^2 units for the CMB spectra, kappa-kappa
    (not phi-phi) for lensing. camb_theory.py already returns these
    conventions.

    data_dict can be pre-loaded once (via act_dr6_lensing_load_data) and
    reused across many likelihood evaluations in an MCMC/nested-sampling
    loop, since load_data() re-reads the data file from disk every call.
    """
    try:
        import act_dr6_lenslike as alike
    except ImportError:
        raise RuntimeError(
            "act_dr6_lenslike not installed. Download from LAMBDA and install locally."
        )

    if data_dict is None:
        data_dict = alike.load_data(variant, lens_only=lens_only,
                                     like_corrections=like_corrections)

    return alike.generic_lnlike(data_dict, ell_kk, cl_kk, ell_cmb, cl_tt, cl_ee, cl_te, cl_bb)


def act_dr6_lensing_load_data(
    variant: str = "act_baseline",
    lens_only: bool = False,
    like_corrections: bool = True,
) -> dict:
    """Load the ACT DR6 lensing data once; pass the result into every
    act_dr6_lensing_loglike call to avoid re-reading from disk per step.

    Requires the data tarball from
    https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/data/
    extracted into the act_dr6_lenslike package's data/ directory.
    """
    try:
        import act_dr6_lenslike as alike
    except ImportError:
        raise RuntimeError(
            "act_dr6_lenslike not installed. Download from LAMBDA and install locally."
        )
    return alike.load_data(variant, lens_only=lens_only, like_corrections=like_corrections)


# ---------------------------------------------------------------------
# Primary CMB, foreground-marginalized (ACT-lite / act_dr6_cmbonly)
# via Cobaya's "use as a library" API — drives the OFFICIAL likelihood
# class + CAMB, so it inherits ACT's real ell-cuts, covariance, and the
# two ACT-lite nuisance params (A_act, P_act) rather than us guessing them.
# ---------------------------------------------------------------------

def build_act_cmbonly_model(packages_path: str | None = None, extra_params: dict | None = None):
    """Build a Cobaya Model wrapping CAMB (theory) + ACTDR6CMBonly
    (likelihood). Returns the Model; call model.loglike(params_dict) to
    evaluate, or model.logposterior(params_dict) if priors are set in info.

    This is the standard "cobaya as a library" pattern
    (cobaya.model.get_model) — NOT a from-scratch reimplementation. Fails
    at model.loglike() time with a clear FileNotFoundError if the ACT-lite
    data hasn't been downloaded; the model itself builds fine without it,
    which is what lets us validate the wiring in this sandbox.
    """
    try:
        from cobaya.model import get_model
    except ImportError:
        raise RuntimeError("Cobaya not installed. Install with: `uv add cobaya`")

    base_params = {
        "H0": {"prior": {"min": 60, "max": 80}, "ref": 67.4, "proposal": 0.5},
        "ombh2": {"prior": {"min": 0.018, "max": 0.026}, "ref": 0.0224, "proposal": 0.0001},
        "omch2": {"prior": {"min": 0.08, "max": 0.20}, "ref": 0.120, "proposal": 0.001},
        "tau": {"prior": {"min": 0.01, "max": 0.15}, "ref": 0.054, "proposal": 0.01},
        "logA": {"prior": {"min": 2.6, "max": 3.6}, "ref": 3.05, "proposal": 0.01,
                 "drop": True, "latex": r"\log(10^{10} A_s)"},
        "As": {"value": "lambda logA: 1e-10*np.exp(logA)"},
        "ns": {"prior": {"min": 0.9, "max": 1.05}, "ref": 0.965, "proposal": 0.002},
    }
    if extra_params:
        base_params.update(extra_params)

    info = {
        "params": base_params,
        "likelihood": {"act_dr6_cmbonly.ACTDR6CMBonly": {}},
        "theory": {"camb": {"stop_at_error": True}},
        "packages_path": packages_path,
    }
    return get_model(info)


def act_cmbonly_loglike(model, param_dict: dict) -> float:
    """Evaluate the ACT-lite log-likelihood at a parameter point. param_dict
    needs every sampled param in the model (cosmological + A_act, P_act
    nuisance params — see act_dr6_cmbonly's own param block, ell_cuts
    TT/TE/EE = [600, 6500])."""
    return model.loglike(param_dict, return_derived=False)