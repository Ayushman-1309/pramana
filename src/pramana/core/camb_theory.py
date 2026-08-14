"""CAMB wrapper: exact theory CMB power spectra and background quantities
(H(z), sound horizon, angular diameter distances) via a real Boltzmann
solver, rather than the EH98 fitting-formula approximation in bao_desi.py.

Use this when: fitting CMB data (ACT DR6, Planck), when BAO precision
better than ~1-2% matters, or whenever a task needs the exact early-universe
physics (recombination, radiation, neutrinos) that the fitting formula
doesn't capture. Use the fitting formula in bao_desi.sound_horizon_rd for
quick BAO-only exploratory fits where CAMB's per-call cost (~0.1-1s vs
~1e-5s for the formula) matters, e.g. inside a long MCMC/nested-sampling
loop over many likelihood evaluations.
"""
import numpy as np

try:
    import camb
except ImportError:
    camb = None


def _require_camb():
    if camb is None:
        raise RuntimeError(
            "CAMB not installed. Install with: `uv add camb` or `pip install camb`"
        )


def get_cmb_theory(
    H0: float,
    ombh2: float,
    omch2: float,
    tau: float = 0.054,
    As: float = 2.1e-9,
    ns: float = 0.965,
    lmax: int = 4000,
    mnu: float = 0.06,
    num_massive_neutrinos: int = 1,
) -> dict:
    """Returns theory C_ell (not D_ell) in uK^2 for TT, EE, TE, BB, and the
    lensing convergence spectrum C_ell^kappakappa — the exact inputs
    act_dr6_lenslike (and any primary-CMB likelihood) expects.
    """
    _require_camb()

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, tau=tau,
                        mnu=mnu, num_massive_neutrinos=num_massive_neutrinos)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=2)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
    lensing = results.get_lens_potential_cls(lmax=lmax, CMB_unit="muK")

    ell = np.arange(powers["total"].shape[0])
    tt, ee, bb, te = (powers["total"][:, i] for i in range(4))

    # get_lens_potential_cls returns [ClPP, ClPT, ClPE] as Dl-like (l(l+1))
    # scaled phi-phi; convert to the kappa-kappa convention act_dr6_lenslike
    # wants: Ckk = (l(l+1))^2 / 4 * Cphiphi / (2pi)
    ell_lens = np.arange(lensing.shape[0])
    clpp = lensing[:, 0] / (ell_lens * (ell_lens + 1) / (2 * np.pi) + 1e-30)
    clpp[0] = 0.0
    ckk = (ell_lens * (ell_lens + 1)) ** 2 / 4.0 * clpp

    derived = results.get_derived_params()

    return {
        "ell": ell,
        "cl_tt": tt,
        "cl_ee": ee,
        "cl_bb": bb,
        "cl_te": te,
        "ell_kk": ell_lens,
        "cl_kk": ckk,
        "rdrag": derived["rdrag"],
        "zdrag": derived["zdrag"],
        "H0": H0,
        "results": results,
    }


def sound_horizon_camb(H0: float, ombh2: float, omch2: float, mnu: float = 0.06) -> float:
    """Exact r_drag (Mpc) from CAMB — use in place of bao_desi.sound_horizon_rd
    when precision matters more than speed."""
    _require_camb()

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu)
    return camb.get_background(pars).get_derived_params()["rdrag"]


def hubble_of_z_camb(z_array: np.ndarray, H0: float, ombh2: float, omch2: float, mnu: float = 0.06) -> np.ndarray:
    """H(z) in km/s/Mpc from CAMB's exact background (includes radiation,
    massive neutrinos — matters at percent level at high z, e.g. for JWST
    high-z SNe or the DESI Lya point at z=2.33)."""
    _require_camb()

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu)
    bg = camb.get_background(pars)
    return bg.hubble_parameter(z_array)