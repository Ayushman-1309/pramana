"""DESI DR2 BAO likelihood: distance-ratio measurements from DESI Collaboration
et al. 2025 (Abdul-Karim et al. 2025), "DESI DR2 Results II: Measurements of
Baryon Acoustic Oscillations", Phys. Rev. D 112, 083515.

13 compressed measurements across 7 tracers, 0.295 <= z <= 2.330. Values
cross-checked against three independent secondary citations of Table IV.
Fiducial sound horizon used for the DESI template: r_d^fid = 147.05 Mpc
(not needed directly here since everything is already in units of r_d).

This module is self-contained (no external data download required) unlike
the ACT/JWST modules — the compressed BAO summary is small enough to ship
as data in code.
"""
import numpy as np

# Each entry: z_eff, DM/rd, DM/rd_err, DH/rd, DH/rd_err, rho(DM,DH), DV/rd, DV/rd_err
# BGS and QSO... only some tracers give both DM and DH; BGS gives DV only.
DESI_DR2_BAO_TABLE = {
    "BGS":       {"z": 0.295, "DV_rd": 7.942,  "DV_rd_err": 0.075},
    "LRG1":      {"z": 0.510, "DM_rd": 13.588, "DM_rd_err": 0.167,
                  "DH_rd": 21.863, "DH_rd_err": 0.425, "rho_MH": -0.459},
    "LRG2":      {"z": 0.706, "DM_rd": 17.351, "DM_rd_err": 0.177,
                  "DH_rd": 19.455, "DH_rd_err": 0.330, "rho_MH": -0.404},
    "LRG3+ELG1": {"z": 0.934, "DM_rd": 21.576, "DM_rd_err": 0.152,
                  "DH_rd": 17.641, "DH_rd_err": 0.193, "rho_MH": -0.416},
    "ELG2":      {"z": 1.321, "DM_rd": 27.601, "DM_rd_err": 0.318,
                  "DH_rd": 14.176, "DH_rd_err": 0.221, "rho_MH": -0.437},
    "QSO":       {"z": 1.484, "DM_rd": 30.512, "DM_rd_err": 0.760,
                  "DH_rd": 12.817, "DH_rd_err": 0.516, "rho_MH": -0.489},
    "Lya":       {"z": 2.330, "DM_rd": 38.988, "DM_rd_err": 0.531,
                  "DH_rd": 8.632,  "DH_rd_err": 0.101, "rho_MH": -0.431},
}
# rho_MH for LRG3+ELG1 averaged across two very close secondary citations
# (-0.425 and -0.408); flagged in the DESI DR2 BAO Data Reference section.

C_LIGHT = 299792.458  # km/s


def build_data_vector_and_cov():
    """Flatten DESI_DR2_BAO_TABLE into a data vector + block-diagonal
    covariance (standard treatment: intra-bin DM/DH correlated via rho_MH,
    zero correlation across different redshift bins)."""
    labels, z_list, data = [], [], []
    blocks = []

    for tracer, d in DESI_DR2_BAO_TABLE.items():
        if "DV_rd" in d:
            labels.append((tracer, "DV_rd"))
            z_list.append(d["z"])
            data.append(d["DV_rd"])
            blocks.append(np.array([[d["DV_rd_err"] ** 2]]))
        else:
            labels.append((tracer, "DM_rd"))
            labels.append((tracer, "DH_rd"))
            z_list.extend([d["z"], d["z"]])
            data.extend([d["DM_rd"], d["DH_rd"]])
            sM, sH, rho = d["DM_rd_err"], d["DH_rd_err"], d["rho_MH"]
            blocks.append(np.array([[sM**2, rho * sM * sH], [rho * sM * sH, sH**2]]))

    data = np.array(data)
    z_arr = np.array(z_list)

    n = len(data)
    cov = np.zeros((n, n))
    i = 0
    for block in blocks:
        k = block.shape[0]
        cov[i : i + k, i : i + k] = block
        i += k

    return labels, z_arr, data, cov


def sound_horizon_rd(Om: float, H0: float, Ob_h2: float = 0.02237, Neff: float = 3.044) -> float:
    """Approximate sound horizon at the drag epoch, Eisenstein & Hu (1998)
    fitting formula. Accurate to ~1-2%, adequate for exploratory fits;
    for publication-grade precision use CAMB/CLASS's exact r_d
    (see camb_theory.sound_horizon_camb for the full Boltzmann result).
    """
    h = H0 / 100.0
    om_h2 = Om * h**2
    ob_h2 = Ob_h2

    theta27 = 2.7255 / 2.7

    b1 = 0.313 * om_h2 ** (-0.419) * (1 + 0.607 * om_h2**0.674)
    b2 = 0.238 * om_h2**0.223
    z_drag = 1291 * om_h2**0.251 / (1 + 0.659 * om_h2**0.828) * (1 + b1 * ob_h2**b2)

    z_eq = 2.5e4 * om_h2 * theta27**-4
    k_eq = 7.46e-2 * om_h2 * theta27**-2

    # R(z) = 31.5 * Ob*h^2 * Theta_2.7^-4 * (z/1e3)^-1 -- baryon density in
    # the numerator (baryon-to-photon momentum density ratio), NOT the total
    # matter density. Using om_h2 here was the bug that gave rd ~ 3000 Mpc
    # instead of ~147 Mpc.
    R_d = 31.5 * ob_h2 * theta27**-4 * (z_drag / 1e3) ** -1
    R_eq = 31.5 * ob_h2 * theta27**-4 * (z_eq / 1e3) ** -1

    r_d = (2 / (3 * k_eq)) * np.sqrt(6 / R_eq) * np.log(
        (np.sqrt(1 + R_d) + np.sqrt(R_d + R_eq)) / (1 + np.sqrt(R_eq))
    )
    return r_d  # Mpc


def model_predictions(
    z_arr: np.ndarray,
    labels: list[tuple[str, str]],
    model_func,
    params: tuple,
    H0: float = 70.0,
    rd: float | None = None,
) -> np.ndarray:
    """Compute DM/rd, DH/rd, DV/rd predictions for the DESI redshifts.

    model_func here must be an E(z) function (models._e_of_z_*), not the
    SN distance-modulus wrapper — BAO needs D_M and D_H/H(z) directly,
    not a magnitude. Pass rd explicitly (e.g. from sound_horizon_rd) since
    unlike the SN module, BAO is NOT degenerate with rd — it's the whole
    point of the measurement.
    """
    from scipy.integrate import cumulative_trapezoid

    if rd is None:
        raise ValueError("BAO fits require an explicit sound horizon rd (Mpc) — "
                          "either fit it as a free parameter or fix it from a CMB prior.")

    zmax = z_arr.max()
    zgrid = np.linspace(0, zmax, 3000)
    Ez = model_func(zgrid, *params)
    comoving_grid = cumulative_trapezoid(1.0 / Ez, zgrid, initial=0)

    preds = {}
    for tracer, obs_type in labels:
        z = DESI_DR2_BAO_TABLE[tracer]["z"]
        Ez_here = model_func(np.array([z]), *params)[0]
        DM = (C_LIGHT / H0) * np.interp(z, zgrid, comoving_grid)
        DH = C_LIGHT / (H0 * Ez_here)
        DV = (z * DM**2 * DH) ** (1 / 3)
        preds[(tracer, "DM_rd")] = DM / rd
        preds[(tracer, "DH_rd")] = DH / rd
        preds[(tracer, "DV_rd")] = DV / rd

    return np.array([preds[key] for key in labels])


def log_likelihood_bao(
    theta: np.ndarray,
    e_of_z_func,
    param_names: list[str],
    H0: float = 70.0,
    rd_mode: str = "eh98",
    labels=None,
    z_arr=None,
    data=None,
    cov=None,
    rd_fixed: float | None = None,
) -> float:
    """BAO log-likelihood. theta must include the E(z) shape params.

    By default uses the shipped DESI DR2 reference table. To evaluate
    against user-loaded or synthetic BAO data (from the Data Hub), pass
    explicit labels/z_arr/data/cov — same format as
    build_data_vector_and_cov().

    If rd_mode == 'free', theta's last entry is rd (Mpc) sampled directly.
    If rd_mode == 'eh98', rd is computed from Om, H0 via the fitting formula.
    If rd_mode == 'planck_prior', a fixed Planck-anchored rd is used
    (pass rd_fixed via functools.partial when wiring into an MCMC).
    """
    if labels is None or z_arr is None or data is None or cov is None:
        labels, z_arr, data, cov = build_data_vector_and_cov()
    cov_inv = np.linalg.inv(cov)

    params = dict(zip(param_names, theta))
    Om = params["Om"]
    shape_params = tuple(params[p] for p in param_names if p != "rd")

    if rd_mode == "free":
        rd = params["rd"]
    elif rd_mode == "planck_prior" and rd_fixed is not None:
        rd = rd_fixed
    else:
        rd = sound_horizon_rd(Om, H0)

    pred = model_predictions(z_arr, labels, e_of_z_func, shape_params, H0=H0, rd=rd)
    delta = data - pred
    chi2 = delta @ cov_inv @ delta
    return -0.5 * chi2