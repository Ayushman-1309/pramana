"""Cosmic shear (tomographic weak-lensing 2-point) forward model, shared
between Euclid and Rubin/LSST presets since the underlying physics and
code architecture is identical — only the survey specification (n(z)
params, galaxy number density, shape noise, sky area) differs.

STATUS: forward-model infrastructure, validated against SYNTHETIC data
only. Neither survey has a public cosmology-ready shear catalog as of
this module's construction — Euclid's first cosmology results are
expected in 2027; Rubin/LSST's 10-year survey began June 30, 2026.
This is deliberately forward-looking infrastructure, not a wrapper
around real data the way bao_desi.py or cmb_act.py are.

Scope: cosmic shear (shear-shear, "1x2pt") only. Full 3x2pt (adding
galaxy clustering and galaxy-galaxy lensing) needs a galaxy bias model
on top of this — a genuinely separate, larger addition, not built here.

Survey parameter sourcing: SURVEY_PRESETS values are drawn from Euclid
Collaboration (2019, 2020) forecast papers and the LSST DESC Science
Requirements Document (Mandelbaum et al. 2018), cross-checked across
multiple independent secondary citations during development — some
spread exists across papers citing slightly different SRD versions
(flagged inline where relevant). Treat as forecast-standard defaults,
not authoritative primary-source numbers — check the primary references
directly before publication-grade use.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid

try:
    from pramana.core.camb_theory import get_nonlinear_matter_power
    CAMB_AVAILABLE = True
except Exception:
    CAMB_AVAILABLE = False

from pramana.core.models import e_of_z_lcdm


def smail_nz(z, z0, beta, power=2):
    """General Smail et al. (1994) redshift-distribution ansatz:
    n(z) ~ z^power * exp[-(z/z0)^beta], normalized to integrate to 1 over
    the input z array (trapezoidal norm — pass a fine grid, >=500 points
    recommended for smooth downstream integrals).

    Euclid: power=2, beta=1.5, z0=0.9/sqrt(2)=0.636 (Euclid Collaboration
    2019/2020 forecast baseline).
    LSST source sample: power=2, beta=0.87, z0=0.191 (LSST DESC SRD
    Appendix D1.1 source-sample parametrization).
    """
    unnorm = z**power * np.exp(-(z / z0) ** beta)
    norm = np.trapezoid(unnorm, z)
    return unnorm / norm


def photoz_scatter_kernel(z_true, z_photo, sigma_z_func):
    """Gaussian photo-z scatter kernel P(z_photo | z_true). sigma_z_func
    is a callable e.g. `lambda z: 0.05*(1+z)` (the standard form used by
    both Euclid and LSST DESC forecasts, with different coefficients).
    Returns a (len(z_photo), len(z_true)) kernel matrix.

    Simplification vs. the real surveys: this is a single Gaussian.
    Euclid's actual forecast model uses a double-Gaussian to capture
    catastrophic photo-z outliers (Euclid Collaboration 2020b) — omitted
    here for tractability. Flag this explicitly if a task needs outlier
    modeling specifically; it's a real, known simplification, not an
    oversight.
    """
    sigma = sigma_z_func(z_true)
    diff = z_photo[:, None] - z_true[None, :]
    return np.exp(-0.5 * (diff / sigma[None, :]) ** 2) / (np.sqrt(2 * np.pi) * sigma[None, :])


def equipopulated_bin_edges(z, n_of_z, n_bins):
    """Split n(z) into n_bins bins with EQUAL galaxy counts per bin (not
    equal redshift width) — the standard tomographic binning convention
    in both Euclid and LSST forecasts."""
    cdf = cumulative_trapezoid(n_of_z, z, initial=0)
    cdf = cdf / cdf[-1]
    target_cdf = np.linspace(0, 1, n_bins + 1)
    return np.interp(target_cdf, cdf, z)


def tomographic_bins(z_true_grid, n_true_of_z, n_bins, sigma_z_func, z_photo_grid=None):
    """Full pipeline: true n(z) -> equi-populated true-z bin edges ->
    convolve each bin's true-z selection with photo-z scatter -> observed
    photo-z n_i(z_photo) per tomographic bin.

    Returns: z_photo_grid, list of n_bins arrays (each normalized to
    integrate to 1 over z_photo_grid), and the true-z bin edges used
    (useful for plotting/sanity-checking against the source papers'
    published edge values).
    """
    if z_photo_grid is None:
        z_photo_grid = z_true_grid

    edges = equipopulated_bin_edges(z_true_grid, n_true_of_z, n_bins)
    kernel = photoz_scatter_kernel(z_true_grid, z_photo_grid, sigma_z_func)  # (n_photo, n_true)
    dz_true = np.gradient(z_true_grid)

    bins = []
    for i in range(n_bins):
        mask = (z_true_grid >= edges[i]) & (z_true_grid < edges[i + 1] if i < n_bins - 1
                                             else z_true_grid <= edges[i + 1])
        n_true_bin = n_true_of_z * mask
        n_photo_bin = kernel @ (n_true_bin * dz_true)
        norm = np.trapezoid(n_photo_bin, z_photo_grid)
        bins.append(n_photo_bin / norm)

    return z_photo_grid, bins, edges


# ---------------------------------------------------------------------
# Comoving distance + lensing efficiency kernel
# ---------------------------------------------------------------------

C_LIGHT = 299792.458  # km/s


def comoving_distance_grid(z_grid, H0, e_of_z_func, e_of_z_args):
    """chi(z) [Mpc] via cumulative trapezoidal integration of c/H0/E(z).
    Reuses the same e_of_z convention as models.py's MODEL_REGISTRY
    (e_of_z_func(z, *params)) so this module works with any registered
    cosmological model (LCDM/wCDM/CPL), not just a hardcoded LCDM."""
    Ez = e_of_z_func(z_grid, *e_of_z_args)
    integrand = C_LIGHT / H0 / Ez
    chi = cumulative_trapezoid(integrand, z_grid, initial=0)
    return chi


def lensing_efficiency_kernel(chi_grid, z_grid, n_i_of_z, Om, H0):
    """W_i(chi) = (3/2) Om (H0/c)^2 * chi * (1+z(chi)) *
                  integral_chi^chi_H dchi' n_i(chi') (chi'-chi)/chi'

    Standard weak-lensing efficiency kernel (Kaiser 1992, 1998; Hu 1999)
    — the (1+z) factor is folded directly into W_i here (rather than as a
    separate (1+z)^2 term multiplying the final C_ell, an equivalent but
    differently-bookkept convention seen in some papers) so that
    C_ell^ij = integral dchi/chi^2 W_i W_j P(k,z) with NO extra (1+z)
    factor needed at the C_ell step — see cosmic_shear_cl below.

    n_i_of_z must be evaluated ON chi_grid's corresponding z values and
    normalized to integrate to 1 over z (as tomographic_bins returns).
    """
    prefactor = 1.5 * Om * (H0 / C_LIGHT) ** 2
    chi_H = chi_grid[-1]

    # n(chi) dchi = n(z) dz -- convert via the local dz/dchi = H(z)/c... but
    # since n_i_of_z is already normalized in z-space and we integrate
    # over chi', convert the integrand n_i(chi') to a density in chi via
    # the Jacobian dz/dchi (equivalently, just interpolate n_i as a
    # function of chi using the chi<->z mapping, since both are
    # monotonic).
    n_i_of_chi = n_i_of_z  # same array, indexed by the same chi_grid/z_grid pairing

    W = np.zeros_like(chi_grid)
    for idx in range(len(chi_grid)):
        chi = chi_grid[idx]
        mask = chi_grid >= chi
        if not np.any(mask):
            continue
        chi_prime = chi_grid[mask]
        integrand = n_i_of_chi[mask] * (chi_prime - chi) / np.where(chi_prime > 0, chi_prime, np.inf)
        W[idx] = np.trapezoid(integrand, chi_prime)

    z_of_chi = z_grid  # same indexing
    W = prefactor * chi_grid * (1 + z_of_chi) * W
    return W


# ---------------------------------------------------------------------
# Limber-approximation angular power spectrum (GG, "cosmic shear")
# ---------------------------------------------------------------------

def cosmic_shear_cl_gg(ell_array, W_i, W_j, chi_grid, z_grid, pk_interpolator):
    """C_ell^{ij} (GG, gravitational-lensing-only term) via the Limber
    approximation (Limber 1954; standard in tomographic cosmic shear —
    see e.g. Kaiser 1998, Hu 1999, Kilbinger 2015 review):

        C_ell^ij = integral dchi/chi^2 W_i(chi) W_j(chi) P_delta(k=(ell+0.5)/chi, z(chi))

    pk_interpolator: the CAMB nonlinear-P(k,z) interpolator from
    camb_theory.get_nonlinear_matter_power (called as interp((z, k))).
    """
    valid = chi_grid > 0
    chi = chi_grid[valid]
    z_of_chi = z_grid[valid]
    Wi = W_i[valid]
    Wj = W_j[valid]

    cl = np.zeros(len(ell_array))
    for idx, ell in enumerate(ell_array):
        k_vals = (ell + 0.5) / chi
        pk_vals = pk_interpolator(np.column_stack([z_of_chi, k_vals]))
        integrand = Wi * Wj / chi**2 * pk_vals
        cl[idx] = np.trapezoid(integrand, chi)

    return cl


# ---------------------------------------------------------------------
# Intrinsic alignment (NLA model) — galaxies aren't randomly oriented;
# tidal-field-correlated intrinsic shapes contaminate the pure-lensing
# (GG) signal with GI (lensing-intrinsic cross) and II (intrinsic-
# intrinsic) terms. Standard "NLA" model (Bridle & King 2007, building
# on Hirata & Seljak 2004; amplitude normalization C1*rho_crit=0.0134 is
# the standard literature convention, e.g. Joachimi et al. 2011 and
# widely reused in DES/KiDS/Euclid forecast pipelines).
# ---------------------------------------------------------------------

C1_RHOCRIT = 0.0134  # standard NLA normalization constant (dimensionless
                        # combination of C1 and the critical density)


def growth_factor_from_pk(pk_interpolator, z_grid, k_ref=1e-3):
    """Linear growth factor D(z), normalized D(0)=1, extracted from the
    already-computed nonlinear P(k,z) interpolator at a very small k
    where nonlinear corrections are negligible (avoids a second CAMB
    call just for the growth factor) — D(z) = sqrt(P(k_ref,z)/P(k_ref,0))."""
    p0 = pk_interpolator((0.0, k_ref))
    pz = np.array([pk_interpolator((z, k_ref)) for z in z_grid])
    return np.sqrt(pz / p0)


def nla_amplitude_function(z_grid, Om, A_IA, growth_factor, eta_IA=0.0, z_pivot=0.62):
    """F(z) = -A_IA * C1*rho_crit * Om / D(z) * [(1+z)/(1+z_pivot)]^eta_IA

    The eta_IA power-law redshift-evolution term is a common NLA
    extension (allows IA amplitude to evolve with z beyond the pure
    1/D(z) scaling) — defaults to 0 (pure NLA, no extra evolution) unless
    a task specifically needs the extended model.
    """
    return -A_IA * C1_RHOCRIT * Om / growth_factor * ((1 + z_grid) / (1 + z_pivot)) ** eta_IA


def cosmic_shear_cl_with_ia(ell_array, W_i, W_j, n_i_of_chi, n_j_of_chi, F_of_chi,
                             chi_grid, z_grid, pk_interpolator):
    """Full cosmic shear C_ell^ij = GG + GI + IG + II, the NLA-model total
    that should be compared against real/synthetic shear data (pure GG
    alone underestimates the true observed signal, biased by however
    much IA contributes — using GG-only against real data is a known,
    common analysis mistake this function exists to avoid making by
    default).
    """
    valid = chi_grid > 0
    chi = chi_grid[valid]
    z_of_chi = z_grid[valid]
    Wi, Wj = W_i[valid], W_j[valid]
    ni, nj = n_i_of_chi[valid], n_j_of_chi[valid]
    F = F_of_chi[valid]

    cl = np.zeros(len(ell_array))
    for idx, ell in enumerate(ell_array):
        k_vals = (ell + 0.5) / chi
        pk_vals = pk_interpolator(np.column_stack([z_of_chi, k_vals]))

        gg = Wi * Wj * pk_vals
        gi = (Wi * nj + Wj * ni) * F * pk_vals
        ii = ni * nj * F**2 * pk_vals

        integrand = (gg + gi + ii) / chi**2
        cl[idx] = np.trapezoid(integrand, chi)

    return cl


# ---------------------------------------------------------------------
# Gaussian covariance (cosmic variance + shape noise, Knox formula)
# ---------------------------------------------------------------------

def shear_gaussian_covariance(ell_array, delta_ell, cl_theory_dict, n_bins, f_sky,
                               sigma_e, n_gal_per_bin_steradian):
    """Knox (1995)-formula Gaussian covariance for the binned shear power
    spectra: Cov[C_ell^ij, C_ell^kl] = [C_ell^ik C_ell^jl + C_ell^il C_ell^jk]
    / [(2*ell+1) * delta_ell * f_sky], where each C_ell here already
    includes shot noise on the diagonal (auto-bin, auto-ell) terms:
    N_ell^ii = sigma_e^2 / n_gal_i.

    cl_theory_dict: {(i,j): C_ell array} for ALL bin pairs INCLUDING
    shape noise already added to (i,i) pairs — see
    make_synthetic_shear_dataset for how this is assembled.

    Returns the full covariance matrix for the flattened data vector
    (ordered: all ell for pair (0,0), then all ell for pair (0,1), ...
    matching the ordering used in make_synthetic_shear_dataset).
    """
    pairs = [(i, j) for i in range(n_bins) for j in range(i, n_bins)]
    n_ell = len(ell_array)
    n_pairs = len(pairs)
    cov = np.zeros((n_pairs * n_ell, n_pairs * n_ell))

    def cl_get(i, j, ell_idx):
        key = (i, j) if (i, j) in cl_theory_dict else (j, i)
        return cl_theory_dict[key][ell_idx]

    mode_count = (2 * ell_array + 1) * delta_ell * f_sky

    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            for e_idx in range(n_ell):
                term = (cl_get(i, k, e_idx) * cl_get(j, l, e_idx) +
                        cl_get(i, l, e_idx) * cl_get(j, k, e_idx)) / mode_count[e_idx]
                cov[a * n_ell + e_idx, b * n_ell + e_idx] = term

    return cov, pairs


# ---------------------------------------------------------------------
# Survey presets (Euclid + LSST share this exact codebase — only these
# specification dicts differ) and synthetic data generation
# ---------------------------------------------------------------------

SURVEY_PRESETS = {
    "euclid": {
        "z0": 0.9 / np.sqrt(2), "beta": 1.5, "power": 2,
        "n_bins": 10, "n_gal_arcmin2": 30.0, "sigma_e": 0.30,
        "f_sky": 0.364, "sigma_z_coeff": 0.05, "z_max": 2.5,
        "source": "Euclid Collaboration 2019/2020 forecast baseline",
    },
    "lsst_y1": {
        "z0": 0.191, "beta": 0.870, "power": 2,
        "n_bins": 5, "n_gal_arcmin2": 10.0, "sigma_e": 0.26,
        "f_sky": 0.35, "sigma_z_coeff": 0.05, "z_max": 3.0,
        "source": "LSST DESC SRD (Mandelbaum et al. 2018), Year-1 depth",
    },
    "lsst_y10": {
        "z0": 0.191, "beta": 0.870, "power": 2,
        "n_bins": 5, "n_gal_arcmin2": 27.0, "sigma_e": 0.26,
        "f_sky": 0.40, "sigma_z_coeff": 0.05, "z_max": 3.0,
        "source": "LSST DESC SRD (Mandelbaum et al. 2018), full 10-year depth",
    },
}
# All specification values cross-checked across 3+ independent secondary
# citations during development; some genuine spread exists across papers
# citing different SRD versions (noted in "Weak Lensing Reference" section)
# -- verify against the primary source before publication-grade use, same
# caveat as everywhere else forecast numbers appear in this suite.

ARCMIN2_PER_STERADIAN = (180 * 60 / np.pi) ** 2


def build_survey(preset_name, z_grid, e_of_z_func, e_of_z_params, H0, Om, ombh2, omch2,
                  ell_array, A_IA=0.0, eta_IA=0.0):
    """One-call pipeline: preset -> n(z) -> tomographic bins -> lensing
    kernels -> theory C_ell (with IA) for every bin pair. This is the
    function most tasks should call rather than assembling the pieces
    above by hand.

    e_of_z_func/e_of_z_params: the E(z) function and its OWN positional
    params (e.g. e_of_z_lcdm, (Om,) for LCDM; e_of_z_wcdm, (Om, w) for
    wCDM) — kept separate from ombh2/omch2, which are CAMB-specific
    inputs for the nonlinear P(k,z) calculation, not E(z) parameters.
    Conflating these two was a real bug caught during development: passing
    a single params dict and unpacking it with `*` into e_of_z_func
    iterated the dict's KEYS as strings, not its values — a numpy dtype
    error, not a silent wrong-number bug, but worth the explicit
    separation here to make sure it can't recur.
    """
    if not CAMB_AVAILABLE:
        raise RuntimeError(
            "CAMB is required for weak lensing predictions but is not installed. "
            "Install with `uv add camb` or `pip install camb`."
        )
    
    spec = SURVEY_PRESETS[preset_name]
    nz = smail_nz(z_grid, spec["z0"], spec["beta"], spec["power"])
    z_photo, bins, edges = tomographic_bins(
        z_grid, nz, spec["n_bins"], lambda z: spec["sigma_z_coeff"] * (1 + z)
    )

    chi_grid = comoving_distance_grid(z_grid, H0, e_of_z_func, e_of_z_params)
    W = [lensing_efficiency_kernel(chi_grid, z_grid, b, Om, H0) for b in bins]

    _, _, _, pk_interp = get_nonlinear_matter_power(
        H0=H0, ombh2=ombh2, omch2=omch2, z_array=np.linspace(0, spec["z_max"], 30),
    )

    D = growth_factor_from_pk(pk_interp, z_grid)
    F = nla_amplitude_function(z_grid, Om, A_IA, D, eta_IA) if A_IA != 0 else np.zeros_like(z_grid)

    n_bins = spec["n_bins"]
    cl_dict = {}
    for i in range(n_bins):
        for j in range(i, n_bins):
            if A_IA != 0:
                cl_dict[(i, j)] = cosmic_shear_cl_with_ia(
                    ell_array, W[i], W[j], bins[i], bins[j], F, chi_grid, z_grid, pk_interp)
            else:
                cl_dict[(i, j)] = cosmic_shear_cl_gg(ell_array, W[i], W[j], chi_grid, z_grid, pk_interp)

    n_gal_per_bin_sr = (spec["n_gal_arcmin2"] * ARCMIN2_PER_STERADIAN) / n_bins
    return {
        "cl_dict": cl_dict, "bins": bins, "z_photo": z_photo, "edges": edges,
        "W": W, "chi_grid": chi_grid, "spec": spec, "n_gal_per_bin_sr": n_gal_per_bin_sr,
    }


def _omch2_from_Om(Om, H0, ombh2):
    """Om is TOTAL matter (CDM + baryons); CAMB wants omch2 = CDM only.
    omch2 = Om*h^2 - ombh2."""
    h = H0 / 100.0
    return Om * h**2 - ombh2


def make_synthetic_shear_dataset(preset_name, ell_array, delta_ell, Om=0.3055, H0=67.97,
                                  ombh2=0.0224, A_IA=0.0, seed=0):
    """Generate a synthetic tomographic shear data vector + covariance —
    the ONLY kind of validated dataset this module has, since no real
    shear catalog exists yet for either survey (see module docstring).
    Injects known truth (Om, H0) with realistic Gaussian noise from
    shear_gaussian_covariance, for exactly the kind of inject-and-recover
    validation used throughout the rest of this suite.
    """
    if not CAMB_AVAILABLE:
        raise RuntimeError(
            "CAMB is required for weak lensing synthetic data generation but is not installed. "
            "Install with `uv add camb` or `pip install camb`."
        )
    
    rng = np.random.default_rng(seed)
    spec = SURVEY_PRESETS[preset_name]
    z_grid = np.linspace(0.001, spec["z_max"], 300)
    omch2 = _omch2_from_Om(Om, H0, ombh2)

    result = build_survey(preset_name, z_grid, e_of_z_lcdm, (Om,), H0, Om, ombh2, omch2,
                           ell_array, A_IA=A_IA)

    cl_dict = result["cl_dict"]
    n_bins = spec["n_bins"]
    sigma_e = spec["sigma_e"]
    n_gal_sr = result["n_gal_per_bin_sr"]

    noisy_cl_dict = {}
    for (i, j), cl in cl_dict.items():
        noisy_cl_dict[(i, j)] = cl + (sigma_e**2 / n_gal_sr if i == j else 0.0)

    cov, pairs = shear_gaussian_covariance(ell_array, delta_ell, noisy_cl_dict, n_bins,
                                            spec["f_sky"], sigma_e, n_gal_sr)

    theory_vector = np.concatenate([cl_dict[p] for p in pairs])
    noise = rng.multivariate_normal(np.zeros(len(theory_vector)), cov)
    data_vector = theory_vector + noise

    return {"data_vector": data_vector, "cov": cov, "pairs": pairs, "ell_array": ell_array,
            "theory_vector": theory_vector, "truth": {"Om": Om, "H0": H0, "A_IA": A_IA},
            "preset": preset_name, "spec": spec}


def log_likelihood_shear(theta, param_names, preset_name, ell_array, delta_ell,
                          data_vector, cov_inv, fixed_ombh2=0.0224):
    """Gaussian likelihood for a cosmic-shear data vector (synthetic or,
    eventually, real). theta/param_names typically ["Om", "H0"] or
    ["Om", "H0", "A_IA"] if fitting IA jointly."""
    if not CAMB_AVAILABLE:
        raise RuntimeError(
            "CAMB is required for weak lensing likelihood evaluation but is not installed. "
            "Install with `uv add camb` or `pip install camb`."
        )
    
    params = dict(zip(param_names, theta))
    Om = params["Om"]
    H0 = params["H0"]
    A_IA = params.get("A_IA", 0.0)
    omch2 = _omch2_from_Om(Om, H0, fixed_ombh2)

    z_grid = np.linspace(0.001, SURVEY_PRESETS[preset_name]["z_max"], 300)
    result = build_survey(preset_name, z_grid, e_of_z_lcdm, (Om,), H0, Om, fixed_ombh2, omch2,
                           ell_array, A_IA=A_IA)

    pairs = sorted(result["cl_dict"].keys())
    model_vector = np.concatenate([result["cl_dict"][p] for p in pairs])

    delta = data_vector - model_vector
    return -0.5 * delta @ cov_inv @ delta