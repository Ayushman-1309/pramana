"""PRAMANA — Unified Cosmological Inference Suite

Sanskrit *pramāṇa*: a means of valid knowledge — the epistemological question of
how you actually justify that something is true. A cosmological inference suite
spanning SN Ia (Pantheon+), BAO (DESI DR2), CMB lensing + primary (ACT DR6), and
JWST-era probes (high-z SNe, H0 tension, S8 tension).
"""

__version__ = "0.1.0"

from pramana.core.models import MODEL_REGISTRY, distance_modulus_lcdm, distance_modulus_wcdm, distance_modulus_cpl
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.likelihood import log_likelihood, log_prior, log_probability
from pramana.core.mcmc import run_fit
from pramana.core.diagnostics import summarize, gelman_rubin
from pramana.core.plotting import corner_plot, getdist_triangle, compare_hubble_diagram
from pramana.core.bao_desi import (
    DESI_DR2_BAO_TABLE,
    build_data_vector_and_cov,
    sound_horizon_rd,
    model_predictions,
    log_likelihood_bao,
)
from pramana.core.jwst_probes import (
    append_supernovae,
    H0_MEASUREMENTS,
    S8_MEASUREMENTS,
    h0_tension_sigma,
    s8_tension_sigma,
    plot_h0_whisker,
    plot_s8_whisker,
)
from pramana.core.nested_sampling import run_nested, equal_weight_posterior, bayes_factor
from pramana.core.fisher_forecast import (
    fisher_matrix_gaussian,
    forecast_errors,
    figure_of_merit,
    fisher_ellipse,
    compare_to_mcmc,
)
from pramana.core.profile_likelihood import profile_scan, confidence_interval_from_profile, global_best_fit
from pramana.core.gp_emulator import latin_hypercube_design, train_emulator, emulate, validate_emulator
from pramana.core.sbi_inference import make_simulator, train_npe, sample_posterior, validate_on_synthetic
from pramana.core.importance_resampling import importance_weights, effective_sample_size, reweight_chain, weighted_quantiles, resample_to_equal_weight
from pramana.core.data_compression import moped_vectors, compress, compressed_log_likelihood, compare_compressed_vs_full
from pramana.core.joint_likelihood import build_joint_log_probability, per_probe_chi2

__all__ = [
    # Models
    "MODEL_REGISTRY",
    "distance_modulus_lcdm",
    "distance_modulus_wcdm",
    "distance_modulus_cpl",
    # Data I/O
    "load_pantheon",
    "make_synthetic_dataset",
    # Likelihood
    "log_likelihood",
    "log_prior",
    "log_probability",
    # MCMC
    "run_fit",
    # Diagnostics
    "summarize",
    "gelman_rubin",
    # Plotting
    "corner_plot",
    "getdist_triangle",
    "compare_hubble_diagram",
    # BAO DESI
    "DESI_DR2_BAO_TABLE",
    "build_data_vector_and_cov",
    "sound_horizon_rd",
    "model_predictions",
    "log_likelihood_bao",
    # JWST Probes
    "append_supernovae",
    "H0_MEASUREMENTS",
    "S8_MEASUREMENTS",
    "h0_tension_sigma",
    "s8_tension_sigma",
    "plot_h0_whisker",
    "plot_s8_whisker",
    # Nested Sampling
    "run_nested",
    "equal_weight_posterior",
    "bayes_factor",
    # Fisher Forecast
    "fisher_matrix_gaussian",
    "forecast_errors",
    "figure_of_merit",
    "fisher_ellipse",
    "compare_to_mcmc",
    # Profile Likelihood
    "profile_scan",
    "confidence_interval_from_profile",
    "global_best_fit",
    # GP Emulator
    "latin_hypercube_design",
    "train_emulator",
    "emulate",
    "validate_emulator",
    # SBI
    "make_simulator",
    "train_npe",
    "sample_posterior",
    "validate_on_synthetic",
    # Importance Resampling
    "importance_weights",
    "effective_sample_size",
    "reweight_chain",
    "weighted_quantiles",
    "resample_to_equal_weight",
    # Data Compression
    "moped_vectors",
    "compress",
    "compressed_log_likelihood",
    "compare_compressed_vs_full",
    # Joint Likelihood
    "build_joint_log_probability",
    "per_probe_chi2",
]