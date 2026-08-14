"""Validation tests for PRAMANA - encoding skill file's verified numbers."""
import numpy as np
import pytest

from pramana.core.models import MODEL_REGISTRY, distance_modulus_lcdm
from pramana.core.data_io import make_synthetic_dataset
from pramana.core.likelihood import log_likelihood
from pramana.core.mcmc import run_fit
from pramana.core.nested_sampling import run_nested, bayes_factor
from pramana.core.fisher_forecast import fisher_matrix_gaussian, forecast_errors, compare_to_mcmc
from pramana.core.profile_likelihood import profile_scan, confidence_interval_from_profile
from pramana.core.gp_emulator import latin_hypercube_design, train_emulator, validate_emulator
from pramana.core.sbi_inference import make_simulator, train_npe, sample_posterior, validate_on_synthetic
from pramana.core.data_compression import moped_vectors, compare_compressed_vs_full
from pramana.core.joint_likelihood import build_joint_log_probability, per_probe_chi2
from pramana.core.bao_desi import log_likelihood_bao


def test_synthetic_data_generation():
    """Test synthetic data generation works."""
    z, mb_obs, cov = make_synthetic_dataset(n=100, seed=42)
    assert len(z) == 100
    assert len(mb_obs) == 100
    assert cov.shape == (100, 100)
    assert np.all(np.diag(cov) > 0)


def test_model_registry():
    """Test all models in registry work."""
    z = np.array([0.1, 0.5, 1.0])
    for name, spec in MODEL_REGISTRY.items():
        # Test with default priors center
        theta = np.array([np.mean(spec["priors"][p]) for p in spec["params"]])
        mu = spec["func"](z, *theta)
        assert len(mu) == 3
        assert np.all(np.isfinite(mu))


def test_likelihood_evaluation():
    """Test likelihood evaluation."""
    z, mb_obs, cov = make_synthetic_dataset(n=50, seed=42)
    cov_inv = np.linalg.inv(cov)
    spec = MODEL_REGISTRY["lcdm"]
    theta = np.array([0.3])
    ll = log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], spec["params"])
    assert np.isfinite(ll)


def test_mcmc_smoke():
    """Smoke test MCMC runs."""
    z, mb_obs, cov = make_synthetic_dataset(n=50, seed=42)
    sampler = run_fit("lcdm", z, mb_obs, cov, nwalkers=8, nsteps=100, progress=False)
    chain = sampler.get_chain(discard=30, flat=True)
    assert chain.shape[0] > 0
    assert chain.shape[1] == 1  # LCDM has 1 param


def test_nested_sampling_smoke():
    """Smoke test nested sampling."""
    z, mb_obs, cov = make_synthetic_dataset(n=50, seed=42)
    cov_inv = np.linalg.inv(cov)
    spec = MODEL_REGISTRY["lcdm"]

    def loglike(theta):
        return log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], spec["params"])

    results = run_nested(loglike, spec["params"], spec["priors"], nlive=50)
    assert np.isfinite(results.logz[-1])
    assert results.logzerr[-1] > 0


def test_fisher_forecast():
    """Test Fisher matrix computation."""
    z, mb_obs, cov = make_synthetic_dataset(n=50, seed=42)
    cov_inv = np.linalg.inv(cov)
    spec = MODEL_REGISTRY["wcdm"]
    param_names = spec["params"]
    theta_fid = np.array([0.3, -1.0])

    def model_predictions(theta):
        params = dict(zip(param_names, theta))
        return spec["func"](z, **params)

    fisher = fisher_matrix_gaussian(model_predictions, theta_fid, cov_inv)
    assert fisher.shape == (2, 2)
    assert np.all(np.linalg.eigvalsh(fisher) > 0)

    errs, cov_mat = forecast_errors(fisher, param_names)
    assert len(errs) == 2
    assert all(v > 0 for v in errs.values())


def test_profile_likelihood():
    """Test profile likelihood."""
    z, mb_obs, cov = make_synthetic_dataset(n=50, seed=42)
    cov_inv = np.linalg.inv(cov)
    spec = MODEL_REGISTRY["lcdm"]

    def neg_log_likelihood(theta):
        return -log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], spec["params"])

    bounds = {"Om": spec["priors"]["Om"]}
    # Use wider scan range to ensure best fit is interior
    scan_vals = np.linspace(0.05, 0.6, 20)
    scan_vals, profile_nll, _ = profile_scan(
        neg_log_likelihood, spec["params"], "Om", scan_vals, bounds
    )
    best, lo, hi = confidence_interval_from_profile(scan_vals, profile_nll)
    # Best should be within scan range (not necessarily strictly between lo/hi if at boundary)
    assert scan_vals.min() <= best <= scan_vals.max()
    assert lo <= best <= hi


def test_gp_emulator():
    """Test GP emulator training and validation."""
    z, mb_obs, cov = make_synthetic_dataset(n=50, seed=42)
    spec = MODEL_REGISTRY["lcdm"]
    bounds = [spec["priors"]["Om"]]

    theta_train = latin_hypercube_design(bounds, 50, seed=1)
    y_train = np.array([spec["func"](z, *theta) for theta in theta_train])

    emulator = train_emulator(theta_train, y_train)

    theta_test = latin_hypercube_design(bounds, 20, seed=2)
    y_test = np.array([spec["func"](z, *theta) for theta in theta_test])

    rel_err = validate_emulator(emulator, theta_test, y_test)
    assert rel_err.max() < 0.1  # Should be very accurate for simple 1D function


def test_sbi_smoke():
    """Smoke test SBI (fast with few sims)."""
    z, mb_obs, cov = make_synthetic_dataset(n=30, seed=42)
    spec = MODEL_REGISTRY["lcdm"]
    priors = spec["priors"]
    mb_err = np.sqrt(np.diag(cov))

    simulator = make_simulator(spec["func"], z, mb_err)
    posterior = train_npe(simulator, priors, spec["params"], n_simulations=100, seed=42)
    samples = sample_posterior(posterior, mb_obs, n_samples=100, seed=42)

    assert samples.shape == (100, 1)
    assert np.all(np.isfinite(samples))


def test_moped_compression():
    """Test MOPED compression accuracy using RAW Gaussian likelihood.

    Note: MOPED is designed for raw Gaussian likelihood (chi2 = delta^T Cinv delta),
    NOT the analytically marginalized SN likelihood. The skill file documents this
    caveat - comparing MOPED-compressed vs marginalized likelihood shows large
    discrepancies. This test uses the matching raw likelihood.
    """
    z, mb_obs, cov = make_synthetic_dataset(n=100, seed=42)
    cov_inv = np.linalg.inv(cov)
    spec = MODEL_REGISTRY["lcdm"]
    theta_fid = np.array([0.3])

    def model_predictions(theta):
        return spec["func"](z, **{"Om": theta[0]})

    def raw_loglike(theta):
        """Raw Gaussian likelihood: -0.5 * (data-model)^T Cinv (data-model)"""
        params = dict(zip(spec["params"], theta))
        mu = spec["func"](z, **params)
        delta = mb_obs - mu
        return -0.5 * (delta @ cov_inv @ delta)

    B = moped_vectors(model_predictions, theta_fid, cov_inv)
    assert B.shape == (1, len(z))

    y_compressed = B @ mb_obs
    test_points = theta_fid + np.random.normal(0, 0.002, size=(5, 1))
    B2, full_diff, comp_diff = compare_compressed_vs_full(
        model_predictions, theta_fid, cov_inv, mb_obs, test_points, raw_loglike
    )
    max_diff = np.max(np.abs(full_diff - comp_diff))
    assert max_diff < 0.1  # Should be << 1 for matching likelihood


def test_joint_likelihood():
    """Test joint SN+BAO likelihood."""
    z, mb_obs, cov = make_synthetic_dataset(n=50, seed=42)
    cov_inv = np.linalg.inv(cov)

    probes = [
        {"kind": "sn", "z": z, "mb_obs": mb_obs, "cov_inv": cov_inv},
        {"kind": "bao", "H0": 70.0, "rd_mode": "eh98"},
    ]

    log_prob, param_names = build_joint_log_probability("lcdm", probes)
    spec = MODEL_REGISTRY["lcdm"]
    theta = np.array([0.3])

    lp = log_prob(theta)
    assert np.isfinite(lp)

    # Per-probe chi2
    per_probe_chi2("lcdm", theta, probes)


def test_bao_likelihood():
    """Test BAO likelihood evaluation."""
    spec = MODEL_REGISTRY["lcdm"]
    theta = np.array([0.3])

    ll = log_likelihood_bao(theta, spec["e_of_z"], spec["params"], H0=70.0, rd_mode="eh98")
    assert np.isfinite(ll)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])