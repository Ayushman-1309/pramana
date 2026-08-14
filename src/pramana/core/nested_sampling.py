"""Nested sampling via dynesty — Bayesian evidence (Z) for model comparison,
not just parameter posteriors. Where MCMC tells you "given this model, what
parameters fit best," nested sampling tells you "how much better does this
model explain the data than that one," via the Bayes factor K = Z1/Z2.

Use this over MCMC when the actual question is model selection (does CPL
beat LambdaCDM once you penalize its extra parameters?), not just parameter
estimation, or when the posterior is multimodal (dynesty handles this far
better than emcee's affine-invariant walkers).
"""
import numpy as np
import dynesty


def prior_transform_factory(param_names: list[str], priors: dict[str, tuple[float, float]]):
    """Build the unit-cube -> physical-parameter transform dynesty needs
    (uniform prior on each param between its registered bounds)."""
    los = np.array([priors[p][0] for p in param_names])
    his = np.array([priors[p][1] for p in param_names])
    widths = his - los

    def prior_transform(u):
        return los + u * widths

    return prior_transform


def run_nested(
    loglike_func,
    param_names: list[str],
    priors: dict[str, tuple[float, float]],
    loglike_args: tuple = (),
    nlive: int = 500,
    sample: str = "auto",
    dlogz: float = 0.1,
    seed: int = 42,
):
    """Run dynesty's static nested sampler.

    loglike_func(theta, *loglike_args) -> log-likelihood (NOT log-prior +
    log-likelihood — dynesty's prior_transform handles the prior, unlike
    the emcee convention in likelihood.py::log_probability which combines
    both). If reusing an existing log_probability function that adds a
    flat-prior log_prior, that's fine — it evaluates to 0 inside the
    (already-bounded) unit-cube transform and just costs a wasted prior
    check per call.

    Returns the dynesty results object: results.logz[-1] is ln(evidence),
    results.logzerr[-1] its uncertainty; results.samples + results.logwt
    give posterior samples via dynesty.utils.resample_equal.
    """
    ndim = len(param_names)
    prior_transform = prior_transform_factory(param_names, priors)

    rng = np.random.default_rng(seed)
    sampler = dynesty.NestedSampler(
        loglike_func, prior_transform, ndim,
        logl_args=loglike_args, nlive=nlive, sample=sample, rstate=rng,
    )
    sampler.run_nested(dlogz=dlogz, print_progress=False)
    return sampler.results


def equal_weight_posterior(results) -> np.ndarray:
    """Convert dynesty's weighted samples into an equal-weight posterior
    sample, directly comparable to an emcee flat_chain."""
    from dynesty.utils import resample_equal

    weights = np.exp(results.logwt - results.logz[-1])
    return resample_equal(results.samples, weights)


def bayes_factor(results_a, results_b, name_a: str = "model A", name_b: str = "model B"):
    """ln(K) = ln(Z_a) - ln(Z_b), plus the standard Jeffreys-scale reading.
    K > 1 favors model A; the Jeffreys scale below is the conventional
    (Kass & Raftery 1995) interpretation, not a hard statistical test."""
    lnZ_a, lnZ_b = results_a.logz[-1], results_b.logz[-1]
    err_a, err_b = results_a.logzerr[-1], results_b.logzerr[-1]
    ln_K = lnZ_a - lnZ_b
    err_K = np.sqrt(err_a**2 + err_b**2)

    abs_lnK = abs(ln_K)
    if abs_lnK < 1.0:
        strength = "inconclusive"
    elif abs_lnK < 2.5:
        strength = "weak"
    elif abs_lnK < 5.0:
        strength = "moderate"
    else:
        strength = "strong"

    favored = name_a if ln_K > 0 else name_b
    print(f"ln(Z_{name_a}) = {lnZ_a:.2f} +/- {err_a:.2f}")
    print(f"ln(Z_{name_b}) = {lnZ_b:.2f} +/- {err_b:.2f}")
    print(f"ln(K) = {ln_K:.2f} +/- {err_K:.2f}  -> {strength} evidence for {favored}")
    return ln_K, err_K