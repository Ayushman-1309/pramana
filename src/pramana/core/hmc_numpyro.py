"""Hamiltonian Monte Carlo / NUTS via numpyro, using the JAX models in
differentiable_models.py. Where emcee explores the posterior with a
gradient-free random walk (needs many walkers, can mix slowly in high
dimensions or strong degeneracies), NUTS uses the gradient to make long,
informed jumps — typically far fewer effective samples needed per unit of
autocorrelation time, especially valuable once you're in a joint
SN+BAO+CMB fit with 8-12+ parameters where emcee starts struggling.

Trade-off: requires everything in the model to be JAX-differentiable
(hence the separate differentiable_models.py) — CAMB and dynesty aren't,
so this doesn't replace those, it's the right tool specifically for the
SN/BAO-style closed-form-likelihood fits.
"""
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def build_sn_model(
    z: np.ndarray,
    mb_obs: np.ndarray,
    cov_inv: np.ndarray,
    model_name: str,
    priors: dict[str, tuple[float, float]],
):
    """Build a numpyro probabilistic model for the marginalized SN
    likelihood (same physics as likelihood.py, reimplemented in JAX so
    NUTS can differentiate through it)."""
    from pramana.core.differentiable_models import JAX_MODEL_REGISTRY

    spec = JAX_MODEL_REGISTRY[model_name]
    param_names = spec["params"]
    cov_inv = jnp.asarray(cov_inv)
    mb_obs = jnp.asarray(mb_obs)
    z = jnp.asarray(z)
    ones = jnp.ones_like(mb_obs)

    def model():
        theta = [numpyro.sample(p, dist.Uniform(*priors[p])) for p in param_names]
        mu_model = spec["func"](z, *theta)
        delta = mb_obs - mu_model

        A = delta @ cov_inv @ delta
        B = ones @ cov_inv @ delta
        C = ones @ cov_inv @ ones
        chi2 = A - (B**2) / C

        numpyro.factor("marginalized_sn_likelihood", -0.5 * chi2)

    return model, param_names


def run_nuts(
    model,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 2,
    seed: int = 0,
):
    """Run NUTS. num_chains > 1 lets you compute a real Gelman-Rubin R-hat
    from independent chains (numpyro reports this automatically in
    mcmc.print_summary())."""
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed))
    return mcmc


def samples_to_flat_chain(mcmc, param_names: list[str]) -> np.ndarray:
    """Convert numpyro's dict-of-arrays samples into the same (N, ndim)
    flat-chain shape used everywhere else in this suite (corner_plot,
    getdist_triangle, compare_to_mcmc, etc.), so NUTS output is a drop-in
    replacement for an emcee flat_chain."""
    import numpy as np

    samples = mcmc.get_samples()
    return np.stack([np.asarray(samples[p]) for p in param_names], axis=1)