"""Simulation-based inference (SBI, aka likelihood-free inference) via the
`sbi` package (Tejero-Cantero et al. 2020). Trains a neural density
estimator on (parameter, simulated-data) pairs, then infers a posterior for
REAL data without ever writing down an explicit likelihood function.

Use this when the data-generating process is easy to simulate but the
likelihood is intractable or unknown in closed form — e.g. a forward model
with complex instrumental systematics, selection effects, or non-Gaussian
noise where likelihood.py's clean analytic marginalization doesn't apply.
For the closed-form-likelihood cases already in this suite (SN, BAO), NUTS
or nested sampling are more efficient and give exact (not amortized-
approximate) posteriors — reach for SBI specifically when a task's
likelihood genuinely can't be written down, not as a default replacement.
"""
import numpy as np
import torch
from sbi.inference import NPE
from sbi.utils import BoxUniform


def make_simulator(model_func, z: np.ndarray, mb_err: np.ndarray):
    """Wrap a deterministic model into a stochastic simulator: draws one
    noisy realization of the data given theta, the way a real observation
    would look. This is the thing SBI trains on instead of a likelihood
    formula — swap in a more realistic noise model here (non-Gaussian
    outliers, selection cuts, etc.) and SBI adapts with zero other code
    changes, unlike an analytic likelihood which would need re-deriving.
    """
    def simulator(theta):
        theta_np = theta.numpy() if torch.is_tensor(theta) else np.asarray(theta)
        mu = model_func(z, *theta_np)
        noisy = mu + np.random.normal(0, mb_err, size=mu.shape)
        return torch.as_tensor(noisy, dtype=torch.float32)

    return simulator


def train_npe(
    simulator,
    prior_bounds: dict[str, tuple[float, float]],
    param_names: list[str],
    n_simulations: int = 2000,
    seed: int = 0,
):
    """Train a Neural Posterior Estimator. prior_bounds: {param: (lo, hi)}.
    n_simulations trades training cost for posterior accuracy — 2000 is a
    reasonable smoke-test/exploratory budget; production use typically
    wants 10^4-10^5+."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    los = torch.tensor([prior_bounds[p][0] for p in param_names], dtype=torch.float32)
    his = torch.tensor([prior_bounds[p][1] for p in param_names], dtype=torch.float32)
    prior = BoxUniform(low=los, high=his)

    inference = NPE(prior=prior)

    theta_train = prior.sample((n_simulations,))
    x_train = torch.stack([simulator(t) for t in theta_train])

    inference = inference.append_simulations(theta_train, x_train)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)

    return posterior


def sample_posterior(posterior, x_observed: np.ndarray, n_samples: int = 5000, seed: int = 0) -> np.ndarray:
    """Sample the amortized posterior for a real (or synthetic-test)
    observation x_observed. This step is fast (no new simulations needed)
    — the expensive part was training; every subsequent dataset reuses the
    same trained posterior."""
    torch.manual_seed(seed)
    x_observed = torch.as_tensor(np.asarray(x_observed), dtype=torch.float32)
    samples = posterior.sample((n_samples,), x=x_observed)
    return samples.numpy()


def validate_on_synthetic(posterior, simulator, theta_true: np.ndarray, param_names: list[str], n_samples: int = 3000):
    """Coverage check: simulate one fake observation at a KNOWN theta_true,
    infer the posterior, and confirm theta_true falls inside the recovered
    credible interval. The standard SBI sanity check before trusting the
    posterior on real data — a systematically biased or overconfident NPE
    will fail this even though training loss looked fine."""
    x_fake = simulator(torch.tensor(theta_true, dtype=torch.float32))
    samples = sample_posterior(posterior, x_fake, n_samples=n_samples)

    print(f"{'param':<8} {'true':>8} {'post. median':>14} {'68% interval':>20}")
    for i, name in enumerate(param_names):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        inside = "OK" if lo <= theta_true[i] <= hi else "OUTSIDE 68% CI"
        print(f"{name:<8} {theta_true[i]:>8.3f} {med:>14.3f}   [{lo:.3f}, {hi:.3f}]  {inside}")

    return samples