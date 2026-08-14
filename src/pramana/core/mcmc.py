"""Run an emcee MCMC fit for a cosmological model against SN Ia data.

As a library:
    from pramana.core.mcmc import run_fit
    sampler, flat_chain = run_fit("lcdm", z, mb_obs, cov)

As a CLI smoke test (synthetic data, no real files needed):
    python -m pramana.core.mcmc --model lcdm --synthetic --nsteps 2000

As a CLI run on real Pantheon+ data:
    python -m pramana.core.mcmc --model cpl \\
        --data Pantheon+SH0ES.dat --cov Pantheon+SH0ES_STAT+SYS.cov \\
        --nwalkers 32 --nsteps 8000 --out cpl_chain.npz
"""
import argparse

import numpy as np
import emcee

from pramana.core.models import MODEL_REGISTRY
from pramana.core.likelihood import log_probability
from pramana.core.data_io import load_pantheon, make_synthetic_dataset


def run_fit(
    model_name: str,
    z: np.ndarray,
    mb_obs: np.ndarray,
    cov: np.ndarray,
    nwalkers: int = 32,
    nsteps: int = 4000,
    seed: int = 42,
    progress: bool = True,
) -> emcee.EnsembleSampler:
    """Run emcee MCMC and return the sampler."""
    spec = MODEL_REGISTRY[model_name]
    param_names = spec["params"]
    priors = spec["priors"]
    ndim = len(param_names)

    cov_inv = np.linalg.inv(cov)

    rng = np.random.default_rng(seed)
    p0_center = np.array([np.mean(priors[p]) for p in param_names])
    p0_spread = np.array([(priors[p][1] - priors[p][0]) * 0.05 for p in param_names])
    p0 = p0_center + p0_spread * rng.normal(size=(nwalkers, ndim))

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(z, mb_obs, cov_inv, spec["func"], param_names, priors),
    )
    sampler.run_mcmc(p0, nsteps, progress=progress)
    return sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lcdm", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--data", default=None, help="Path to Pantheon+SH0ES.dat")
    ap.add_argument("--cov", default=None, help="Path to Pantheon+SH0ES_STAT+SYS.cov")
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    ap.add_argument("--nwalkers", type=int, default=32)
    ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--out", default="chain.npz")
    args = ap.parse_args()

    if args.synthetic or args.data is None:
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(args.data, args.cov)

    sampler = run_fit(args.model, z, mb_obs, cov, nwalkers=args.nwalkers, nsteps=args.nsteps)

    burn_in = int(args.nsteps * 0.3)
    flat_chain = sampler.get_chain(discard=burn_in, flat=True)
    np.savez(args.out, chain=flat_chain, params=MODEL_REGISTRY[args.model]["params"])
    print(f"\nSaved {flat_chain.shape[0]} post-burn-in samples to {args.out}")


if __name__ == "__main__":
    main()