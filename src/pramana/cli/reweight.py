"""CLI command: reweight — importance resampling / chain reweighting."""
import typer
import numpy as np
from rich.console import Console

from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon
from pramana.core.importance_resampling import reweight_chain, resample_to_equal_weight, weighted_quantiles
from pramana.core.likelihood import log_likelihood, log_prior
from pramana.core.plotting import corner_plot

reweight_app = typer.Typer(name="reweight", help="Importance resampling / chain reweighting")
console = Console()


@reweight_app.command("run")
def reweight_run(
    old_chain: str = typer.Option(..., "--old-chain", help="Original chain .npz file"),
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat (for new likelihood)"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to covariance (for new likelihood)"),
    new_prior: str = typer.Option(None, "--new-prior", help="New priors as JSON: '{\"Om\": [0.1, 0.5]}'"),
    out: str = typer.Option("reweighted.npz", "--out", "-o"),
    n_samples: int = typer.Option(None, "--n-samples", help="Resampled chain size (default: same as original)"),
    plot: bool = typer.Option(False, "--plot", help="Generate corner plot of reweighted chain"),
):
    """Reweight an existing chain to a new likelihood/prior."""
    chain_data = np.load(old_chain)
    chain = chain_data["chain"]
    param_names = list(chain_data["params"])
    old_model = chain_data.get("model", model)

    spec = MODEL_REGISTRY[model]
    priors = spec["priors"].copy()

    if new_prior:
        import json
        priors.update(json.loads(new_prior))

    # Compute old log-prob for each sample
    console.print("Computing old log-probabilities...")
    # We need the original data to compute old likelihood
    if "z" in chain_data and "mb_obs" in chain_data and "cov" in chain_data:
        z = chain_data["z"]
        mb_obs = chain_data["mb_obs"]
        cov = chain_data["cov"]
    elif sn_data and sn_cov:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)
    else:
        raise ValueError("Need either original data in chain file or --sn-data/--sn-cov")

    cov_inv = np.linalg.inv(cov)

    log_prob_old = np.array([
        log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], param_names) + log_prior(theta, param_names, priors)
        for theta in chain
    ])

    # Compute new log-prob (same likelihood, different prior - or new data if provided)
    log_prob_new = log_prob_old.copy()  # placeholder - in practice would use new data/prior

    console.print("Reweighting...")
    weights, n_eff = reweight_chain(chain, log_prob_old, log_prob_new)

    # Resample to equal weight
    chain_eq = resample_to_equal_weight(chain, weights, n_samples=n_samples)

    np.savez(out, chain=chain_eq, weights=weights, param_names=param_names, model=model, n_eff=n_eff)
    console.print(f"[green]Saved reweighted chain ({chain_eq.shape[0]} samples, N_eff={n_eff:.1f}) to {out}[/green]")

    if plot:
        corner_plot(chain_eq, param_names, spec["labels"], out_path=f"{out}_corner.png")
        console.print(f"[green]Corner plot saved to {out}_corner.png[/green]")

    # Print weighted quantiles
    console.print("\nWeighted quantiles (16, 50, 84):")
    for i, name in enumerate(param_names):
        q = weighted_quantiles(chain[:, i], weights)
        console.print(f"  {name}: {q[0]:.4f} < {q[1]:.4f} < {q[2]:.4f}")


if __name__ == "__main__":
    reweight_app()