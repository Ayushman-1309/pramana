"""CLI command: fit — run parameter inference with various methods."""
import typer
import numpy as np
from rich.console import Console
from rich.table import Table

from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.mcmc import run_fit as run_mcmc_fit
from pramana.core.diagnostics import summarize, gelman_rubin
from pramana.core.plotting import corner_plot, compare_hubble_diagram
from pramana.core.nested_sampling import run_nested, equal_weight_posterior, bayes_factor
from pramana.core.profile_likelihood import profile_scan, confidence_interval_from_profile
from pramana.core.sbi_inference import make_simulator, train_npe, sample_posterior, validate_on_synthetic
from pramana.core.hmc_numpyro import build_sn_model, run_nuts, samples_to_flat_chain

fit_app = typer.Typer(name="fit", help="Run parameter inference")
console = Console()


@fit_app.command("mcmc")
def fit_mcmc(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    nwalkers: int = typer.Option(32, "--nwalkers"),
    nsteps: int = typer.Option(4000, "--nsteps"),
    out: str = typer.Option("chain.npz", "--out", "-o"),
    seed: int = typer.Option(42, "--seed"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use synthetic data"),
    plot: bool = typer.Option(False, "--plot", help="Generate corner plot"),
):
    """Run emcee MCMC fit."""
    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    spec = MODEL_REGISTRY[model]
    console.print(f"Running MCMC for {model} with {nwalkers} walkers, {nsteps} steps...")
    sampler = run_mcmc_fit(model, z, mb_obs, cov, nwalkers=nwalkers, nsteps=nsteps, seed=seed)

    burn_in = int(nsteps * 0.3)
    flat_chain = sampler.get_chain(discard=burn_in, flat=True)

    # Save
    np.savez(out, chain=flat_chain, params=spec["params"], model=model)
    console.print(f"[green]Saved {flat_chain.shape[0]} samples to {out}[/green]")

    # Diagnostics
    summarize(sampler, spec["params"])

    # Plot
    if plot:
        corner_plot(flat_chain, spec["params"], spec["labels"], out_path=f"{out}_corner.png")
        console.print(f"[green]Corner plot saved to {out}_corner.png[/green]")


@fit_app.command("nested")
def fit_nested(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    nlive: int = typer.Option(500, "--nlive"),
    out: str = typer.Option("nested_results.npz", "--out", "-o"),
    seed: int = typer.Option(42, "--seed"),
    synthetic: bool = typer.Option(False, "--synthetic"),
):
    """Run dynesty nested sampling (evidence + posteriors)."""
    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    priors = spec["priors"]
    cov_inv = np.linalg.inv(cov)

    from pramana.core.likelihood import log_likelihood

    def loglike(theta):
        return log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], param_names)

    console.print(f"Running nested sampling for {model} with {nlive} live points...")
    results = run_nested(loglike, param_names, priors, nlive=nlive, seed=seed)

    # Equal-weight posterior
    posterior = equal_weight_posterior(results)

    # Save
    np.savez(out, chain=posterior, params=param_names, model=model,
             logz=results.logz[-1], logzerr=results.logzerr[-1])
    console.print(f"[green]Saved {posterior.shape[0]} posterior samples to {out}[/green]")
    console.print(f"ln(Z) = {results.logz[-1]:.2f} +/- {results.logzerr[-1]:.2f}")


@fit_app.command("nuts")
def fit_nuts(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    num_warmup: int = typer.Option(1000, "--warmup"),
    num_samples: int = typer.Option(2000, "--samples"),
    num_chains: int = typer.Option(2, "--chains"),
    out: str = typer.Option("nuts_chain.npz", "--out", "-o"),
    seed: int = typer.Option(0, "--seed"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    plot: bool = typer.Option(False, "--plot"),
):
    """Run NUTS/HMC via numpyro (requires JAX)."""
    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    spec = MODEL_REGISTRY[model]
    priors = spec["priors"]
    cov_inv = np.linalg.inv(cov)

    console.print(f"Running NUTS for {model} with {num_chains} chains...")
    model_numpyro, param_names = build_sn_model(z, mb_obs, cov_inv, model, priors)
    mcmc = run_nuts(model_numpyro, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=num_chains, seed=seed)

    flat_chain = samples_to_flat_chain(mcmc, param_names)

    # Save
    np.savez(out, chain=flat_chain, params=param_names, model=model)
    console.print(f"[green]Saved {flat_chain.shape[0]} samples to {out}[/green]")

    # Print summary
    mcmc.print_summary()

    if plot:
        corner_plot(flat_chain, param_names, spec["labels"], out_path=f"{out}_corner.png")
        console.print(f"[green]Corner plot saved to {out}_corner.png[/green]")


@fit_app.command("profile")
def fit_profile(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    param_of_interest: str = typer.Option(..., "--param", "-p", help="Parameter to profile"),
    n_points: int = typer.Option(30, "--points", help="Number of scan points"),
    out: str = typer.Option("profile.npz", "--out", "-o"),
    synthetic: bool = typer.Option(False, "--synthetic"),
):
    """Run profile likelihood (frequentist)."""
    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    priors = spec["priors"]
    cov_inv = np.linalg.inv(cov)

    from pramana.core.likelihood import log_likelihood

    def neg_log_likelihood(theta):
        return -log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], param_names)

    # Build bounds
    bounds = priors.copy()

    console.print(f"Running profile likelihood for {param_of_interest}...")
    scan_vals = np.linspace(bounds[param_of_interest][0], bounds[param_of_interest][1], n_points)
    scan_vals, profile_nll, best_fits = profile_scan(
        neg_log_likelihood, param_names, param_of_interest, scan_vals, bounds
    )

    best, lo, hi = confidence_interval_from_profile(scan_vals, profile_nll, cl=0.6827)
    console.print(f"Best fit {param_of_interest} = {best:.4f}  68% CI: [{lo:.4f}, {hi:.4f}]")

    np.savez(out, scan_vals=scan_vals, profile_nll=profile_nll, best_fits=best_fits,
             param_of_interest=param_of_interest, model=model)
    console.print(f"[green]Saved profile to {out}[/green]")


@fit_app.command("sbi")
def fit_sbi(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    n_simulations: int = typer.Option(2000, "--sims"),
    n_samples: int = typer.Option(5000, "--samples"),
    out: str = typer.Option("sbi_posterior.npz", "--out", "-o"),
    seed: int = typer.Option(0, "--seed"),
    synthetic: bool = typer.Option(False, "--synthetic"),
):
    """Run simulation-based inference (SBI/NPE)."""
    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    priors = spec["priors"]

    mb_err = np.sqrt(np.diag(cov))  # diagonal approx for simulator

    console.print(f"Training NPE for {model} with {n_simulations} simulations...")
    simulator = make_simulator(spec["func"], z, mb_err)
    posterior = train_npe(simulator, priors, param_names, n_simulations=n_simulations, seed=seed)

    console.print("Sampling posterior...")
    samples = sample_posterior(posterior, mb_obs, n_samples=n_samples, seed=seed)

    np.savez(out, chain=samples, params=param_names, model=model)
    console.print(f"[green]Saved {samples.shape[0]} posterior samples to {out}[/green]")

    # Validate on synthetic if using synthetic data
    if synthetic:
        from pramana.core.data_io import make_synthetic_dataset
        z_true, mb_true, _ = make_synthetic_dataset(seed=seed+1)
        validate_on_synthetic(posterior, simulator, np.array([0.3]), param_names)


if __name__ == "__main__":
    fit_app()