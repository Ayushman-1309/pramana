"""CLI command: forecast — Fisher matrix forecasting."""
import typer
import numpy as np
from rich.console import Console

from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.fisher_forecast import (
    fisher_matrix_gaussian,
    forecast_errors,
    figure_of_merit,
    fisher_ellipse,
    compare_to_mcmc,
)
from pramana.core.plotting import corner_plot

forecast_app = typer.Typer(name="forecast", help="Fisher matrix forecasting")
console = Console()


@forecast_app.command("run")
def forecast_run(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    fiducial: str = typer.Option(None, "--fiducial", help="Fiducial params as JSON: '{\"Om\": 0.3, \"w0\": -1, \"wa\": 0}'"),
    out: str = typer.Option("fisher.npz", "--out", "-o"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    compare_mcmc: str = typer.Option(None, "--compare-mcmc", help="MCMC chain file for Fisher vs MCMC comparison"),
):
    """Run Fisher forecast for a model."""
    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    cov_inv = np.linalg.inv(cov)

    # Fiducial values
    if fiducial:
        import json
        fid_dict = json.loads(fiducial)
        theta_fid = np.array([fid_dict[p] for p in param_names])
    else:
        theta_fid = np.array([np.mean(spec["priors"][p]) for p in param_names])

    # Model function for Fisher (returns predictions at data points)
    def model_predictions(theta):
        params = dict(zip(param_names, theta))
        return spec["func"](z, **params)

    console.print(f"Computing Fisher matrix for {model} at fiducial: {dict(zip(param_names, theta_fid))}")
    fisher = fisher_matrix_gaussian(model_predictions, theta_fid, cov_inv)
    errs, cov_mat = forecast_errors(fisher, param_names)

    # Figure of merit for w0-wa if present
    if "w0" in param_names and "wa" in param_names:
        i, j = param_names.index("w0"), param_names.index("wa")
        fom = figure_of_merit(fisher, i, j)
        console.print(f"FoM (w0-wa) = {fom:.2f}")

    np.savez(out, fisher=fisher, cov=cov_mat, theta_fid=theta_fid, param_names=param_names)
    console.print(f"[green]Saved Fisher matrix to {out}[/green]")

    if compare_mcmc:
        chain_data = np.load(compare_mcmc)
        mcmc_chain = chain_data["chain"]
        compare_to_mcmc(errs, mcmc_chain, param_names)


@forecast_app.command("ellipse")
def forecast_ellipse(
    fisher_file: str = typer.Option(..., "--fisher", help="Fisher .npz file from forecast run"),
    param_i: str = typer.Option(..., "--param-i", help="First parameter name"),
    param_j: str = typer.Option(..., "--param-j", help="Second parameter name"),
    out: str = typer.Option("ellipse.png", "--out", "-o"),
):
    """Plot Fisher ellipse overlaid on MCMC corner (if chain provided)."""
    import matplotlib.pyplot as plt

    data = np.load(fisher_file)
    fisher = data["fisher"]
    theta_fid = data["theta_fid"]
    param_names = list(data["param_names"])
    cov_mat = data["cov"]

    i, j = param_names.index(param_i), param_names.index(param_j)

    x, y = fisher_ellipse(cov_mat, i, j, (theta_fid[i], theta_fid[j]))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, "r-", lw=2, label=f"Fisher {1}$\\sigma$")
    ax.set_xlabel(param_i)
    ax.set_ylabel(param_j)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    console.print(f"[green]Ellipse plot saved to {out}[/green]")


if __name__ == "__main__":
    forecast_app()