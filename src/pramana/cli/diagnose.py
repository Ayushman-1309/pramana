"""CLI command: diagnose — convergence diagnostics and chain analysis."""
import typer
import numpy as np
from rich.console import Console

from pramana.core.diagnostics import summarize, gelman_rubin
from pramana.core.plotting import corner_plot

diagnose_app = typer.Typer(name="diagnose", help="Convergence diagnostics")
console = Console()


@diagnose_app.command("chain")
def diagnose_chain(
    chain_file: str = typer.Option(..., "--chain", "-c", help="Chain .npz file"),
    model: str = typer.Option(None, "--model", "-m", help="Model name (if not in chain file)"),
    burn_in: float = typer.Option(0.3, "--burn-in", help="Burn-in fraction"),
    gelman_rubin_chains: list[str] = typer.Option(None, "--gelman-rubin", help="Additional chain files for R-hat"),
    plot: bool = typer.Option(False, "--plot", help="Generate corner plot"),
    out: str = typer.Option("diagnostics.png", "--out", "-o"),
):
    """Run diagnostics on an MCMC chain."""
    data = np.load(chain_file)
    flat_chain = data["chain"]
    param_names = list(data["params"])
    chain_model = data.get("model", model)

    if chain_model is None:
        raise ValueError("Model name required (not in chain file)")

    from pramana.core.models import MODEL_REGISTRY
    spec = MODEL_REGISTRY[chain_model]
    labels = spec["labels"]

    # For summarize we need the sampler - can only do basic stats from flat chain
    console.print(f"Chain shape: {flat_chain.shape}")
    console.print(f"Parameters: {param_names}")

    for i, name in enumerate(param_names):
        med = np.median(flat_chain[:, i])
        lo, hi = np.percentile(flat_chain[:, i], [16, 84])
        console.print(f"  {name}: {med:.4f}  +{hi - med:.4f}/-{med - lo:.4f}")

    if gelman_rubin_chains:
        chains = [flat_chain]
        for f in gelman_rubin_chains:
            chains.append(np.load(f)["chain"])
        rhat = gelman_rubin(chains)
        console.print("\nGelman-Rubin R-hat:")
        for i, name in enumerate(param_names):
            console.print(f"  {name}: {rhat[i]:.4f}  {'✓' if rhat[i] < 1.01 else '✗'}")

    if plot:
        corner_plot(flat_chain, param_names, labels, out_path=out)
        console.print(f"[green]Corner plot saved to {out}[/green]")


@diagnose_app.command("compare")
def diagnose_compare(
    chain_files: list[str] = typer.Option(..., "--chains", "-c", help="Chain .npz files to compare"),
    labels: list[str] = typer.Option(None, "--labels", help="Labels for each chain"),
    out: str = typer.Option("comparison.png", "--out", "-o"),
):
    """Compare multiple chains with getdist triangle plot."""
    from pramana.core.plotting import getdist_triangle

    chains = {}
    param_names_dict = {}
    for i, f in enumerate(chain_files):
        data = np.load(f)
        label = labels[i] if labels else f"chain_{i}"
        chains[label] = data["chain"]
        param_names_dict[label] = list(data["params"])

    getdist_triangle(chains, param_names_dict, out_path=out)
    console.print(f"[green]Triangle plot saved to {out}[/green]")


if __name__ == "__main__":
    diagnose_app()