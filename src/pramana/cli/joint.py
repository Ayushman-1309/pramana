"""CLI command: joint — multi-probe joint fits (SN + BAO + CMB)."""
import typer
import numpy as np
from rich.console import Console

from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon
from pramana.core.joint_likelihood import build_joint_log_probability, per_probe_chi2
from pramana.core.mcmc import run_fit as run_mcmc_fit
from pramana.core.diagnostics import summarize
from pramana.core.plotting import corner_plot

joint_app = typer.Typer(name="joint", help="Joint multi-probe fits")
console = Console()


@joint_app.command("fit")
def joint_fit(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Use synthetic data"),
    bao: bool = typer.Option(True, "--bao/--no-bao", help="Include DESI DR2 BAO"),
    bao_H0: float = typer.Option(70.0, "--bao-H0", help="H0 for BAO (if not fitting H0)"),
    rd_mode: str = typer.Option("eh98", "--rd-mode", help="rd mode: eh98, free, planck_prior"),
    nwalkers: int = typer.Option(32, "--nwalkers"),
    nsteps: int = typer.Option(8000, "--nsteps"),
    out: str = typer.Option("joint_chain.npz", "--out", "-o"),
    seed: int = typer.Option(42, "--seed"),
    plot: bool = typer.Option(False, "--plot"),
    per_probe: bool = typer.Option(True, "--per-probe/--no-per-probe", help="Print per-probe chi2"),
):
    """Run joint SN+BAO MCMC fit."""
    from pramana.core.data_io import load_pantheon, make_synthetic_dataset

    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    probes = [{"kind": "sn", "z": z, "mb_obs": mb_obs, "cov_inv": np.linalg.inv(cov)}]
    if bao:
        probes.append({"kind": "bao", "H0": bao_H0, "rd_mode": rd_mode})

    log_prob, param_names = build_joint_log_probability(model, probes)
    spec = MODEL_REGISTRY[model]
    priors = spec["priors"]
    ndim = len(param_names)

    # Initial guess
    rng = np.random.default_rng(seed)
    p0_center = np.array([np.mean(priors[p]) for p in param_names])
    p0_spread = np.array([(priors[p][1] - priors[p][0]) * 0.05 for p in param_names])
    p0 = p0_center + p0_spread * rng.normal(size=(nwalkers, ndim))

    console.print(f"Running joint fit for {model} with {len(probes)} probes...")

    import emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=True)

    burn_in = int(nsteps * 0.3)
    flat_chain = sampler.get_chain(discard=burn_in, flat=True)

    np.savez(out, chain=flat_chain, params=param_names, model=model, probes=probes)
    console.print(f"[green]Saved {flat_chain.shape[0]} samples to {out}[/green]")

    if per_probe:
        best = np.median(flat_chain, axis=0)
        per_probe_chi2(model, best, probes)

    summarize(sampler, param_names)

    if plot:
        corner_plot(flat_chain, param_names, spec["labels"], out_path=f"{out}_corner.png")
        console.print(f"[green]Corner plot saved to {out}_corner.png[/green]")


@joint_app.command("compare")
def joint_compare(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(..., "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(..., "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    out: str = typer.Option("joint_compare.npz", "--out", "-o"),
):
    """Compare single-probe vs joint constraints."""
    from pramana.core.mcmc import run_fit
    from pramana.core.plotting import getdist_triangle

    z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    probes_sn = [{"kind": "sn", "z": z, "mb_obs": mb_obs, "cov_inv": np.linalg.inv(cov)}]
    probes_joint = probes_sn + [{"kind": "bao", "H0": 70.0, "rd_mode": "eh98"}]

    chains = {}
    param_names_dict = {}

    for label, probes in [("SN only", probes_sn), ("SN+BAO", probes_joint)]:
        log_prob, param_names = build_joint_log_probability(model, probes)
        spec = MODEL_REGISTRY[model]

        # Run shorter chains for comparison
        import emcee
        ndim = len(param_names)
        rng = np.random.default_rng(42)
        p0_center = np.array([np.mean(spec["priors"][p]) for p in param_names])
        p0_spread = np.array([(spec["priors"][p][1] - spec["priors"][p][0]) * 0.05 for p in param_names])
        p0 = p0_center + p0_spread * rng.normal(size=(32, ndim))

        sampler = emcee.EnsembleSampler(32, ndim, log_prob)
        sampler.run_mcmc(p0, 4000, progress=True)
        chains[label] = sampler.get_chain(discard=1200, flat=True)
        param_names_dict[label] = param_names

    getdist_triangle(chains, param_names_dict, out_path=f"{out}_triangle.png")
    console.print(f"[green]Triangle plot saved to {out}_triangle.png[/green]")


if __name__ == "__main__":
    joint_app()