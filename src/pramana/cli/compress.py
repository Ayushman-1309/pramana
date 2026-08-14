"""CLI command: compress — MOPED optimal data compression."""
import typer
import numpy as np
from rich.console import Console

from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.data_compression import moped_vectors, compress, compressed_log_likelihood, compare_compressed_vs_full
from pramana.core.likelihood import log_likelihood

compress_app = typer.Typer(name="compress", help="MOPED optimal data compression")
console = Console()


@compress_app.command("run")
def compress_run(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    fiducial: str = typer.Option(None, "--fiducial", help="Fiducial params as JSON"),
    out: str = typer.Option("moped.npz", "--out", "-o"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate against full likelihood"),
):
    """Run MOPED compression on SN data."""
    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    cov_inv = np.linalg.inv(cov)

    if fiducial:
        import json
        fid_dict = json.loads(fiducial)
        theta_fid = np.array([fid_dict[p] for p in param_names])
    else:
        theta_fid = np.array([np.mean(spec["priors"][p]) for p in param_names])

    def model_predictions(theta):
        params = dict(zip(param_names, theta))
        return spec["func"](z, **params)

    console.print(f"Computing MOPED vectors for {model}...")
    B = moped_vectors(model_predictions, theta_fid, cov_inv)

    console.print(f"Compressing data ({len(mb_obs)} -> {B.shape[0]} numbers)...")
    y_compressed = compress(B, mb_obs)

    np.savez(out, B=B, y_compressed=y_compressed, theta_fid=theta_fid, param_names=param_names)
    console.print(f"[green]Saved compressed data to {out}[/green]")

    if validate:
        console.print("Validating compressed vs full likelihood...")
        def full_loglike(theta):
            return log_likelihood(theta, z, mb_obs, cov_inv, spec["func"], param_names)

        # Test points around fiducial
        test_points = theta_fid + np.random.normal(0, 0.02, size=(10, len(param_names)))
        compare_compressed_vs_full(
            model_predictions, theta_fid, cov_inv, mb_obs, test_points, full_loglike
        )


@compress_app.command("predict")
def compress_predict(
    moped_file: str = typer.Option(..., "--moped", help="MOPED .npz file from compress run"),
    theta: str = typer.Option(..., "--theta", help="Parameter values as JSON"),
):
    """Evaluate compressed likelihood at a parameter point."""
    import json

    data = np.load(moped_file)
    B = data["B"]
    y_compressed = data["y_compressed"]
    param_names = list(data["param_names"])
    theta_fid = data["theta_fid"]

    # Need model func - would need to reconstruct from param_names
    console.print("[yellow]Need model function to evaluate - use in Python directly[/yellow]")


if __name__ == "__main__":
    compress_app()