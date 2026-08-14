"""CLI command: emulate — GP emulator training and validation."""
import typer
import numpy as np
from rich.console import Console

from pramana.core.models import MODEL_REGISTRY
from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.gp_emulator import latin_hypercube_design, train_emulator, emulate, validate_emulator

emulate_app = typer.Typer(name="emulate", help="GP emulator training/validation")
console = Console()


@emulate_app.command("train")
def emulate_train(
    model: str = typer.Option(..., "--model", "-m", help="Model name: lcdm, wcdm, cpl"),
    sn_data: str = typer.Option(None, "--sn-data", help="Path to Pantheon+SH0ES.dat"),
    sn_cov: str = typer.Option(None, "--sn-cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    n_train: int = typer.Option(200, "--n-train", help="Number of training points"),
    n_test: int = typer.Option(50, "--n-test", help="Number of test points"),
    out: str = typer.Option("emulator.pkl", "--out", "-o"),
    synthetic: bool = typer.Option(False, "--synthetic"),
):
    """Train GP emulator for model predictions."""
    if synthetic or not sn_data:
        console.print("[yellow]Using synthetic data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
    else:
        z, mb_obs, cov, _ = load_pantheon(sn_data, sn_cov)

    spec = MODEL_REGISTRY[model]
    param_names = spec["params"]
    priors = spec["priors"]

    bounds = [priors[p] for p in param_names]

    console.print(f"Generating {n_train} training points via Latin hypercube...")
    theta_train = latin_hypercube_design(bounds, n_train)

    console.print("Evaluating model at training points...")
    y_train = np.array([spec["func"](z, *theta) for theta in theta_train])

    console.print("Training GP emulator...")
    emulator = train_emulator(theta_train, y_train)

    # Validation
    console.print(f"Generating {n_test} test points for validation...")
    theta_test = latin_hypercube_design(bounds, n_test, seed=123)
    y_test = np.array([spec["func"](z, *theta) for theta in theta_test])

    console.print("Validating...")
    validate_emulator(emulator, theta_test, y_test)

    import pickle
    with open(out, "wb") as f:
        pickle.dump({"emulator": emulator, "param_names": param_names, "z": z}, f)
    console.print(f"[green]Saved emulator to {out}[/green]")


@emulate_app.command("predict")
def emulate_predict(
    emulator_file: str = typer.Option(..., "--emulator", "-e", help="Trained emulator .pkl file"),
    theta: str = typer.Option(..., "--theta", help="Parameter values as JSON: '{\"Om\": 0.3, \"w0\": -1, \"wa\": 0}'"),
):
    """Evaluate emulator at a parameter point."""
    import pickle
    import json

    with open(emulator_file, "rb") as f:
        data = pickle.load(f)

    emulator = data["emulator"]
    param_names = data["param_names"]

    theta_dict = json.loads(theta)
    theta_arr = np.array([theta_dict[p] for p in param_names])

    y_pred, y_std = emulate(emulator, theta_arr, return_std=True)
    console.print(f"Prediction: {y_pred}")
    console.print(f"Uncertainty: {y_std}")


if __name__ == "__main__":
    emulate_app()