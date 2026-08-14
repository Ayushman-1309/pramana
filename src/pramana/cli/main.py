"""PRAMANA CLI — Unified Cosmological Inference Suite.

Usage:
    pramana fit --method mcmc --model lcdm --sn-data data/pantheon/Pantheon+SH0ES.dat --sn-cov data/pantheon/Pantheon+SH0ES_STAT+SYS.cov
    pramana fit --method nested --model cpl --sn-data data/pantheon/... --sn-cov data/pantheon/...
    pramana fit --method nuts --model wcdm --sn-data data/pantheon/... --sn-cov data/pantheon/...
    pramana fit --method profile --model lcdm --sn-data data/pantheon/... --sn-cov data/pantheon/...
    pramana fit --method sbi --model lcdm --sn-data data/pantheon/... --sn-cov data/pantheon/...

    pramana joint --model cpl --sn-data data/pantheon/... --sn-cov data/pantheon/... --bao

    pramana forecast --model cpl --sn-data data/pantheon/... --sn-cov data/pantheon/...

    pramana emulate --model cpl --sn-data data/pantheon/... --sn-cov data/pantheon/...

    pramana compress --model cpl --sn-data data/pantheon/... --sn-cov data/pantheon/...

    pramana reweight --old-chain chain1.npz --new-likelihood joint --sn-data ... --sn-cov ...

    pramana tension --h0 --s8

    pramana diagnose --chain chain.npz --model lcdm
"""
import typer
from rich.console import Console

from pramana.cli.fit import fit_app
from pramana.cli.joint import joint_app
from pramana.cli.forecast import forecast_app
from pramana.cli.emulate import emulate_app
from pramana.cli.compress import compress_app
from pramana.cli.reweight import reweight_app
from pramana.cli.tension import tension_app
from pramana.cli.diagnose import diagnose_app
from pramana.cli.data import data_app

app = typer.Typer(
    name="pramana",
    help="PRAMANA — Unified Cosmological Inference Suite",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

# Register subcommands
app.add_typer(fit_app, name="fit")
app.add_typer(joint_app, name="joint")
app.add_typer(forecast_app, name="forecast")
app.add_typer(emulate_app, name="emulate")
app.add_typer(compress_app, name="compress")
app.add_typer(reweight_app, name="reweight")
app.add_typer(tension_app, name="tension")
app.add_typer(diagnose_app, name="diagnose")
app.add_typer(data_app, name="data")


@app.callback()
def main(
    ctx: typer.Context,
    jax_backend: str = typer.Option(
        "auto",
        "--jax-backend",
        help="JAX backend: auto, cpu, cuda, metal",
        envvar="PRAMANA_JAX_BACKEND",
    ),
):
    """PRAMANA — Unified Cosmological Inference Suite.

    Sanskrit *pramāṇa*: a means of valid knowledge.
    """
    from pramana.utils.jax_config import configure_jax
    configure_jax(jax_backend)


if __name__ == "__main__":
    app()