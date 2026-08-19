"""CLI command: data — data loading, validation, and exploration."""
import typer
import numpy as np
from rich.console import Console
from rich.table import Table

from pramana.core.data_io import load_pantheon, make_synthetic_dataset
from pramana.core.bao_desi import DESI_DR2_BAO_TABLE, build_data_vector_and_cov
from pramana.core.jwst_probes import H0_MEASUREMENTS, S8_MEASUREMENTS
from pramana.utils.validators import validate_pantheon_data, validate_pantheon_cov

data_app = typer.Typer(name="data", help="Data loading, validation, exploration")
console = Console()


@data_app.command("pantheon")
def data_pantheon(
    data_file: str = typer.Option(..., "--data", help="Path to Pantheon+SH0ES.dat"),
    cov_file: str = typer.Option(..., "--cov", help="Path to Pantheon+SH0ES_STAT+SYS.cov"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate file format"),
    stats: bool = typer.Option(True, "--stats", help="Print basic statistics"),
    synthetic: bool = typer.Option(False, "--synthetic", help="Generate synthetic data"),
    out: str = typer.Option(None, "--out", help="Save as .npz"),
):
    """Load and explore Pantheon+ data."""
    if synthetic:
        console.print("[yellow]Generating synthetic Pantheon+ data[/yellow]")
        z, mb_obs, cov = make_synthetic_dataset()
        df = None
    else:
        if validate:
            console.print("Validating data file...")
            val = validate_pantheon_data(data_file)
            console.print(f"  Rows: {val['n_rows']}, z range: {val['z_range']}")
            val_cov = validate_pantheon_cov(cov_file, val['n_rows'])
            console.print(f"  Cov shape: {val_cov['shape']}, cond: {val_cov['condition_number']:.2e}")

        z, mb_obs, cov, df = load_pantheon(data_file, cov_file)

    if stats:
        table = Table(title="Pantheon+ Data Summary")
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("N SNe", str(len(z)))
        table.add_row("z min", f"{z.min():.4f}")
        table.add_row("z max", f"{z.max():.4f}")
        table.add_row("m_b mean", f"{mb_obs.mean():.3f}")
        table.add_row("m_b std", f"{mb_obs.std():.3f}")
        table.add_row("Cov shape", f"{cov.shape}")
        table.add_row("Cov cond. no.", f"{np.linalg.cond(cov):.2e}")
        console.print(table)

    if out:
        np.savez(out, z=z, mb_obs=mb_obs, cov=cov)
        console.print(f"[green]Saved to {out}[/green]")


@data_app.command("desi")
def data_desi(
    show_table: bool = typer.Option(True, "--table/--no-table", help="Show DESI DR2 BAO table"),
    validate: bool = typer.Option(False, "--validate", help="Validate reference table"),
):
    """Explore DESI DR2 BAO data (reference table)."""
    if show_table:
        table = Table(title="DESI DR2 BAO Measurements")
        table.add_column("Tracer", style="cyan")
        table.add_column("z", style="green")
        table.add_column("Observable", style="yellow")
        table.add_column("Value", style="white")
        table.add_column("Error", style="white")

        for tracer, d in DESI_DR2_BAO_TABLE.items():
            if "DV_rd" in d:
                table.add_row(tracer, f"{d['z']:.3f}", "DV/rd", f"{d['DV_rd']:.3f}", f"{d['DV_rd_err']:.3f}")
            else:
                table.add_row(tracer, f"{d['z']:.3f}", "DM/rd", f"{d['DM_rd']:.3f}", f"{d['DM_rd_err']:.3f}")
                table.add_row("", "", "DH/rd", f"{d['DH_rd']:.3f}", f"{d['DH_rd_err']:.3f}")
                table.add_row("", "", "rho(DM,DH)", f"{d['rho_MH']:.3f}", "")
        console.print(table)

    if validate:
        labels, z_arr, data, cov = build_data_vector_and_cov()
        console.print(f"Data vector length: {len(data)}")
        console.print(f"Covariance shape: {cov.shape}")
        console.print(f"Condition number: {np.linalg.cond(cov):.2e}")
        console.print(f"z range: {z_arr.min():.3f} - {z_arr.max():.3f}")


@data_app.command("h0")
def data_h0(
    list_all: bool = typer.Option(True, "--list/--no-list", help="List all H0 measurements"),
):
    """Show H0 tension measurements."""
    table = Table(title="H0 Measurements")
    table.add_column("Measurement", style="cyan")
    table.add_column("H0", style="green")
    table.add_column("Error", style="yellow")
    table.add_column("Family", style="white")

    for name, d in H0_MEASUREMENTS.items():
        table.add_row(name, f"{d['H0']:.2f}", f"{d['err']:.2f}", d['family'])
    console.print(table)


@data_app.command("s8")
def data_s8(
    list_all: bool = typer.Option(True, "--list/--no-list", help="List all S8 measurements"),
):
    """Show S8 tension measurements."""
    table = Table(title="S8 Measurements")
    table.add_column("Measurement", style="cyan")
    table.add_column("S8", style="green")
    table.add_column("Error", style="yellow")
    table.add_column("Family", style="white")

    for name, d in S8_MEASUREMENTS.items():
        table.add_row(name, f"{d['S8']:.3f}", f"{d['err']:.4f}", d['family'])
    console.print(table)


if __name__ == "__main__":
    data_app()