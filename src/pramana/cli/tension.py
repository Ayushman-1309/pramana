"""CLI command: tension — H0/S8 tension analysis and whisker plots."""
import typer
from rich.console import Console

from pramana.core.jwst_probes import (
    H0_MEASUREMENTS,
    S8_MEASUREMENTS,
    h0_tension_sigma,
    s8_tension_sigma,
    plot_h0_whisker,
    plot_s8_whisker,
    append_supernovae,
)

tension_app = typer.Typer(name="tension", help="H0/S8 tension analysis")
console = Console()


@tension_app.command("h0")
def tension_h0(
    list_measurements: bool = typer.Option(False, "--list", "-l", help="List available measurements"),
    tension: tuple[str, str] = typer.Option(None, "--tension", "-t", help="Compute tension between two measurements"),
    plot: bool = typer.Option(False, "--plot", help="Generate whisker plot"),
    out: str = typer.Option("h0_tension.png", "--out", "-o"),
):
    """H0 tension analysis."""
    if list_measurements:
        table = Console().print("Available H0 measurements:")
        for name, d in H0_MEASUREMENTS.items():
            console.print(f"  {name}: {d['H0']} +/- {d['err']} ({d['family']})")
        return

    if tension:
        name_a, name_b = tension
        sigma = h0_tension_sigma(name_a, name_b)
        console.print(f"Tension between {name_a} and {name_b}: {sigma:.2f} sigma")
        return

    if plot:
        console.print("Generating H0 whisker plot...")
        plot_h0_whisker(out_path=out)
        console.print(f"[green]Saved to {out}[/green]")


@tension_app.command("s8")
def tension_s8(
    list_measurements: bool = typer.Option(False, "--list", "-l", help="List available measurements"),
    tension: tuple[str, str] = typer.Option(None, "--tension", "-t", help="Compute tension between two measurements"),
    plot: bool = typer.Option(False, "--plot", help="Generate whisker plot"),
    out: str = typer.Option("s8_tension.png", "--out", "-o"),
):
    """S8 tension analysis."""
    if list_measurements:
        for name, d in S8_MEASUREMENTS.items():
            console.print(f"  {name}: {d['S8']} +/- {d['err']} ({d['family']})")
        return

    if tension:
        name_a, name_b = tension
        sigma = s8_tension_sigma(name_a, name_b)
        console.print(f"Tension between {name_a} and {name_b}: {sigma:.2f} sigma")
        return

    if plot:
        console.print("Generating S8 whisker plot...")
        plot_s8_whisker(out_path=out)
        console.print(f"[green]Saved to {out}[/green]")


@tension_app.command("append-sn")
def tension_append_sn(
    base_data: str = typer.Option(..., "--base-data", help="Base Pantheon+ data .npz or files"),
    base_cov: str = typer.Option(None, "--base-cov"),
    z_new: str = typer.Option(..., "--z-new", help="New SN redshifts (comma-separated)"),
    mb_new: str = typer.Option(..., "--mb-new", help="New SN magnitudes (comma-separated)"),
    mb_err_new: str = typer.Option(..., "--mb-err-new", help="New SN magnitude errors (comma-separated)"),
    out: str = typer.Option("extended_data.npz", "--out", "-o"),
):
    """Append new high-z SNe to Pantheon+ (e.g., JWST discoveries)."""
    import numpy as np

    # Load base
    if base_data.endswith(".npz"):
        data = np.load(base_data)
        z_base = data["z"]
        mb_base = data["mb_obs"]
        cov_base = data["cov"]
    else:
        from pramana.core.data_io import load_pantheon
        z_base, mb_base, cov_base, _ = load_pantheon(base_data, base_cov)

    z_new_arr = np.array([float(x) for x in z_new.split(",")])
    mb_new_arr = np.array([float(x) for x in mb_new.split(",")])
    mb_err_new_arr = np.array([float(x) for x in mb_err_new.split(",")])

    z_out, mb_out, cov_out = append_supernovae(z_base, mb_base, cov_base, z_new_arr, mb_new_arr, mb_err_new_arr)

    np.savez(out, z=z_out, mb_obs=mb_out, cov=cov_out)
    console.print(f"[green]Saved extended dataset ({len(z_out)} SNe) to {out}[/green]")


if __name__ == "__main__":
    tension_app()