"""Corner plots and multi-model comparison plots for MCMC chains."""
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_COLOR = "#3a3a8c"


def corner_plot(
    flat_chain: np.ndarray,
    param_names: list[str],
    labels: dict[str, str],
    truths: dict[str, float] | None = None,
    out_path: str | None = None,
):
    """Single-model corner plot using corner.py."""
    import corner

    fig = corner.corner(
        flat_chain,
        labels=[labels.get(p, p) for p in param_names],
        truths=[truths[p] for p in param_names] if truths else None,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        color=DEFAULT_COLOR,
    )
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def getdist_triangle(
    chains_dict: dict[str, np.ndarray],
    param_names_dict: dict[str, list[str]],
    out_path: str | None = None,
):
    """Overlay multiple models' posteriors on one triangle plot (getdist)."""
    from getdist import MCSamples, plots

    samples_list = [
        MCSamples(samples=chain, names=param_names_dict[name], label=name)
        for name, chain in chains_dict.items()
    ]

    g = plots.get_subplot_plotter()
    g.triangle_plot(samples_list, filled=True)
    if out_path:
        g.export(out_path)
    return g


def compare_hubble_diagram(
    z: np.ndarray,
    mb_obs: np.ndarray,
    model_fits: dict[str, tuple],
    out_path: str | None = None,
):
    """Overlay best-fit models on the SN Hubble diagram (magnitude residual view).

    model_fits: {model_name: (model_func, best_fit_params_dict)}
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(z, mb_obs, s=6, alpha=0.35, color="gray", label="SN data")

    zgrid = np.linspace(max(z.min(), 1e-3), z.max(), 200)
    for name, (func, params) in model_fits.items():
        ax.plot(zgrid, func(zgrid, **params), lw=2, label=name)

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Corrected apparent magnitude")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig