"""Lazy/optional imports for heavy dependencies (CAMB, ACT, SBI, etc.).

These modules are only imported when the user explicitly uses the features
that require them, avoiding import errors if optional dependencies aren't installed.
"""


def get_camb():
    """Import CAMB or raise with install instructions."""
    try:
        import camb
        return camb
    except ImportError:
        raise RuntimeError(
            "CAMB not installed. Install with: `uv add camb` or `pip install camb`\n"
            "Then ensure it's available in your Python environment."
        )


def get_act_lenslike():
    """Import act_dr6_lenslike or raise with install instructions."""
    try:
        import act_dr6_lenslike
        return act_dr6_lenslike
    except ImportError:
        raise RuntimeError(
            "act_dr6_lenslike not installed.\n"
            "Download from NASA LAMBDA: https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr6/likelihood/\n"
            "Then install the package locally."
        )


def get_act_cmbonly():
    """Import act_dr6_cmbonly or raise with install instructions."""
    try:
        import act_dr6_cmbonly
        return act_dr6_cmbonly
    except ImportError:
        raise RuntimeError(
            "act_dr6_cmbonly not installed.\n"
            "Install via: `pip install \"act_dr6_cmbonly @ git+https://github.com/ACTCollaboration/DR6-ACT-lite.git\"`\n"
            "Also requires Cobaya: `uv add cobaya`"
        )


def get_cobaya():
    """Import Cobaya or raise with install instructions."""
    try:
        import cobaya
        return cobaya
    except ImportError:
        raise RuntimeError(
            "Cobaya not installed. Install with: `uv add cobaya` or `pip install cobaya`"
        )


def get_sbi():
    """Import sbi or raise with install instructions."""
    try:
        import sbi
        return sbi
    except ImportError:
        raise RuntimeError(
            "sbi not installed. Install with: `uv add sbi` or `pip install sbi`"
        )


def get_dynesty():
    """Import dynesty or raise with install instructions."""
    try:
        import dynesty
        return dynesty
    except ImportError:
        raise RuntimeError(
            "dynesty not installed. Install with: `uv add dynesty` or `pip install dynesty`"
        )


def get_emcee():
    """Import emcee or raise with install instructions."""
    try:
        import emcee
        return emcee
    except ImportError:
        raise RuntimeError(
            "emcee not installed. Install with: `uv add emcee` or `pip install emcee`"
        )


def get_numpyro():
    """Import numpyro or raise with install instructions."""
    try:
        import numpyro
        return numpyro
    except ImportError:
        raise RuntimeError(
            "numpyro not installed. Install with: `uv add numpyro` or `pip install numpyro`\n"
            "Also requires JAX: `uv add jax jaxlib`"
        )


def get_corner():
    """Import corner or raise with install instructions."""
    try:
        import corner
        return corner
    except ImportError:
        raise RuntimeError(
            "corner not installed. Install with: `uv add corner` or `pip install corner`"
        )


def get_getdist():
    """Import getdist or raise with install instructions."""
    try:
        from getdist import MCSamples, plots
        return MCSamples, plots
    except ImportError:
        raise RuntimeError(
            "getdist not installed. Install with: `uv add getdist` or `pip install getdist`"
        )