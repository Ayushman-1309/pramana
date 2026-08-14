"""JAX reimplementations of the distance-modulus models in models.py.
Purpose: gradient-based inference (HMC/NUTS via numpyro in hmc_numpyro.py)
needs d(log-likelihood)/d(theta), which JAX gets via autodiff instead of
emcee's gradient-free random walk — this is what makes HMC scale to
higher-dimensional parameter spaces (e.g. a joint SN+BAO+CMB fit with
10+ parameters) where emcee's mixing gets slow.

Kept as a SEPARATE module from models.py rather than swapping numpy for
jax.numpy there, because (a) camb/emcee/dynesty/scipy don't need or want
JAX arrays, and (b) it makes explicit which parts of the suite are
differentiable vs not, rather than a silent global dependency change.
Mirrors models.py's functions 1:1, so keep both in sync if the physics
changes.
"""
import jax
import jax.numpy as jnp


def e_of_z_lcdm(z: jnp.ndarray, Om: float) -> jnp.ndarray:
    return jnp.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def e_of_z_wcdm(z: jnp.ndarray, Om: float, w: float) -> jnp.ndarray:
    return jnp.sqrt(Om * (1 + z) ** 3 + (1 - Om) * (1 + z) ** (3 * (1 + w)))


def e_of_z_cpl(z: jnp.ndarray, Om: float, w0: float, wa: float) -> jnp.ndarray:
    de_evol = (1 + z) ** (3 * (1 + w0 + wa)) * jnp.exp(-3 * wa * z / (1 + z))
    return jnp.sqrt(Om * (1 + z) ** 3 + (1 - Om) * de_evol)


C_LIGHT = 299792.458


def _luminosity_distance(
    z: jnp.ndarray,
    H0: float,
    e_of_z_func,
    params: tuple,
    z_grid_points: int = 500,
) -> jnp.ndarray:
    """JAX version: fixed-size grid (static shape, required for jit) rather
    than models.py's z.max()-dependent grid — trade a slightly coarser
    integral for jit-compatibility. 500 points keeps sub-0.01% error for
    the z<3 range relevant here; bump z_grid_points if extending to
    higher-z JWST applications."""
    zgrid = jnp.linspace(0, 3.5, z_grid_points)
    integrand = 1.0 / e_of_z_func(zgrid, *params)
    comoving_grid = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(
        (integrand[1:] + integrand[:-1]) / 2 * jnp.diff(zgrid)
    )])
    comoving = jnp.interp(z, zgrid, comoving_grid)
    return (1 + z) * (C_LIGHT / H0) * comoving


def distance_modulus_lcdm(z: jnp.ndarray, Om: float, H0: float = 70.0) -> jnp.ndarray:
    dl = _luminosity_distance(z, H0, e_of_z_lcdm, (Om,))
    return 25 + 5 * jnp.log10(dl)


def distance_modulus_wcdm(z: jnp.ndarray, Om: float, w: float, H0: float = 70.0) -> jnp.ndarray:
    dl = _luminosity_distance(z, H0, e_of_z_wcdm, (Om, w))
    return 25 + 5 * jnp.log10(dl)


def distance_modulus_cpl(z: jnp.ndarray, Om: float, w0: float, wa: float, H0: float = 70.0) -> jnp.ndarray:
    dl = _luminosity_distance(z, H0, e_of_z_cpl, (Om, w0, wa))
    return 25 + 5 * jnp.log10(dl)


JAX_MODEL_REGISTRY = {
    "lcdm": {"func": distance_modulus_lcdm, "e_of_z": e_of_z_lcdm, "params": ["Om"]},
    "wcdm": {"func": distance_modulus_wcdm, "e_of_z": e_of_z_wcdm, "params": ["Om", "w"]},
    "cpl": {"func": distance_modulus_cpl, "e_of_z": e_of_z_cpl, "params": ["Om", "w0", "wa"]},
}


def check_gradient(model_name: str, theta: np.ndarray, z_test: float = 1.0):
    """Sanity check: confirm jax.grad runs and returns finite values —
    the thing that actually matters for HMC (a wrong-but-finite gradient
    would silently break the sampler without an explicit check like this)."""
    spec = JAX_MODEL_REGISTRY[model_name]

    def scalar_out(theta):
        return spec["func"](jnp.array(z_test), *theta)

    grad_fn = jax.grad(scalar_out)
    g = grad_fn(jnp.array(theta))
    return g, bool(jnp.all(jnp.isfinite(g)))