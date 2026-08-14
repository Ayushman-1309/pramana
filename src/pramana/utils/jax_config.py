"""JAX backend configuration — CPU default, auto-detect GPU, manual override."""
import os
import platform
import subprocess


def configure_jax(backend: str = "auto") -> None:
    """
    Configure JAX platform before any JAX import.

    backend: "auto" | "cpu" | "cuda" | "metal"
    - auto: detect NVIDIA GPU -> cuda, Apple Silicon -> metal, else cpu
    - Sets JAX_PLATFORM_NAME env var (must be set before jax import)
    """
    if backend == "auto":
        # Try to detect NVIDIA GPU
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            if result.returncode == 0:
                backend = "cuda"
            else:
                raise FileNotFoundError
        except (FileNotFoundError, subprocess.SubprocessError):
            # Check for Apple Silicon (Metal)
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                backend = "metal"
            else:
                backend = "cpu"

    # Map to JAX platform names
    platform_map = {
        "cpu": "cpu",
        "cuda": "gpu",
        "metal": "metal",
    }
    platform_name = platform_map.get(backend, "cpu")

    os.environ["JAX_PLATFORM_NAME"] = platform_name
    os.environ["JAX_ENABLE_X64"] = "1"  # Enable 64-bit for precision

    print(f"JAX backend configured: {platform_name}")


def get_jax_backend() -> str:
    """Get the currently configured JAX backend."""
    import jax
    return jax.default_backend()