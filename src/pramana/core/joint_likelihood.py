"""Combine multiple probes (SN, BAO, and — once real data is available —
CMB lensing/primary) into a single log-posterior, usable with ANY of the
inference engines in this suite (emcee, dynesty, NUTS, profile likelihood).

Independence assumption: total log-likelihood = sum of each probe's
log-likelihood. This is standard practice (SN, BAO, and CMB are
independent experiments with uncorrelated systematics, to good
approximation) — flag it explicitly if a task ever needs to account for
shared systematics across probes (e.g. a shared calibration), since that
requires a joint covariance this simple sum doesn't capture.

H0 CAVEAT (found during validation, worth knowing before it looks like a
bug): the SN likelihood is H0-marginalized (models.py fixes H0=70 as an
arbitrary fiducial — see model_definitions.md), but BAO genuinely measures
absolute distances via rd and IS sensitive to H0. Combining synthetic SN
(generated at H0=70) with REAL DESI BAO data at a fixed H0=70 in
per_probe_chi2 produced a chi2/N ~13 for BAO alone — not a bug, just real
DESI data disagreeing with an arbitrary H0=70 assumption. For a real joint
fit, either (a) add H0 as an explicit free parameter shared across probes
(physically correct, recommended), or (b) fix H0 to a value from an
external CMB-based prior rather than an arbitrary fiducial.
"""
import numpy as np


def build_joint_log_probability(
    model_name: str,
    probes: list[dict],
    priors: dict[str, tuple[float, float]] | None = None,
):
    """probes: list of dicts, each either
        {"kind": "sn", "z": ..., "mb_obs": ..., "cov_inv": ...}
        {"kind": "bao", "H0": ..., "rd_mode": "eh98"}  (uses DESI DR2 table
         internally via bao_desi.py)
    Both probes share the same underlying model (Om, w, w0/wa per
    model_name) — this is what makes the combination meaningful: BAO and
    SN are constraining the SAME E(z), so combining them breaks
    degeneracies neither can break alone (classic Om-w0-wa banana from SN
    alone narrows a lot once BAO's near-orthogonal degeneracy direction is
    added).

    Returns log_probability(theta) -> float, and param_names for the model.
    """
    from pramana.core.models import MODEL_REGISTRY
    from pramana.core.likelihood import log_likelihood as sn_log_likelihood, log_prior
    from pramana.core.bao_desi import log_likelihood_bao

    spec = MODEL_REGISTRY[model_name]
    param_names = spec["params"]
    if priors is None:
        priors = spec["priors"]

    def log_probability(theta):
        lp = log_prior(theta, param_names, priors)
        if not np.isfinite(lp):
            return -np.inf

        total = lp
        for probe in probes:
            if probe["kind"] == "sn":
                total += sn_log_likelihood(
                    theta, probe["z"], probe["mb_obs"], probe["cov_inv"],
                    spec["func"], param_names,
                )
            elif probe["kind"] == "bao":
                bao_kwargs = {"H0": probe.get("H0", 70.0), "rd_mode": probe.get("rd_mode", "eh98")}
                if "labels" in probe:
                    bao_kwargs.update({
                        "labels": probe["labels"], "z_arr": probe["z_arr"],
                        "data": probe["data"], "cov": probe["cov"],
                    })
                total += log_likelihood_bao(theta, spec["e_of_z"], param_names, **bao_kwargs)
            else:
                raise ValueError(f"Unknown probe kind: {probe['kind']!r}")

            if not np.isfinite(total):
                return -np.inf

        return total

    return log_probability, param_names


def per_probe_chi2(model_name: str, theta: np.ndarray, probes: list[dict]):
    """Diagnostic: chi2 contribution from EACH probe separately at a given
    theta — essential for spotting probe tension (e.g. SN wants Om=0.25,
    BAO wants Om=0.32 — the joint fit will report something in between
    with an inflated total chi2 that this breakdown makes visible, where
    the combined log_probability alone would hide it)."""
    from pramana.core.models import MODEL_REGISTRY
    from pramana.core.likelihood import log_likelihood as sn_log_likelihood
    from pramana.core.bao_desi import log_likelihood_bao

    spec = MODEL_REGISTRY[model_name]
    param_names = spec["params"]

    print(f"Per-probe chi2 at theta={dict(zip(param_names, theta))}:")
    for probe in probes:
        if probe["kind"] == "sn":
            ll = sn_log_likelihood(theta, probe["z"], probe["mb_obs"], probe["cov_inv"],
                                    spec["func"], param_names)
            n_data = len(probe["z"])
        elif probe["kind"] == "bao":
            bao_kwargs = {"H0": probe.get("H0", 70.0), "rd_mode": probe.get("rd_mode", "eh98")}
            if "labels" in probe:
                bao_kwargs.update({
                    "labels": probe["labels"], "z_arr": probe["z_arr"],
                    "data": probe["data"], "cov": probe["cov"],
                })
            ll = log_likelihood_bao(theta, spec["e_of_z"], param_names, **bao_kwargs)
            n_data = len(bao_kwargs.get("data", np.arange(13)))
        print(f"  {probe['kind']}: chi2 = {-2*ll:.2f}  ({n_data} data points, "
              f"chi2/N = {-2*ll/n_data:.2f})")