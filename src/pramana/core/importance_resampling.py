"""Importance resampling: reweight an EXISTING chain (from MCMC, nested
sampling, or NUTS) to reflect a different likelihood or prior, without
rerunning the sampler. Use this for "how would adding this new dataset
shift my posterior" or "what if I'd used a different prior on wa" as a
fast approximate answer — valid as long as the new target isn't too far
from the chain's original distribution (see the effective-sample-size
diagnostic below, which tells you when it's failing).
"""
import numpy as np


def importance_weights(
    chain: np.ndarray,
    log_prob_old: np.ndarray,
    log_prob_new: np.ndarray,
) -> np.ndarray:
    """log_prob_old, log_prob_new: (N,) arrays — log-posterior of each
    chain sample under the ORIGINAL and NEW target respectively (NOT the
    parameters themselves; the caller evaluates these two functions on
    `chain` first). Returns normalized weights summing to 1."""
    log_w = log_prob_new - log_prob_old
    log_w -= log_w.max()  # numerical stability before exponentiating
    w = np.exp(log_w)
    return w / w.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's ESS: N_eff = (sum w)^2 / sum(w^2), for ALREADY-normalized
    weights this is 1 / sum(w^2). Rule of thumb: trust the reweighted
    posterior if N_eff is at least a few hundred, and at minimum >~5% of
    the original chain length — below that, a handful of samples are
    carrying all the weight and the reweighted posterior is unreliable
    no matter how large the original chain was."""
    return 1.0 / np.sum(weights**2)


def reweight_chain(
    chain: np.ndarray,
    log_prob_old: np.ndarray,
    log_prob_new: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """Full pipeline: compute weights, ESS, and a warning if reweighting
    is unreliable for this chain/target pair."""
    weights = importance_weights(chain, log_prob_old, log_prob_new)
    n_eff = effective_sample_size(weights)
    frac = n_eff / len(chain)

    if verbose:
        print(f"N_eff = {n_eff:.1f} / {len(chain)} samples ({frac:.1%})")
        if frac < 0.05:
            print("WARNING: effective sample size below 5% of chain length — "
                  "the new target is too far from the original distribution for "
                  "reliable reweighting. Rerun a real sampler on the new target instead.")

    return weights, n_eff


def weighted_quantiles(
    chain_param: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple = (0.16, 0.5, 0.84),
) -> np.ndarray:
    """Weighted quantiles of a single parameter column under the
    reweighted distribution (median +/- 1-sigma by default)."""
    order = np.argsort(chain_param)
    sorted_param = chain_param[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)
    cum_weights /= cum_weights[-1]
    return np.interp(quantiles, cum_weights, sorted_param)


def resample_to_equal_weight(
    chain: np.ndarray,
    weights: np.ndarray,
    n_samples: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Convert a weighted chain into an equal-weight sample (e.g. for
    feeding into corner_plot/getdist_triangle, which expect equal-weight
    samples) via multinomial resampling."""
    rng = np.random.default_rng(seed)
    if n_samples is None:
        n_samples = len(chain)
    idx = rng.choice(len(chain), size=n_samples, p=weights, replace=True)
    return chain[idx]