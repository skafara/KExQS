"""
Fast comparison of two QC measurement result files using:

- Total Variation Distance (TVD)
- Jensen–Shannon Divergence (JSD, bits)

INPUT:
  Two text files with one integer per line (raw outcomes),
  values in range [0, num_states-1].

DEFAULT:
  num_states = 1024 (10 qubits)

METHOD:
  - Build histograms
  - Compute TVD & JSD
  - Fast multinomial bootstrap for confidence intervals
  - Optional equivalence decision using TVD tolerance ε
"""

import argparse
import numpy as np


# ============================================================
#                         I/O
# ============================================================

def load_samples(path: str) -> np.ndarray:
    """Load raw integer outcomes from file."""
    with open(path, "r") as f:
        data = [int(line.strip()) for line in f if line.strip()]
    if not data:
        raise ValueError(f"No samples in {path}")
    return np.asarray(data, dtype=np.int64)


def samples_to_histogram(samples: np.ndarray, num_states: int) -> np.ndarray:
    """Convert raw outcomes to a dense histogram."""
    if samples.min() < 0 or samples.max() >= num_states:
        raise ValueError("Sample value out of allowed range")
    return np.bincount(samples, minlength=num_states)


# ============================================================
#                 PROBABILITY & DISTANCES
# ============================================================

def probs_from_hist(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float64)
    return h / h.sum()


def tvd_from_probs(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance."""
    return 0.5 * np.sum(np.abs(p - q))


def jsd_from_probs(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon Divergence (bits)."""
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    return 0.5 * (
        np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])) +
        np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q]))
    )


# ============================================================
#              FAST MULTINOMIAL BOOTSTRAP
# ============================================================

def bootstrap_ci_fast(
    h1: np.ndarray,
    h2: np.ndarray,
    n_boot: int = 2000,
    conf: float = 0.95,
    seed: int | None = None,
):
    """
    Fast bootstrap using multinomial resampling of histograms.
    Complexity: O(K * n_boot)
    """
    rng = np.random.default_rng(seed)

    n1 = int(h1.sum())
    n2 = int(h2.sum())

    p1 = probs_from_hist(h1)
    p2 = probs_from_hist(h2)

    tvds = np.empty(n_boot)
    jsds = np.empty(n_boot)

    for i in range(n_boot):
        b1 = rng.multinomial(n1, p1)
        b2 = rng.multinomial(n2, p2)

        pb1 = b1 / n1
        pb2 = b2 / n2

        tvds[i] = tvd_from_probs(pb1, pb2)
        jsds[i] = jsd_from_probs(pb1, pb2)

    alpha = 1.0 - conf
    lo = 100 * alpha / 2
    hi = 100 * (1 - alpha / 2)

    return (
        (np.percentile(tvds, lo), np.percentile(tvds, hi)),
        (np.percentile(jsds, lo), np.percentile(jsds, hi)),
    )


# ============================================================
#                     INTERPRETATION
# ============================================================

def interpret_tvd(tvd: float) -> str:
    if tvd < 0.001:
        return "🔵 Extremely close (<0.1% mass)"
    if tvd < 0.005:
        return "🔵 Close (<0.5% mass)"
    if tvd < 0.01:
        return "🟠 Noticeable (<1% mass)"
    if tvd < 0.05:
        return "🟠 Moderate (<5% mass)"
    return "🔴 Large difference (≥5% mass)"


def interpret_jsd(jsd: float) -> str:
    if jsd < 1e-4:
        return "🔵 Extremely close"
    if jsd < 1e-3:
        return "🔵 Very close"
    if jsd < 1e-2:
        return "🟠 Some difference"
    return "🔴 Large difference"


# ============================================================
#                         MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file1", help="First measurement file")
    ap.add_argument("file2", help="Second measurement file")
    ap.add_argument("--num-states", type=int, default=1024,
                    help="Number of possible outcomes (default: 1024)")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="Bootstrap samples (default: 2000)")
    ap.add_argument("--conf", type=float, default=0.95,
                    help="Confidence level (default: 0.95)")
    ap.add_argument("--eps", type=float, default=None,
                    help="TVD tolerance ε for equivalence decision")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # Load and histogram
    s1 = load_samples(args.file1)
    s2 = load_samples(args.file2)

    h1 = samples_to_histogram(s1, args.num_states)
    h2 = samples_to_histogram(s2, args.num_states)

    # Metrics
    p1 = probs_from_hist(h1)
    p2 = probs_from_hist(h2)

    tvd = tvd_from_probs(p1, p2)
    jsd = jsd_from_probs(p1, p2)

    tvd_ci, jsd_ci = bootstrap_ci_fast(
        h1, h2,
        n_boot=args.bootstrap,
        conf=args.conf,
        seed=args.seed
    )

    # Output
    print("\n=== INPUT ===")
    print(f"File 1 samples: {s1.size}")
    print(f"File 2 samples: {s2.size}")
    print(f"States: {args.num_states}")

    print("\n=== TOTAL VARIATION DISTANCE (TVD) ===")
    print(f"TVD: {tvd:.8g}")
    print(f"{int(args.conf*100)}% CI: [{tvd_ci[0]:.8g}, {tvd_ci[1]:.8g}]")
    print(interpret_tvd(tvd))

    print("\n=== JENSEN–SHANNON DIVERGENCE (JSD) ===")
    print(f"JSD (bits): {jsd:.8g}")
    print(f"{int(args.conf*100)}% CI: [{jsd_ci[0]:.8g}, {jsd_ci[1]:.8g}]")
    print(interpret_jsd(jsd))

    print("\n=== EQUIVALENCE DECISION (TVD-based) ===")
    if args.eps is None:
        print("No ε provided → informational only")
    else:
        lo, hi = tvd_ci
        print(f"Tolerance ε = {args.eps}")
        if hi <= args.eps:
            print("🟢 SAME: Equivalent within tolerance")
        elif lo > args.eps:
            print("🔴 DIFFERENT: Beyond tolerance")
        else:
            print("🟡 INCONCLUSIVE: CI overlaps ε")


if __name__ == "__main__":
    main()
