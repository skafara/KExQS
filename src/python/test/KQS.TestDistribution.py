import numpy as np

# ============================================================
#                 DISTANCES
# ============================================================

def tvd_from_probs(p, q):
    return 0.5 * np.sum(np.abs(p - q))


def jsd_from_probs(p, q):
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    return 0.5 * (
        np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])) +
        np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q]))
    )


def probs_from_hist(h):
    return h / h.sum()


def samples_to_histogram(samples, num_states):
    return np.bincount(samples, minlength=num_states)


# ============================================================
#              FAST MULTINOMIAL BOOTSTRAP
# ============================================================

def bootstrap_ci_fast(h1, h2, n_boot=500, conf=0.95, seed=0):
    rng = np.random.default_rng(seed)

    n1 = int(h1.sum())
    n2 = int(h2.sum())

    p1 = probs_from_hist(h1)
    p2 = probs_from_hist(h2)

    tvds = np.empty(n_boot)

    for i in range(n_boot):
        b1 = rng.multinomial(n1, p1)
        b2 = rng.multinomial(n2, p2)

        pb1 = b1 / n1
        pb2 = b2 / n2

        tvds[i] = tvd_from_probs(pb1, pb2)

    alpha = 1.0 - conf
    lo = 100 * alpha / 2
    hi = 100 * (1 - alpha / 2)

    return np.percentile(tvds, lo), np.percentile(tvds, hi)


# ============================================================
#                   INTERPRETATION
# ============================================================

def interpret_tvd(tvd):
    if tvd < 0.001:
        return "Practically identical"
    if tvd < 0.005:
        return "Very close"
    if tvd < 0.01:
        return "Close / acceptable"
    if tvd < 0.05:
        return "Noticeable difference"
    return "Large systematic difference"


def format_shots(n: int) -> str:
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n // 1_000}K"
    if n < 1_000_000_000:
        return f"{n // 1_000_000}M"
    return f"{n // 1_000_000_000}B"


# ============================================================
#                   CONFIGURATION
# ============================================================

NUM_STATES = 1024
SEED_PHILOX = 123
SEED_RANDOMORG = 999

LOG_SHOTS_MIN = 10   # 1K
LOG_SHOTS_MAX = 28   # 256M

BOOTSTRAP_MAX_SHOTS = 2**20
BOOTSTRAP_SAMPLES = 500


# ============================================================
#            SAMPLE GENERATORS (PLACEHOLDERS)
# ============================================================

def sample_philox(num_states, shots, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_states, size=shots)


def sample_randomorg(num_states, shots, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_states, size=shots)


# ============================================================
#                       EXPERIMENT
# ============================================================

def run_log_sweep():
    print(
        f"{'log2(N)':>7} | {'shots':>7} | {'TVD':>10} | "
        f"{'JSD(bits)':>12} | {'TVD CI':>17} | Interpretation"
    )
    print("-" * 90)

    for logN in range(LOG_SHOTS_MIN, LOG_SHOTS_MAX + 1):
        shots = 1 << logN

        s_philox = sample_philox(NUM_STATES, shots, SEED_PHILOX)
        s_random = sample_randomorg(NUM_STATES, shots, SEED_RANDOMORG)

        h_philox = samples_to_histogram(s_philox, NUM_STATES)
        h_random = samples_to_histogram(s_random, NUM_STATES)

        p_philox = probs_from_hist(h_philox)
        p_random = probs_from_hist(h_random)

        tvd = tvd_from_probs(p_philox, p_random)
        jsd = jsd_from_probs(p_philox, p_random)

        if shots <= BOOTSTRAP_MAX_SHOTS:
            lo, hi = bootstrap_ci_fast(
                h_philox,
                h_random,
                n_boot=BOOTSTRAP_SAMPLES
            )
            ci_str = f"[{lo:.3g}, {hi:.3g}]"
        else:
            ci_str = "—"

        print(
            f"{logN:7d} | {format_shots(shots):>7} | "
            f"{tvd:10.6g} | {jsd:12.6g} | {ci_str:>17} | {interpret_tvd(tvd)}"
        )


# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":
    run_log_sweep()
