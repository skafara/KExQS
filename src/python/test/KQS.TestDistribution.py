import sys
import os
import csv
import numpy as np
from tqdm import tqdm

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
#                   HELPERS
# ============================================================

def load_samples(path):
    with open(path, "rb") as f:
        count = np.fromfile(f, dtype=np.uint64, count=1)[0]
        return np.fromfile(f, dtype=np.uint16, count=count).astype(np.int64, copy=False)


def format_shots(n):
    if n >= 1_024**2:
        return f"{n // (1_024**2)}M"
    if n >= 1_024:
        return f"{n // 1_024}K"
    return str(n)


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


def append_csv(path, name, shots, tvd, jsd):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Name", "Shots", "TVD", "JSD"])
        w.writerow([name, shots, tvd, jsd])


# ============================================================
#                   CONFIGURATION
# ============================================================

NUM_STATES = 1024
LOG_SHOTS_MIN = 10
LOG_SHOTS_MAX = 25
RESULTS_CSV = "results/KQS.TestDistribution.summary.csv"


# ============================================================
#                         MAIN
# ============================================================

def main():
    if len(sys.argv) != 3:
        print("Usage: script RandomOrg.txt Philox.txt")
        sys.exit(1)

    file_random = sys.argv[1]
    file_philox = sys.argv[2]

    name = os.path.basename(file_random)
    name = name.replace("KQS.TestDistribution.", "")
    name = name.replace(".RandomOrg.txt", "")

    print("Loading samples...")
    samples_random = load_samples(file_random)
    samples_philox = load_samples(file_philox)
    print("Samples loaded.")

    max_shots = min(len(samples_random), len(samples_philox))

    print(
        f"{'log2(N)':>7} | {'shots':>7} | {'TVD':>10} | "
        f"{'JSD(bits)':>12} | Interpretation"
    )
    print("-" * 75)

    for logN in range(LOG_SHOTS_MIN, LOG_SHOTS_MAX + 1):
        shots = 1 << logN
        if shots > max_shots:
            break

        # 🔥 THIS IS THE FAST PART 🔥
        h_random = samples_to_histogram(samples_random[:shots], NUM_STATES)
        h_philox = samples_to_histogram(samples_philox[:shots], NUM_STATES)

        p_random = probs_from_hist(h_random)
        p_philox = probs_from_hist(h_philox)

        tvd = tvd_from_probs(p_random, p_philox)
        jsd = jsd_from_probs(p_random, p_philox)

        append_csv(RESULTS_CSV, name, shots, tvd, jsd)

        print(
            f"{logN:7d} | {format_shots(shots):>7} | "
            f"{tvd:10.6g} | {jsd:12.6g} | {interpret_tvd(tvd)}"
        )


if __name__ == "__main__":
    main()
