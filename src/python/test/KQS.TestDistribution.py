#!/usr/bin/env python3
import sys
import numpy as np
from scipy.stats import chi2_contingency, chi2


def load_histogram(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


# ===================================================================== #
#                        BIN MERGING FOR TESTS                           #
# ===================================================================== #

def merge_low_expected_bins(h1, h2, min_expected=5):
    """
    Merge adjacent bins until all expected frequencies >= min_expected.
    """
    h1 = list(map(float, h1))
    h2 = list(map(float, h2))

    while True:
        total1 = sum(h1)
        total2 = sum(h2)
        total = total1 + total2

        expected1 = [(total1 / total) * (a + b) for a, b in zip(h1, h2)]
        expected2 = [(total2 / total) * (a + b) for a, b in zip(h1, h2)]

        bad_bins = [
            i for i, (e1, e2) in enumerate(zip(expected1, expected2))
            if e1 < min_expected or e2 < min_expected
        ]

        if not bad_bins:
            break

        i = bad_bins[0]

        if i == 0:
            j = 1
        elif i == len(h1) - 1:
            j = i - 1
        else:
            j = i + 1

        h1[j] += h1[i]
        h2[j] += h2[i]

        del h1[i]
        del h2[i]

    return np.array(h1), np.array(h2)


# ===================================================================== #
#                        G-TEST (LOG-LIKELIHOOD)                          #
# ===================================================================== #

def g_test(h1, h2):
    h1 = np.array(h1, dtype=float)
    h2 = np.array(h2, dtype=float)

    total1 = h1.sum()
    total2 = h2.sum()
    total = total1 + total2

    expected1 = (total1 / total) * (h1 + h2)
    expected2 = (total2 / total) * (h1 + h2)

    if np.any(expected1 == 0) or np.any(expected2 == 0):
        raise ValueError("Zero expected frequency in G-test")

    mask1 = h1 > 0
    mask2 = h2 > 0

    G = 0.0
    G += 2 * np.sum(h1[mask1] * np.log(h1[mask1] / expected1[mask1]))
    G += 2 * np.sum(h2[mask2] * np.log(h2[mask2] / expected2[mask2]))

    dof = len(h1) - 1
    p_value = 1 - chi2.cdf(G, dof)

    return G, p_value, dof


# ===================================================================== #
#                    JENSEN–SHANNON DIVERGENCE                            #
# ===================================================================== #

def js_divergence(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    mask_p = p > 0
    mask_q = q > 0

    return (
        0.5 * np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p]))
        + 0.5 * np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q]))
    )


# ===================================================================== #
#               BOOTSTRAP SIGNIFICANCE FOR JSD                            #
# ===================================================================== #

def bootstrap_js_test(h1, h2, samples=1500):
    h1 = np.array(h1, dtype=float)
    h2 = np.array(h2, dtype=float)

    n1 = int(h1.sum())
    n2 = int(h2.sum())

    js_obs = js_divergence(h1, h2)
    combined = (h1 + h2) / (n1 + n2)

    extreme = 0
    for _ in range(samples):
        boot1 = np.random.multinomial(n1, combined)
        boot2 = np.random.multinomial(n2, combined)
        if js_divergence(boot1, boot2) >= js_obs:
            extreme += 1

    return js_obs, extreme / samples


# ===================================================================== #
#                     TOTAL VARIATION DISTANCE                            #
# ===================================================================== #

def total_variation_distance(h1, h2):
    p = np.array(h1, dtype=float)
    q = np.array(h2, dtype=float)
    p /= p.sum()
    q /= q.sum()
    return 0.5 * np.sum(np.abs(p - q))


# ===================================================================== #
#                           INTERPRETATION                               #
# ===================================================================== #

def interpret_js(js):
    if js < 0.001:
        return "🔵 Extremely close distributions (JSD < 0.001)"
    elif js < 0.01:
        return "🔵 Very similar distributions (JSD < 0.01)"
    elif js < 0.05:
        return "🟠 Moderate difference (JSD < 0.05)"
    else:
        return "🔴 Large distribution difference (JSD ≥ 0.05)"


def interpret_tvd(tvd):
    if tvd < 0.001:
        return "🔵 Almost perfect match (TVD < 0.001)"
    elif tvd < 0.01:
        return "🔵 Difference < 1% (TVD < 0.01)"
    elif tvd < 0.05:
        return "🟠 Noticeable difference (TVD < 0.05)"
    else:
        return "🔴 Significant difference (TVD ≥ 0.05)"


# ===================================================================== #
#                                 MAIN                                   #
# ===================================================================== #

def main():
    if len(sys.argv) != 3:
        print("Usage: python TestDistribution.py hist1.txt hist2.txt")
        sys.exit(1)

    h1 = load_histogram(sys.argv[1])
    h2 = load_histogram(sys.argv[2])

    if len(h1) != len(h2):
        print("Error: histogram files must have equal length.")
        sys.exit(1)

    # ---------------- Chi-square ---------------- #
    print("\n=== Chi-square test ===")
    try:
        h1_m, h2_m = merge_low_expected_bins(h1, h2, min_expected=5)
        table = np.array([h1_m, h2_m])

        chi2_stat, p, dof, _ = chi2_contingency(table)

        print(f"Chi-square statistic: {chi2_stat}")
        print(f"p-value:              {p}")
        print(f"Degrees of freedom:   {dof}")
        print(f"Bins merged:          {len(h1)} → {len(h1_m)}")

        print("🔴 FAIL" if p < 0.05 else "🟢 PASS")
    except Exception as e:
        print(f"🟡 Chi-square failed: {e}")

    # ---------------- G-test ---------------- #
    print("\n=== G-test (log-likelihood ratio) ===")
    try:
        h1_m, h2_m = merge_low_expected_bins(h1, h2, min_expected=1)
        G, p_g, dof_g = g_test(h1_m, h2_m)

        print(f"G statistic:          {G}")
        print(f"p-value:              {p_g}")
        print(f"Degrees of freedom:   {dof_g}")
        print(f"Bins merged:          {len(h1)} → {len(h1_m)}")

        print("🔴 FAIL" if p_g < 0.05 else "🟢 PASS")
    except Exception as e:
        print(f"🟡 G-test failed: {e}")

    # ---------------- Divergences ---------------- #
    js = js_divergence(np.array(h1, float), np.array(h2, float))
    tvd = total_variation_distance(h1, h2)

    print("\n=== Jensen–Shannon Divergence ===")
    print(f"JSD (bits): {js}")
    print(interpret_js(js))

    print("\n=== Total Variation Distance ===")
    print(f"TVD: {tvd}")
    print(interpret_tvd(tvd))

    # ---------------- Bootstrap ---------------- #
    print("\n=== Bootstrap JSD significance ===")
    js_obs, p_boot = bootstrap_js_test(h1, h2)
    print(f"Observed JSD:      {js_obs}")
    print(f"Bootstrap p-value: {p_boot}")

    print("🔴 FAIL" if p_boot < 0.05 else "🟢 PASS")


if __name__ == "__main__":
    main()
