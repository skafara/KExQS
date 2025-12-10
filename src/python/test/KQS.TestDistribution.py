#!/usr/bin/env python3
import sys
import numpy as np
from scipy.stats import chi2_contingency


def load_histogram(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


# ---------------- G-TEST (log-likelihood ratio) ---------------- #

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

    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(G, dof)

    return G, p_value, dof


# ---------------- Jensen–Shannon Divergence ---------------- #

def js_divergence(p, q):
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    mask_p = p > 0
    mask_q = q > 0

    js = (
        0.5 * np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p]))
        + 0.5 * np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q]))
    )
    return js


# ---------------- Bootstrap Significance for JSD ---------------- #

def bootstrap_js_test(h1, h2, samples=2000):
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
        js_boot = js_divergence(boot1, boot2)
        if js_boot >= js_obs:
            extreme += 1

    p_value = extreme / samples
    return js_obs, p_value


# ---------------- Total Variation Distance ---------------- #

def total_variation_distance(h1, h2):
    p = np.array(h1, dtype=float)
    q = np.array(h2, dtype=float)
    p /= p.sum()
    q /= q.sum()
    return 0.5 * np.sum(np.abs(p - q))


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

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_histograms.py hist1.txt hist2.txt")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    h1 = load_histogram(file1)
    h2 = load_histogram(file2)

    if len(h1) != len(h2):
        print("Error: histogram files must contain the same number of lines.")
        sys.exit(1)

    table = np.array([h1, h2])

    # ---- Chi-square test ----
    print("\n=== Chi-square test ===")
    try:
        chi2, p, dof, expected = chi2_contingency(table)
        print(f"Chi-square statistic: {chi2}")
        print(f"p-value:              {p}")
        if p < 0.05:
            print("🔴 FAIL: Statistically significant difference detected (reject H0)")
        else:
            print("🟢 PASS: No significant difference (fail to reject H0)")
    except ValueError as e:
        print(f"🟡 Chi-square test skipped: {e}")

    # ---- G-test ----
    print("\n=== G-test (log-likelihood ratio) ===")
    try:
        G, p_g, dof_g = g_test(h1, h2)
        print(f"G statistic:          {G}")
        print(f"p-value:              {p_g}")
        if p_g < 0.05:
            print("🔴 FAIL: Statistically significant difference detected (reject H0)")
        else:
            print("🟢 PASS: No significant difference (fail to reject H0)")
    except ValueError as e:
        print(f"🟡 G-test skipped: {e}")

    # ---- Jensen–Shannon Divergence ----
    js = js_divergence(np.array(h1, float), np.array(h2, float))
    print("\n=== Jensen–Shannon Divergence ===")
    print(f"JSD (bits):           {js}")
    print(interpret_js(js))

    # ---- Total Variation Distance ----
    tvd = total_variation_distance(h1, h2)
    print("\n=== Total Variation Distance ===")
    print(f"TVD:                  {tvd}")
    print(interpret_tvd(tvd))

    # ---- Bootstrap JSD test ----
    print("\n=== Bootstrap significance test for JSD ===")
    js_obs, p_boot = bootstrap_js_test(h1, h2, samples=1500)
    print(f"Observed JSD:         {js_obs}")
    print(f"Bootstrap p-value:    {p_boot}")
    if p_boot < 0.05:
        print("🔴 FAIL: Divergence unusually large → distributions differ")
    else:
        print("🟢 PASS: Divergence explained by sampling noise → no difference")


if __name__ == "__main__":
    main()
