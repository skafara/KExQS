import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.patheffects as pe


# ============================================================
#                 FORMATTING
# ============================================================

def fmt_log_scale_num(n, _=None):
    if n >= 1_024**2:
        return f"{n // (1_024**2)}M"
    if n >= 1_024:
        return f"{n // 1_024}K"
    return str(n)


# ============================================================
#                 GENERIC PLOT FUNCTION
# ============================================================

def plot_metric(df, metric, ylabel, title, outfile):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for i, name in enumerate(sorted(df["Name"].unique())):
        d = df[df["Name"] == name].sort_values("Shots")

        shots = d["Shots"].values
        values = d[metric].values
        color = colors[i % len(colors)]

        ax.plot(
            shots,
            values,
            color=color,
            linewidth=3,
            marker="o",
            markersize=6,
            label=name,
            path_effects=[pe.Stroke(linewidth=5, foreground="black"), pe.Normal()],
        )

    # ---------------- Axes ----------------
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_log_scale_num(int(v))))
    ax.xaxis.set_minor_locator(LogLocator(base=2, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))


    #ax.set_yscale("log")
    # ax.yaxis.set_major_locator(LogLocator(base=10))
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    # ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    # ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    ax.set_xlabel("Number of Shots")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.85)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.65)

    ax.legend(ncol=2, fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    #plt.show()


# ============================================================
#                         MAIN
# ============================================================

def main():
    df = pd.read_csv("results/KQS.TestDistribution.summary.csv")

    # -------- TVD plot --------
    plot_metric(
        df,
        metric="TVD",
        ylabel="Total Variation Distance (TVD)",
        title="Total Variation Distance\nPhilox vs Random.org",
        outfile="results/KQS.TestDistribution.TVD.png",
    )

    # -------- JSD plot --------
    plot_metric(
        df,
        metric="JSD",
        ylabel="Jensen–Shannon Divergence (bits)",
        title="Jensen–Shannon Divergence\nPhilox vs Random.org",
        outfile="results/KQS.TestDistribution.JSD.png",
    )


if __name__ == "__main__":
    main()