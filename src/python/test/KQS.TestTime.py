import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import matplotlib.patheffects as pe


def fmt_log_scale_num(n):
    if n >= 1_024**2:
        return f"{n // (1_024**2)}M"
    if n >= 1_024:
        return f"{n // 1_024}K"
    return str(n)


def fmt_speedup(val):
    s = f"{val:.2f}".rstrip("0").rstrip(".")
    return s + "×"


def plot_numshots_performance(func_name: str):
    dataSeq = pd.read_csv(f"results/KQS.TestTime.{func_name}.Sequential.txt", sep="\t")
    dataPar = pd.read_csv(f"results/KQS.TestTime.{func_name}.Parallel.txt", sep="\t")
    dataAcc = pd.read_csv(f"results/KQS.TestTime.{func_name}.Accelerated.txt", sep="\t")

    shots = dataSeq["NumShots"].values

    # Convert from ns → ms
    seq = dataSeq["Mean_ns"].values / 1e6
    par = dataPar["Mean_ns"].values / 1e6
    gpu = dataAcc["Mean_ns"].values / 1e6

    # Compute speedups
    speedup_par = seq / par
    speedup_gpu = seq / gpu

    x = np.arange(len(shots))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 9))

    # --- BAR CHART ---
    ax.bar(x - width, seq, width, label="Sequential", color="#999999")
    ax.bar(x,         par, width, label="Parallel + SIMD", color="#4C72B0")
    ax.bar(x + width, gpu, width, label="GPU", color="#DD8452")

    # --- Left axis: Runtime ---
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    ax.set_xlabel("Number of Shots")
    ax.set_ylabel("Time [ms]")
    ax.set_title(f"Execution Mode Performance: {func_name}")

    labels = [fmt_log_scale_num(n) for n in shots]
    ax.set_xticks(x, labels)
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_xticklabels(), rotation=0)

    # --- Right axis: Speedup ---
    ax2 = ax.twinx()
    ax2.set_yscale("log")

    ax2.plot(
        x, speedup_par,
        color="#4C72B0", marker="o", linestyle="-", linewidth=3,
        markersize=6,
        path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()]
    )

    ax2.plot(
        x, speedup_gpu,
        color="#DD8452", marker="o", linestyle="-", linewidth=3,
        markersize=6,
        path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()]
    )

    # ---------- ONLY vertical line marking first speedup > 1 ----------
    idx_par = np.argmax(speedup_par > 1)
    idx_par = idx_par if speedup_par[idx_par] > 1 else None

    idx_gpu = np.argmax(speedup_gpu > 1)
    idx_gpu = idx_gpu if speedup_gpu[idx_gpu] > 1 else None

    if idx_par is not None:
        ax.axvline(x[idx_par], linestyle="--", color="#4C72B0", alpha=1)

    if idx_gpu is not None:
        ax.axvline(x[idx_gpu], linestyle="--", color="#DD8452", alpha=1)
    # -------------------------------------------------------------------

    # Speedup ticks and custom final ticks
    major_ticks = list(LogLocator(base=10).tick_values(
        min(speedup_par.min(), speedup_gpu.min()),
        max(speedup_par.max(), speedup_gpu.max())
    ))

    final_par = speedup_par[-1]
    final_gpu = speedup_gpu[-1]

    if final_par not in major_ticks:
        major_ticks.append(final_par)
    if final_gpu not in major_ticks:
        major_ticks.append(final_gpu)

    major_ticks = sorted(major_ticks)

    ax2.set_yticks(major_ticks)
    ax2.set_yticklabels([fmt_speedup(t) for t in major_ticks])

    ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax2.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    ax2.set_ylabel("Speedup [×]")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"results/KQS.TestTime.{func_name}.png")


def plot_numstates_performance(func_name: str):
    dataSeq = pd.read_csv(f"results/KQS.TestTime.{func_name}.Sequential.txt", sep="\t")
    dataPar = pd.read_csv(f"results/KQS.TestTime.{func_name}.Parallel.txt", sep="\t")
    dataAcc = pd.read_csv(f"results/KQS.TestTime.{func_name}.Accelerated.txt", sep="\t")

    states = dataSeq["NumStates"].values

    # Convert from ns → ms
    seq = dataSeq["Mean_ns"].values / 1e6
    par = dataPar["Mean_ns"].values / 1e6
    gpu = dataAcc["Mean_ns"].values / 1e6

    # Compute speedups
    speedup_par = seq / par
    speedup_gpu = seq / gpu

    x = np.arange(len(states))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 9))

    # --- BAR CHART ---
    ax.bar(x - width, seq, width, label="Sequential", color="#999999")
    ax.bar(x,         par, width, label="Parallel + SIMD", color="#4C72B0")
    ax.bar(x + width, gpu, width, label="GPU", color="#DD8452")

    # --- Left axis: Runtime ---
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    ax.set_xlabel("Number of States")
    ax.set_ylabel("Time [ms]")
    ax.set_title(f"Execution Mode Performance: {func_name}")

    labels = [fmt_log_scale_num(n) for n in states]
    ax.set_xticks(x, labels)
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_xticklabels(), rotation=0)

    # --- Right axis: Speedup ---
    ax2 = ax.twinx()
    ax2.set_yscale("log")

    ax2.plot(
        x, speedup_par,
        color="#4C72B0", marker="o", linestyle="-", linewidth=3,
        markersize=6,
        path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()]
    )

    ax2.plot(
        x, speedup_gpu,
        color="#DD8452", marker="o", linestyle="-", linewidth=3,
        markersize=6,
        path_effects=[pe.Stroke(linewidth=4, foreground="black"), pe.Normal()]
    )

    # ---------- ONLY vertical line marking first speedup > 1 ----------
    idx_par = np.argmax(speedup_par > 1)
    idx_par = idx_par if speedup_par[idx_par] > 1 else None

    idx_gpu = np.argmax(speedup_gpu > 1)
    idx_gpu = idx_gpu if speedup_gpu[idx_gpu] > 1 else None

    if idx_par is not None:
        ax.axvline(x[idx_par], linestyle="--", color="#4C72B0", alpha=1)

    if idx_gpu is not None:
        ax.axvline(x[idx_gpu], linestyle="--", color="#DD8452", alpha=1)
    # -------------------------------------------------------------------

    # Speedup ticks and custom final ticks
    major_ticks = list(LogLocator(base=10).tick_values(
        min(speedup_par.min(), speedup_gpu.min()),
        max(speedup_par.max(), speedup_gpu.max())
    ))

    final_par = speedup_par[-1]
    final_gpu = speedup_gpu[-1]

    if final_par not in major_ticks:
        major_ticks.append(final_par)
    if final_gpu not in major_ticks:
        major_ticks.append(final_gpu)

    major_ticks = sorted(major_ticks)

    ax2.set_yticks(major_ticks)
    ax2.set_yticklabels([fmt_speedup(t) for t in major_ticks])

    ax2.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
    ax2.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    ax2.set_ylabel("Speedup [×]")

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(f"results/KQS.TestTime.{func_name}.png")


def main():
    plot_numstates_performance("_CalculateProbabilities")
    plot_numshots_performance("GenerateRandomDiscrete")
    plot_numshots_performance("GenerateRandomContinuous")
    plot_numshots_performance("_SampleAliasTable")


if __name__ == "__main__":
    main()
