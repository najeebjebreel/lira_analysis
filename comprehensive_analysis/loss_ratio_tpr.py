"""Generate Figure 7: loss ratio versus TPR.

Loss ratios and TPR@FPR values must be collected manually from per-run
summary CSVs produced by run_analysis.py across all experiments, then
assembled into a single CSV with columns "Loss Ratio" and "TPR" before
running this script.
"""

from __future__ import annotations

# NOTE: This script requires pre-aggregated multi-run CSV inputs that are NOT produced by a single
# benchmark run. See ARTIFACT-APPENDIX.md (Section: Reproducibility Analysis) for instructions on
# how to generate these inputs before running this script.

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from scipy.stats import pearsonr, spearmanr

try:
    from .plot_style import configure_popets_style
except ImportError:
    from plot_style import configure_popets_style


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "Loss_Ratio": "Loss Ratio",
            "loss_ratio": "Loss Ratio",
            "TPR_0.1FPR": "TPR",
            "tpr": "TPR",
            "TPR@0.1%FPR": "TPR",
        }
    )


def generate_plot(ratios: np.ndarray, tpr: np.ndarray, out_path: Path) -> None:
    """Plot loss ratio vs TPR@FPR scatter with log-scale x-axis.

    Parameters
    ----------
    ratios:
        1-D array of loss ratios (test loss / train loss), one value per run/experiment.
    tpr:
        1-D array of TPR@FPR values (in %), same length as ratios.
    out_path:
        Destination PDF path.
    """
    x = np.clip(np.asarray(ratios, dtype=float), 1e-12, None)
    y = np.asarray(tpr, dtype=float)
    logx = np.log10(x)

    pearson_r, pearson_p = pearsonr(logx, y)
    spearman_r, spearman_p = spearmanr(x, y)
    slope, intercept = np.polyfit(logx, y, 1)

    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    ax.scatter(x, y, s=12, facecolor="white", edgecolor="black", linewidth=0.55, marker="o")
    xx_log = np.linspace(logx.min(), logx.max(), 256)
    ax.plot(10**xx_log, slope * xx_log + intercept, color="black", linewidth=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Loss Ratio (Test / Train), log scale")
    ax.set_ylabel("TPR @ 0.1% FPR [%]")
    ax.grid(True, which="major", linestyle=":", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="plain")
    ax.text(
        0.02,
        0.98,
        f"Pearson r (log-x): {pearson_r:.2f} (p={pearson_p:.1e})\nSpearman \u03c1: {spearman_r:.2f} (p={spearman_p:.1e})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        bbox=dict(facecolor="white", edgecolor="0.6", linewidth=0.45, boxstyle="round,pad=0.2"),
    )

    fig.tight_layout(pad=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate the loss-ratio vs TPR figure")
    parser.add_argument("--csv", type=Path, default=repo_root / "analysis_results" / "loss_ratio.csv")
    parser.add_argument("--out", type=Path, default=repo_root / "analysis_results" / "figures" / "lossratio_tpr.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = normalize_columns(pd.read_csv(args.csv))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Loss Ratio", "TPR"])
    df = df.astype({"Loss Ratio": float, "TPR": float}).sort_values("Loss Ratio", kind="mergesort")
    configure_popets_style(base_fontsize=9.0, family="serif")
    generate_plot(df["Loss Ratio"].to_numpy(), df["TPR"].to_numpy(), args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
