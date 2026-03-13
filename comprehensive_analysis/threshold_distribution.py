"""Generate Figure 1 threshold boxplots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from .plot_style import configure_popets_style
except ImportError:
    from plot_style import configure_popets_style


OI_BLUE = "#0072B2"
OI_ORANGE = "#E69F00"
OI_VERMIL = "#D55E00"


def _eq_str(series: pd.Series, target: str) -> pd.Series:
    return series.astype(str).str.strip().str.casefold() == str(target).strip().casefold()


def load_thresholds_filtered(
    csv_path: Path,
    attack: str,
    target_fpr: float,
    prior_equals: float = 0.01,
    mode_equals: str = "target",
) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df["target_fpr"] = pd.to_numeric(df.get("target_fpr"), errors="coerce")
    df["prior"] = pd.to_numeric(df.get("prior"), errors="coerce")

    mask = (
        _eq_str(df["mode"], mode_equals)
        & _eq_str(df["attack"], attack)
        & np.isclose(df["target_fpr"].to_numpy(), float(target_fpr), rtol=1e-6, atol=1e-12)
        & np.isclose(df["prior"].to_numpy(), float(prior_equals), rtol=1e-6, atol=1e-12)
    )
    values = pd.to_numeric(df.loc[mask, "threshold"], errors="coerce").to_numpy()
    return values[np.isfinite(values)]


def load_pooled_thresholds(csv_paths: list[Path], attack: str, target_fpr: float, prior: float = 0.01) -> np.ndarray:
    chunks = []
    for csv_path in csv_paths:
        values = load_thresholds_filtered(csv_path, attack=attack, target_fpr=target_fpr, prior_equals=prior)
        if values.size:
            chunks.append(values)
    if not chunks:
        raise ValueError(f"No thresholds found for attack={attack!r} target_fpr={target_fpr}")
    return np.concatenate(chunks)


def _median_rmad(vals: np.ndarray) -> tuple[float, float]:
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    rmad = 100.0 * 1.4826 * mad / med if med != 0 else np.nan
    return med, rmad


def plot_online_thresholds(csv_paths: list[Path], out_path: Path, whisker_mode: str = "tukey", show_fliers: bool = False) -> None:
    attack = "LiRA (online)"
    datasets = [
        load_pooled_thresholds(csv_paths, attack=attack, target_fpr=1e-5),
        load_pooled_thresholds(csv_paths, attack=attack, target_fpr=1e-3),
    ]
    colors = [OI_BLUE, OI_ORANGE]
    fills = [color + "33" for color in colors]
    labels = [r"Online, $10^{-5}$", r"Online, $10^{-3}$"]
    positions = [0.7, 1.05]

    whis = (5, 95) if whisker_mode == "p05p95" else 1.5
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    plt.subplots_adjust(left=0.14, right=0.995, top=0.95, bottom=0.26)

    bp = ax.boxplot(
        datasets,
        positions=positions,
        widths=0.2,
        whis=whis,
        patch_artist=True,
        showfliers=show_fliers,
    )
    for patch, fill_color, edge_color in zip(bp["boxes"], fills, colors):
        patch.set_facecolor(fill_color)
        patch.set_edgecolor(edge_color)
        patch.set_linewidth(1.0)
    for med in bp["medians"]:
        med.set_color(OI_VERMIL)
        med.set_linewidth(1.2)
    for i, line in enumerate(bp["whiskers"]):
        line.set_color(colors[i // 2])
        line.set_linewidth(0.9)
    for i, line in enumerate(bp["caps"]):
        line.set_color(colors[i // 2])
        line.set_linewidth(0.9)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(0.5, 1.25)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.45)
    ax.tick_params(width=0.6)

    for pos_idx, (xpos, vals) in enumerate(zip(positions, datasets)):
        med, rmad = _median_rmad(vals)
        q3 = float(np.percentile(vals, 75))
        offset_x, offset_y = (8, 5) if pos_idx == 0 else (25, 8)
        ax.annotate(
            f"Median = {med:.2f}\nrMAD = {rmad:.1f}%",
            xy=(xpos, q3),
            xytext=(offset_x, offset_y), textcoords="offset points",
            ha="left" if pos_idx == 0 else "right", va="bottom",
            fontsize=6.5, color="#1A2732",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.4),
            clip_on=False,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Generate LiRA threshold distribution boxplots")
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        type=Path,
        default=[repo_root / "analysis_results" / "cifar10" / "resnet18" / f"seed{i}" / "per_model_metrics_two_modes.csv" for i in range(1, 13)],
    )
    parser.add_argument(
        "--single-input",
        type=Path,
        default=repo_root / "analysis_results" / "cifar10" / "resnet18" / "seed4" / "per_model_metrics_two_modes.csv",
    )
    parser.add_argument("--single-out", type=Path, default=repo_root / "analysis_results" / "figures" / "thresh_boxplot_single.pdf")
    parser.add_argument("--multi-out", type=Path, default=repo_root / "analysis_results" / "figures" / "thresh_boxplot_multi.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_popets_style(base_fontsize=12, family="serif")
    sns.set_theme(context="paper", style="whitegrid")

    plot_online_thresholds([args.single_input], args.single_out)
    plot_online_thresholds(list(args.input_csvs), args.multi_out)
    print(f"Saved: {args.single_out}")
    print(f"Saved: {args.multi_out}")


if __name__ == "__main__":
    main()
