"""
Threshold distribution analysis script.

This script creates box plots showing threshold distributions across
leave-one-out target models for the LiRA attack.

Usage:
    python threshold_dist.py --experiment_dir PATH_TO_EXPERIMENT [--output OUTPUT_PATH]
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Import from local modules
from analysis_utils import load_threshold_info, get_experiment_info
from visualization import COLORS, setup_paper_style
from metrics import compute_median_and_rmad


def load_thresholds_for_attack(threshold_df: pd.DataFrame,
                                target_fpr: float,
                                attack_contains: str = "online") -> np.ndarray:
    """
    Load threshold values for a specific attack and target FPR.

    Args:
        threshold_df: Threshold information DataFrame
        target_fpr: Target FPR value
        attack_contains: Substring to match in attack name

    Returns:
        Array of threshold values
    """
    threshold_df["target_fpr"] = pd.to_numeric(threshold_df["target_fpr"], errors="coerce")
    m_attack = threshold_df["attack"].astype(str).str.lower().str.contains(
        attack_contains.lower()
    )
    m_tfpr = np.isclose(
        threshold_df["target_fpr"].to_numpy(), float(target_fpr),
        rtol=1e-6, atol=1e-12
    )

    vals = pd.to_numeric(
        threshold_df.loc[m_attack & m_tfpr, "threshold"], errors="coerce"
    ).to_numpy()
    vals = vals[np.isfinite(vals)]

    return vals


def plot_threshold_distribution(experiment_dir: str,
                                 target_fpr: float = 1e-5,
                                 output_path: str = None,
                                 attack_name: str = "online",
                                 show_fliers: bool = False,
                                 whisker_mode: str = "tukey") -> str:
    """
    Create box plot of threshold distribution.

    Args:
        experiment_dir: Path to experiment directory
        target_fpr: Target FPR value
        output_path: Output path for plot (auto-generated if None)
        attack_name: Attack name substring to match
        show_fliers: Whether to show outliers
        whisker_mode: "tukey" (1.5*IQR) or "p05p95" (5-95% percentiles)

    Returns:
        Path to saved figure
    """
    experiment_dir = Path(experiment_dir)

    # Load threshold information
    threshold_df = load_threshold_info(experiment_dir)

    # Load thresholds for the specified attack
    thresholds = load_thresholds_for_attack(threshold_df, target_fpr, attack_name)

    if len(thresholds) == 0:
        raise ValueError(f"No thresholds found for attack='{attack_name}' at FPR={target_fpr}")

    # Get experiment info for title
    exp_info = get_experiment_info(experiment_dir)

    # Setup plotting style
    setup_paper_style()

    # Create plot
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.subplots_adjust(left=0.15, right=0.96, top=0.92, bottom=0.15)

    whis = (5, 95) if whisker_mode == "p05p95" else 1.5

    bp = ax.boxplot(
        [thresholds],
        vert=True,
        patch_artist=True,
        widths=0.55,
        whis=whis,
        showfliers=show_fliers
    )

    # Style the box
    bp['boxes'][0].set_facecolor(COLORS['blue'] + '33')
    bp['boxes'][0].set_edgecolor(COLORS['blue'])
    bp['boxes'][0].set_linewidth(1.0)
    bp['medians'][0].set_color(COLORS['vermillion'])
    bp['medians'][0].set_linewidth(1.2)

    for item in ['whiskers', 'caps']:
        for line in bp[item]:
            line.set_color(COLORS['blue'])
            line.set_linewidth(0.9)

    # Add statistics annotation
    median, rmad = compute_median_and_rmad(thresholds)
    q3 = float(np.percentile(thresholds, 75))

    ax.annotate(
        f"Median = {median:.2f}\\nrMAD = {rmad:.1f}%\\nn = {len(thresholds)}",
        xy=(1, q3),
        xytext=(10, 15),
        textcoords="offset points",
        ha="left", va="bottom",
        fontsize=7, color="#1A2732",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        clip_on=False
    )

    # Labels
    ax.set_xticks([1])
    ax.set_xticklabels([f"LiRA ({attack_name})"])
    ax.set_ylabel(r"Threshold $\tau$")

    dataset = exp_info.get('dataset', 'unknown')
    model = exp_info.get('model', 'unknown')
    ax.set_title(f"{dataset.upper()} - {model}", fontsize=9)

    ax.grid(True, alpha=0.3, axis='y')

    # Determine output path
    if output_path is None:
        output_path = experiment_dir / f"threshold_distribution_{attack_name}_fpr{target_fpr:.0e}.pdf"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)

    print(f"Saved threshold distribution plot to {output_path}")
    return str(output_path)


def main():
    """Command-line interface for threshold distribution analysis."""
    parser = argparse.ArgumentParser(
        description="Plot threshold distributions from LiRA leave-one-out evaluation"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: auto-generated in experiment dir)"
    )
    parser.add_argument(
        "--target_fpr",
        type=float,
        default=1e-5,
        help="Target FPR value (default: 1e-5)"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="online",
        help="Attack name substring to match (default: online)"
    )
    parser.add_argument(
        "--show_fliers",
        action="store_true",
        help="Show outliers in box plot"
    )
    parser.add_argument(
        "--whisker_mode",
        type=str,
        choices=["tukey", "p05p95"],
        default="tukey",
        help="Whisker mode: tukey (1.5*IQR) or p05p95 (5-95 percentiles)"
    )

    args = parser.parse_args()

    plot_threshold_distribution(
        experiment_dir=args.experiment_dir,
        target_fpr=args.target_fpr,
        output_path=args.output,
        attack_name=args.attack,
        show_fliers=args.show_fliers,
        whisker_mode=args.whisker_mode
    )


if __name__ == "__main__":
    main()
