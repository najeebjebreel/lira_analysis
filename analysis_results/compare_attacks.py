"""
Compare multiple attack variants and generate comparison plots.

This script loads attack results from an experiment and creates:
- ROC curve comparisons
- Score distribution plots
- Metrics comparison bar charts

Usage:
    python compare_attacks.py --experiment_dir PATH_TO_EXPERIMENT [--output_dir OUTPUT_DIR]
"""

import argparse
import numpy as np
from pathlib import Path

# Import from local modules
from analysis_utils import (
    load_attack_scores,
    load_membership_labels,
    load_metrics_csv,
    get_experiment_info
)
from metrics import compute_roc_metrics
from visualization import (
    plot_roc_curves,
    plot_score_distributions,
    plot_comparison_bars,
    setup_paper_style
)


def compare_attacks(experiment_dir: str,
                    output_dir: str = None,
                    target_model_idx: int = 0) -> dict:
    """
    Compare all attack variants for a given experiment.

    Args:
        experiment_dir: Path to experiment directory
        output_dir: Output directory for plots (defaults to experiment_dir)
        target_model_idx: Which model to use as target for single plots

    Returns:
        Dictionary with paths to generated files
    """
    experiment_dir = Path(experiment_dir)
    if output_dir is None:
        output_dir = experiment_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    setup_paper_style()

    # Load data
    print("Loading attack scores...")
    scores_dict = load_attack_scores(experiment_dir, mode='leave_one_out')
    labels = load_membership_labels(experiment_dir)

    if len(scores_dict) == 0:
        raise ValueError("No attack scores found in experiment directory")

    print(f"Found {len(scores_dict)} attack variants")
    print(f"Loaded scores for {labels.shape[0]} models, {labels.shape[1]} samples")

    # Get experiment info
    exp_info = get_experiment_info(experiment_dir)
    dataset = exp_info.get('dataset', 'unknown')
    model = exp_info.get('model', 'unknown')

    output_files = {}

    # 1. ROC Curve Comparison
    print("\nGenerating ROC curve comparison...")
    roc_data = {}
    for attack_name, scores in scores_dict.items():
        # Use specified target model
        target_scores = scores[target_model_idx]
        target_labels = labels[target_model_idx]

        fpr, tpr, thresholds, auc_score, accuracy = compute_roc_metrics(
            target_scores, target_labels
        )
        roc_data[attack_name] = (fpr, tpr, auc_score)

    roc_path = output_dir / "roc_comparison.pdf"
    plot_roc_curves(
        roc_data,
        save_path=roc_path,
        title=f"ROC Curves - {dataset.upper()} {model} (Model {target_model_idx})"
    )
    output_files['roc_curves'] = str(roc_path)

    # 2. Score Distributions
    print("Generating score distribution plots...")
    dist_dir = output_dir / "distributions"
    dist_dir.mkdir(exist_ok=True)

    for attack_name, scores in scores_dict.items():
        target_scores = scores[target_model_idx]
        target_labels = labels[target_model_idx]

        member_scores = target_scores[target_labels]
        non_member_scores = target_scores[~target_labels]

        dist_path = dist_dir / f"distribution_{attack_name.replace(' ', '_')}.pdf"
        plot_score_distributions(
            member_scores,
            non_member_scores,
            attack_name=attack_name,
            save_path=dist_path
        )

    output_files['distributions'] = str(dist_dir)

    # 3. Metrics Comparison
    print("Generating metrics comparison...")
    try:
        metrics_df = load_metrics_csv(experiment_dir, mode='leave_one_out')

        # Extract AUC for comparison
        auc_data = {}
        for attack in scores_dict.keys():
            attack_data = metrics_df[metrics_df['Attack'] == attack]
            if not attack_data.empty and 'AUC Mean' in attack_data.columns:
                auc_data[attack] = attack_data['AUC Mean'].iloc[0]

        if auc_data:
            auc_path = output_dir / "auc_comparison.pdf"
            plot_comparison_bars(
                auc_data,
                save_path=auc_path,
                ylabel="AUC (%)",
                title=f"AUC Comparison - {dataset.upper()} {model}"
            )
            output_files['auc_comparison'] = str(auc_path)

    except FileNotFoundError:
        print("Warning: Metrics CSV not found, skipping metrics comparison")

    print(f"\nGenerated {len(output_files)} output files in {output_dir}")
    return output_files


def main():
    """Command-line interface for attack comparison."""
    parser = argparse.ArgumentParser(
        description="Compare LiRA attack variants and generate plots"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: experiment_dir)"
    )
    parser.add_argument(
        "--target_model",
        type=int,
        default=0,
        help="Target model index for single plots (default: 0)"
    )

    args = parser.parse_args()

    compare_attacks(
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
        target_model_idx=args.target_model
    )


if __name__ == "__main__":
    main()
