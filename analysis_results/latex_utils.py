"""
LaTeX table generation utilities for LiRA attack analysis.

This module provides functions for:
- Generating publication-ready LaTeX tables
- Formatting metrics with proper precision
- Creating comparison tables across benchmarks
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    """
    Format mean ± std for LaTeX.

    Args:
        mean: Mean value
        std: Standard deviation
        digits: Number of decimal digits

    Returns:
        LaTeX-formatted string
    """
    if pd.isna(mean) or pd.isna(std):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def format_multiplier(reduction_factor: float, digits: int = 1) -> str:
    """
    Format reduction factor as multiplier for LaTeX.

    Args:
        reduction_factor: baseline / variant
        digits: Number of decimal digits

    Returns:
        LaTeX-formatted string like "($\\times$5.2)"
    """
    if reduction_factor is None or np.isinf(reduction_factor) or np.isnan(reduction_factor):
        return "($\\times\\infty$)"

    if reduction_factor >= 10:
        return f"($\\times${int(round(reduction_factor))})"

    return f"($\\times${reduction_factor:.{digits}f})"


def create_metrics_table(results_df: pd.DataFrame,
                         attacks: List[str],
                         target_fprs: List[float],
                         caption: str,
                         label: str,
                         include_auc: bool = True,
                         include_accuracy: bool = True) -> str:
    """
    Create a LaTeX table of attack metrics.

    Args:
        results_df: DataFrame with attack results
        attacks: List of attack names to include
        target_fprs: List of target FPR values
        caption: Table caption
        label: Table label
        include_auc: Whether to include AUC column
        include_accuracy: Whether to include accuracy column

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    # Build column format
    n_cols = 1  # Attack name
    if include_auc:
        n_cols += 1
    if include_accuracy:
        n_cols += 1
    n_cols += len(target_fprs) * 2  # TPR and Precision for each FPR

    col_format = "l" + "c" * (n_cols - 1)
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    lines.append("\\toprule")

    # Header
    header = ["Attack"]
    if include_auc:
        header.append("AUC (\\%)")
    if include_accuracy:
        header.append("Acc (\\%)")
    for fpr in target_fprs:
        fpr_pct = fpr * 100
        header.append(f"TPR@{fpr_pct:.4g}\\%FPR")
        header.append(f"Prec@{fpr_pct:.4g}\\%FPR")

    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for attack in attacks:
        row_data = results_df[results_df['Attack'] == attack]
        if row_data.empty:
            continue

        row = [attack]

        if include_auc:
            auc_val = row_data['AUC'].iloc[0] if 'AUC' in row_data.columns else np.nan
            row.append(f"{auc_val:.2f}" if not np.isnan(auc_val) else "--")

        if include_accuracy:
            acc_val = row_data['Acc'].iloc[0] if 'Acc' in row_data.columns else np.nan
            row.append(f"{acc_val:.2f}" if not np.isnan(acc_val) else "--")

        for fpr in target_fprs:
            fpr_pct = fpr * 100
            tpr_col = f"TPR@{fpr_pct:.4g}%FPR"
            prec_col = f"Prec@{fpr_pct:.4g}%FPR"

            tpr_val = row_data[tpr_col].iloc[0] if tpr_col in row_data.columns else np.nan
            prec_val = row_data[prec_col].iloc[0] if prec_col in row_data.columns else np.nan

            row.append(f"{tpr_val:.3f}" if not np.isnan(tpr_val) else "--")
            row.append(f"{prec_val:.2f}" if not np.isnan(prec_val) else "--")

        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def create_comparison_table(benchmark_results: Dict[str, pd.DataFrame],
                            attacks: List[str],
                            target_fpr: float,
                            prior: float,
                            caption: str,
                            label: str,
                            baseline_name: Optional[str] = None) -> str:
    """
    Create a LaTeX table comparing multiple benchmarks.

    Args:
        benchmark_results: Dictionary mapping benchmark names to results DataFrames
        attacks: List of attack names
        target_fpr: Target FPR value
        prior: Membership prior
        caption: Table caption
        label: Table label
        baseline_name: Name of baseline benchmark for reduction factors

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table*}[!ht]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")

    fpr_pct = target_fpr * 100
    lines.append(f"Benchmark & Attack & TPR@{fpr_pct:.4g}\\%FPR (\\%) & Prec@{prior*100:.0f}\\% Prior (\\%) & AUC (\\%) \\\\")
    lines.append("\\midrule")

    # Get baseline results for reduction factors
    baseline_tprs = {}
    if baseline_name and baseline_name in benchmark_results:
        baseline_df = benchmark_results[baseline_name]
        for attack in attacks:
            attack_data = baseline_df[baseline_df['Attack'] == attack]
            if not attack_data.empty:
                tpr_col = f"TPR@{fpr_pct:.4g}%FPR"
                if tpr_col in attack_data.columns:
                    baseline_tprs[attack] = attack_data[tpr_col].iloc[0]

    # Process each benchmark
    for bench_name, results_df in benchmark_results.items():
        n_attacks = len(attacks)
        lines.append(f"\\multirow{{{n_attacks}}}{{*}}{{{bench_name}}} & ")

        for i, attack in enumerate(attacks):
            attack_data = results_df[results_df['Attack'] == attack]

            if attack_data.empty:
                tpr_text = "--"
                prec_text = "--"
                auc_text = "--"
            else:
                tpr_col = f"TPR@{fpr_pct:.4g}%FPR"
                prec_col = f"Prec@{fpr_pct:.4g}%FPR"

                tpr_val = attack_data[tpr_col].iloc[0] if tpr_col in attack_data.columns else np.nan
                prec_val = attack_data[prec_col].iloc[0] if prec_col in attack_data.columns else np.nan
                auc_val = attack_data['AUC'].iloc[0] if 'AUC' in attack_data.columns else np.nan

                tpr_text = f"{tpr_val:.3f}" if not np.isnan(tpr_val) else "--"
                prec_text = f"{prec_val:.2f}" if not np.isnan(prec_val) else "--"
                auc_text = f"{auc_val:.2f}" if not np.isnan(auc_val) else "--"

                # Add reduction factor if not baseline
                if bench_name != baseline_name and attack in baseline_tprs:
                    baseline_tpr = baseline_tprs[attack]
                    if not np.isnan(tpr_val) and tpr_val > 0:
                        reduction = baseline_tpr / tpr_val
                        tpr_text += " " + format_multiplier(reduction)

            if i == 0:
                lines[-1] += f"{attack} & {tpr_text} & {prec_text} & {auc_text} \\\\"
            else:
                lines.append(f"& {attack} & {tpr_text} & {prec_text} & {auc_text} \\\\")

        lines.append("\\midrule")

    # Remove last midrule and add bottomrule
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")

    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def save_latex_table(latex_str: str, output_path: str):
    """
    Save LaTeX table to file.

    Args:
        latex_str: LaTeX table string
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved LaTeX table to {output_path}")
