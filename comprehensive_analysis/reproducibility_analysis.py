"""Unified reproducibility analysis driver.

This script provides one script-based entry point for:

- cross-run reproducibility/stability/coverage panels
- TP>=x heatmaps
- TP>=x @0 FP heatmaps
- rank-stability outputs via ``rank_stability_utils.py``

The default paths are aligned with the checked-in CIFAR-10 artifact snapshot.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator, PercentFormatter

try:
    from .analysis_utils import (
        SAMPLE_ID_COL,
        _nice_bounds_numeric,
        aggregate_over_configs,
        compute_dynamic_ylims,
        compute_scenario,
        to_id_set,
    )
    from .plot_style import configure_popets_style
    from .rank_stability_utils import RankStabilityConfig, run_rank_stability_analysis
except ImportError:
    from analysis_utils import (
        SAMPLE_ID_COL,
        _nice_bounds_numeric,
        aggregate_over_configs,
        compute_dynamic_ylims,
        compute_scenario,
        to_id_set,
    )
    from plot_style import configure_popets_style
    from rank_stability_utils import RankStabilityConfig, run_rank_stability_analysis


_REQUIRED_CSV = "summary_statistics_two_modes.csv"
_SEED_RE = re.compile(r"^seed\d+$")

DEFAULT_FPRS = [1e-5, 1e-3]
DEFAULT_TP_THRESHOLDS = [1, 2, 3, 4, 5, 10, 20, 64]
DEFAULT_TOPQ_PCTS = [0.1, 0.25, 0.5, 0.75, 1, 2, 2.24, 3, 4, 5, 10.38] + list(range(10, 101, 10))
DEFAULT_BIN_EDGES = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
DEFAULT_INRUN_DELTAS = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
DEFAULT_INRUN_Q_PRINT = [0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.0, 20.0]
BASE_FONTSIZE = 12
LINE_WIDTH = 2.5
HEATMAP_CMAP = "viridis"


@dataclass
class ReproducibilityConfig:
    script_dir: Path
    repo_root: Path
    analysis_inputs: dict[str, Path]
    rank_base_dir: Path
    figures_dir: Path
    tables_dir: Path
    rank_figures_dir: Path
    rank_latex_dir: Path
    fprs: list[float]
    tp_thresholds: list[int]
    max_sample_id_excl: int = 50000
    skip_threshold_panels: bool = False
    skip_heatmaps: bool = False
    skip_rank: bool = False
    baseline_tokens: list[str] = field(default_factory=list)
    variant_tokens: list[str] = field(default_factory=list)


def fpr_to_tag(fpr: float) -> str:
    pct = fpr * 100.0
    value = f"{pct:.3f}".rstrip("0").rstrip(".")
    return value.replace(".", "p") + "pct"


def build_cli_config() -> ReproducibilityConfig:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Unified reproducibility analysis driver.")
    parser.add_argument("--fprs", nargs="+", type=float, default=DEFAULT_FPRS, help="FPR operating points to process.")
    parser.add_argument(
        "--tp-thresholds",
        nargs="+",
        type=int,
        default=DEFAULT_TP_THRESHOLDS,
        help="TP support thresholds used in heatmaps.",
    )
    parser.add_argument(
        "--max-sample-id-excl",
        type=int,
        default=50000,
        help="Keep sample ids < this value for rank-stability analysis.",
    )
    _analysis_root_default = repo_root / "analysis_results" / "cifar10" / "resnet18"
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=_analysis_root_default,
        help="Root for baseline CIFAR-10 analysis_results runs.",
    )
    parser.add_argument(
        "--arch-analysis-root",
        type=Path,
        default=repo_root / "analysis_results" / "cifar10" / "wrn28-2" / "seed42",
        help="Analysis-results directory for the architecture-change run.",
    )
    parser.add_argument(
        "--rank-base-dir",
        type=Path,
        default=repo_root / "experiments" / "cifar10" / "resnet18",
        help="Root experiment directory used for rank-stability analysis.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "analysis_results" / "figures",
        help="Directory for threshold-based reproducibility figures.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=repo_root / "analysis_results" / "tables",
        help="Directory for threshold-based reproducibility tables.",
    )
    parser.add_argument(
        "--rank-figures-dir",
        type=Path,
        default=repo_root / "analysis_results" / "figures",
        help="Directory for rank-stability figures (default: analysis_results/figures).",
    )
    parser.add_argument(
        "--rank-latex-dir",
        type=Path,
        default=repo_root / "analysis_results" / "tables",
        help="Directory for rank-stability LaTeX tables (default: analysis_results/tables).",
    )
    parser.add_argument(
        "--skip-threshold-panels",
        action="store_true",
        help="Skip the line-panel reproducibility/stability/coverage figures.",
    )
    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip TP-support and TP-support@0FP heatmaps.",
    )
    parser.add_argument(
        "--skip-rank",
        action="store_true",
        help="Skip rank-stability analysis.",
    )
    args = parser.parse_args()

    analysis_inputs: dict[str, Path] = {}
    if args.analysis_root.exists():
        for p in sorted(args.analysis_root.iterdir()):
            if p.is_dir() and (p / _REQUIRED_CSV).exists():
                analysis_inputs[p.name] = p
    if args.arch_analysis_root and args.arch_analysis_root.exists():
        analysis_inputs.setdefault("arch", args.arch_analysis_root)

    baseline_tokens = sorted(k for k in analysis_inputs if _SEED_RE.match(k))
    variant_tokens = sorted(k for k in analysis_inputs if not _SEED_RE.match(k))

    if not analysis_inputs:
        print(f"Warning: no completed runs found under {args.analysis_root}")

    return ReproducibilityConfig(
        script_dir=script_dir,
        repo_root=repo_root,
        analysis_inputs=analysis_inputs,
        rank_base_dir=args.rank_base_dir,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        rank_figures_dir=args.rank_figures_dir,
        rank_latex_dir=args.rank_latex_dir,
        fprs=args.fprs,
        tp_thresholds=args.tp_thresholds,
        max_sample_id_excl=args.max_sample_id_excl,
        skip_threshold_panels=args.skip_threshold_panels,
        skip_heatmaps=args.skip_heatmaps,
        skip_rank=args.skip_rank,
        baseline_tokens=baseline_tokens,
        variant_tokens=variant_tokens,
    )


def ensure_required_inputs(inputs: dict[str, Path], required: list[str]) -> None:
    missing = [token for token in required if token not in inputs]
    if missing:
        raise KeyError(f"Missing configured inputs: {missing}")


def draw_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_key: str,
    y_label: str,
    percent_axis: bool,
    y_limits: dict[str, tuple[float, float]],
    order: list[str],
    colors: dict[str, str],
    styles: dict[str, object],
    show_values_legend: bool = False,
    show_xlabel: bool = True,
) -> None:
    for scenario in order:
        sub = df[df["scenario"] == scenario].sort_values("k")
        if sub.empty:
            continue
        ax.plot(
            sub["k"],
            sub[y_key],
            color=colors[scenario],
            linestyle=styles[scenario],
            lw=LINE_WIDTH,
            label=scenario,
        )

    if show_xlabel:
        ax.set_xlabel(r"$k$")
        ax.tick_params(axis="x", which="both", direction="out", length=3, width=0.6, labelbottom=True)
    else:
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    ax.set_ylabel(y_label)
    ax.set_ylim(*y_limits[y_key])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="y", which="both", direction="out", length=3, width=0.6)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.minorticks_on()
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.4, alpha=0.2)

    if percent_axis:
        ax.set_yticks(np.arange(0.0, y_limits[y_key][1] + 0.01, 0.1))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.axhline(0.10, color="grey", lw=0.8, ls=":", alpha=0.45)
        ax.axhline(0.05, color="grey", lw=0.8, ls=":", alpha=0.45)

    if show_values_legend:
        from matplotlib.lines import Line2D

        legend_elements = []
        for scenario in order:
            sub = df[df["scenario"] == scenario].sort_values("k")
            if sub.empty:
                continue
            y_end = sub[y_key].iloc[-1]
            final_value = f"{y_end * 100:.1f}%" if percent_axis else f"{y_end:.0f}"
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[scenario],
                    linestyle=styles[scenario],
                    linewidth=LINE_WIDTH * 0.9,
                    label=final_value,
                )
            )

        ax.legend(
            handles=legend_elements,
            loc="best",
            fontsize=BASE_FONTSIZE - 1,
            frameon=True,
            framealpha=0.95,
            edgecolor="gray",
            ncol=2,
            handlelength=1.3,
            handletextpad=0.35,
            columnspacing=0.8,
            borderpad=0.35,
            labelspacing=0.25,
            title="Final values",
            title_fontsize=BASE_FONTSIZE - 1,
        )


def save_panel(
    path_pdf: Path,
    all_results: pd.DataFrame,
    y_key: str,
    y_label: str,
    percent_axis: bool,
    y_limits: dict[str, tuple[float, float]],
    order: list[str],
    colors: dict[str, str],
    styles: dict[str, object],
    show_xlabel: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    draw_lines(
        ax,
        all_results,
        y_key,
        y_label,
        percent_axis,
        y_limits,
        order,
        colors,
        styles,
        show_values_legend=True,
        show_xlabel=show_xlabel,
    )
    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight", dpi=400)
    plt.close(fig)


def save_legend(
    path_pdf: Path,
    all_results: pd.DataFrame,
    y_limits: dict[str, tuple[float, float]],
    order: list[str],
    colors: dict[str, str],
    styles: dict[str, object],
) -> None:
    fig_tmp, ax_tmp = plt.subplots()
    draw_lines(
        ax_tmp,
        all_results,
        "avg_jaccard",
        "Avg. Jaccard",
        True,
        y_limits,
        order,
        colors,
        styles,
        show_values_legend=False,
        show_xlabel=False,
    )
    handles, labels = ax_tmp.get_legend_handles_labels()
    plt.close(fig_tmp)

    fig_leg = plt.figure(figsize=(5, 0.55))
    fig_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=4,
        frameon=True,
        fontsize=BASE_FONTSIZE - 0.5,
        columnspacing=1.0,
        handletextpad=0.6,
        borderpad=0.6,
    )
    fig_leg.savefig(path_pdf, bbox_inches="tight", pad_inches=0.02, dpi=400)
    plt.close(fig_leg)


def _format_percent_cell(value: float) -> str:
    return "" if pd.isna(value) else f"{value * 100:.1f}"


def _format_int_cell(value: float) -> str:
    return "" if pd.isna(value) else f"{int(round(value))}"


def _formatted_matrix(df: pd.DataFrame, formatter) -> pd.DataFrame:
    return df.apply(lambda col: col.map(formatter))


def compute_identical_over_thresholds(
    inputs: dict[str, Path],
    baseline_tokens: list[str],
    tp_values: list[int],
    csv_basename: str,
    fp_equals: int | None,
) -> pd.DataFrame:
    baseline_names = [(token, inputs[token]) for token in baseline_tokens]
    max_runs = len(baseline_names)
    label = f"Identical (2-{max_runs} runs)"

    rows = []
    for tp_threshold in tp_values:
        sets_baseline = [
            (
                name,
                to_id_set(
                    path,
                    csv_basename=csv_basename,
                    id_col=SAMPLE_ID_COL,
                    tp_threshold=tp_threshold,
                    fp_equals=fp_equals,
                ),
            )
            for name, path in baseline_names
        ]
        df = compute_scenario(label, sets_baseline, kmin=2, kmax=max_runs)
        df["tp_threshold"] = tp_threshold
        rows.append(df)

    return pd.concat(rows, ignore_index=True)


def _pivot_threshold_matrix(df: pd.DataFrame, value: str, tp_thresholds: list[int]) -> pd.DataFrame:
    matrix = df.pivot(index="k", columns="tp_threshold", values=value).sort_index()
    return matrix.reindex(columns=tp_thresholds)


def save_threshold_heatmap(
    ident_only: pd.DataFrame,
    out_heatmap: Path,
    out_table: Path,
    tp_thresholds: list[int],
    x_label: str,
) -> None:
    ident_only[["tp_threshold", "k", "avg_jaccard", "avg_intersection", "avg_union"]].to_csv(
        out_table,
        index=False,
        float_format="%.6f",
    )

    p_j = _pivot_threshold_matrix(ident_only, "avg_jaccard", tp_thresholds)
    p_i = _pivot_threshold_matrix(ident_only, "avg_intersection", tp_thresholds)
    p_u = _pivot_threshold_matrix(ident_only, "avg_union", tp_thresholds)

    lims_j = (0, 1)
    lims_i = _nice_bounds_numeric(np.nanmin(p_i.values), np.nanmax(p_i.values))
    lims_u = _nice_bounds_numeric(np.nanmin(p_u.values), np.nanmax(p_u.values))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    plt.subplots_adjust(wspace=0.08, left=0.07, right=0.99, top=0.86, bottom=0.26)

    sns.heatmap(
        p_j,
        ax=axes[0],
        cmap=HEATMAP_CMAP,
        vmin=lims_j[0],
        vmax=lims_j[1],
        annot=_formatted_matrix(p_j, _format_percent_cell),
        fmt="",
        annot_kws={"fontsize": BASE_FONTSIZE - 1},
        cbar=False,
        linewidths=0.4,
        linecolor="white",
    )
    axes[0].set_title("Avg. Jaccard (%)", fontsize=BASE_FONTSIZE)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Runs combined ($k$)")
    axes[0].tick_params(axis="both", labelsize=BASE_FONTSIZE - 1)

    sns.heatmap(
        p_i,
        ax=axes[1],
        cmap=HEATMAP_CMAP,
        vmin=lims_i[0],
        vmax=lims_i[1],
        annot=_formatted_matrix(p_i, _format_int_cell),
        fmt="",
        annot_kws={"fontsize": BASE_FONTSIZE - 1},
        cbar=False,
        linewidths=0.4,
        linecolor="white",
        yticklabels=False,
    )
    axes[1].set_title("Avg. Intersection", fontsize=BASE_FONTSIZE)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="both", labelsize=BASE_FONTSIZE - 1)

    heatmap = sns.heatmap(
        p_u,
        ax=axes[2],
        cmap=HEATMAP_CMAP,
        vmin=lims_u[0],
        vmax=lims_u[1],
        annot=_formatted_matrix(p_u, _format_int_cell),
        fmt="",
        annot_kws={"fontsize": BASE_FONTSIZE - 1},
        cbar=True,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"orientation": "vertical", "fraction": 0.05, "pad": 0.03},
        yticklabels=False,
    )
    axes[2].set_title("Avg. Union", fontsize=BASE_FONTSIZE)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    axes[2].tick_params(axis="both", labelsize=BASE_FONTSIZE - 1)

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label("")
    colorbar.set_ticks([])
    colorbar.outline.set_linewidth(0.5)

    xticklabels = [str(value) for value in tp_thresholds]
    for ax in axes:
        ax.set_xticklabels(xticklabels, rotation=0)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)

    fig.savefig(out_heatmap, bbox_inches="tight", dpi=400)
    plt.close(fig)


_VARIANT_COLORS = ["#D55E00", "#118A69", "#7A2655", "#7E771F", "#E69F00", "#56B4E9"]
_VARIANT_STYLES = [(0, (5, 2)), (0, (3, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (2, 1, 1, 1))]
_COMBO_COLORS = {2: "#000000", 3: "#B11515", 4: "#999999"}
_COMBO_STYLES = {2: (0, (5, 1, 1, 1)), 3: (0, (7, 2, 1, 2)), 4: (0, (1, 2))}


def generate_reproducibility_panels(config: ReproducibilityConfig) -> None:
    if not config.baseline_tokens:
        print("No baseline runs found; skipping panels.")
        return
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)

    for fpr in config.fprs:
        tag = fpr_to_tag(fpr)
        csv_basename = f"samples_vulnerability_ranked_online_shadow_{tag}.csv"

        sets_baseline = [
            (token, to_id_set(config.analysis_inputs[token], csv_basename=csv_basename, id_col=SAMPLE_ID_COL))
            for token in config.baseline_tokens
        ]
        max_runs = len(sets_baseline)
        label_identical = f"Identical (2-{max_runs} runs)"
        scen_identical = compute_scenario(label_identical, sets_baseline, kmin=2, kmax=max_runs)

        variant_sets = [
            (key, to_id_set(config.analysis_inputs[key], csv_basename=csv_basename, id_col=SAMPLE_ID_COL))
            for key in config.variant_tokens
            if key in config.analysis_inputs
        ]

        variant_scenarios = [
            compute_scenario(f"+1 different ({key})", sets_baseline + [(key, vs)])
            for key, vs in variant_sets
        ]

        def configs_for_m(size: int) -> list:
            return [sets_baseline + list(combo) for combo in combinations(variant_sets, size)]

        combo_scenarios = {
            m: aggregate_over_configs(f"+{m} different", configs_for_m(m))
            for m in range(2, min(len(variant_sets) + 1, 5))
        }

        all_results = pd.concat(
            [scen_identical] + variant_scenarios + list(combo_scenarios.values()),
            ignore_index=True,
        )

        n_base = len(config.baseline_tokens)
        n_var = len(variant_sets)
        table_path = config.tables_dir / f"reproducibility_{n_base}runs_{n_var}variants_{tag}.csv"
        all_results.to_csv(table_path, index=False, float_format="%.4f")
        y_limits = compute_dynamic_ylims(all_results)

        colors = {label_identical: "#0072B2"}
        styles = {label_identical: "solid"}
        order = [label_identical]
        for i, (key, _) in enumerate(variant_sets):
            label = f"+1 different ({key})"
            colors[label] = _VARIANT_COLORS[i % len(_VARIANT_COLORS)]
            styles[label] = _VARIANT_STYLES[i % len(_VARIANT_STYLES)]
            order.append(label)
        for m, scen in combo_scenarios.items():
            label = f"+{m} different"
            colors[label] = _COMBO_COLORS.get(m, "#888888")
            styles[label] = _COMBO_STYLES.get(m, "dotted")
            order.append(label)

        save_panel(config.figures_dir / f"jaccard_noleg_{tag}.pdf", all_results, "avg_jaccard", "Avg. Jaccard", True, y_limits, order, colors, styles, True)
        save_panel(config.figures_dir / f"intersection_{tag}.pdf", all_results, "avg_intersection", "Avg. intersection", False, y_limits, order, colors, styles, True)
        save_panel(config.figures_dir / f"union_{tag}.pdf", all_results, "avg_union", "Avg. union", False, y_limits, order, colors, styles, True)
        save_legend(config.figures_dir / f"legend_{tag}.pdf", all_results, y_limits, order, colors, styles)

        print(f"[{tag}] panels written to {config.figures_dir}")
        print(f"[{tag}] table written to {table_path}")


def generate_heatmaps(config: ReproducibilityConfig) -> None:
    if not config.baseline_tokens:
        print("No baseline runs found; skipping heatmaps.")
        return
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    n = len(config.baseline_tokens)

    for fpr in config.fprs:
        tag = fpr_to_tag(fpr)
        csv_basename = f"samples_vulnerability_ranked_online_shadow_{tag}.csv"

        ident_all_fp = compute_identical_over_thresholds(config.analysis_inputs, config.baseline_tokens, config.tp_thresholds, csv_basename, fp_equals=None)
        save_threshold_heatmap(
            ident_all_fp,
            config.figures_dir / f"tpgeq_x_identical_heatmaps_{tag}.pdf",
            config.tables_dir / f"tpgeq_x_identical_{tag}_{n}runs.csv",
            config.tp_thresholds,
            "TP >= $x$",
        )

        ident_zero_fp = compute_identical_over_thresholds(config.analysis_inputs, config.baseline_tokens, config.tp_thresholds, csv_basename, fp_equals=0)
        save_threshold_heatmap(
            ident_zero_fp,
            config.figures_dir / f"tpgeq_x_0fp_identical_heatmaps_{tag}.pdf",
            config.tables_dir / f"tpgeq_x_0fp_identical_{tag}_{n}runs.csv",
            config.tp_thresholds,
            "TP >= $x$ @0 FP",
        )

        print(f"[{tag}] heatmaps written to {config.figures_dir}")


def run_rank_section(config: ReproducibilityConfig) -> None:
    rank_config = RankStabilityConfig(
        base_dir=config.rank_base_dir,
        runs=config.baseline_tokens.copy(),
        topq_pcts=DEFAULT_TOPQ_PCTS.copy(),
        bin_edges=DEFAULT_BIN_EDGES.copy(),
        max_sample_id_excl=config.max_sample_id_excl,
        out_fig_dir=str(config.rank_figures_dir),
        out_tex_dir=str(config.rank_latex_dir),
        tag="rank_scores",
        vuln_mode="all",
        cifar_root=config.repo_root / "data",
        include_all_runs_metrics=True,
        include_inrun_displacement=True,
        inrun_deltas=DEFAULT_INRUN_DELTAS.copy(),
        inrun_q_print=DEFAULT_INRUN_Q_PRINT.copy(),
        save_inrun_plot=True,
    )
    run_rank_stability_analysis(rank_config)


def main() -> None:
    config = build_cli_config()
    configure_popets_style(base_fontsize=BASE_FONTSIZE, family="serif")
    sns.set_theme(context="paper", style="whitegrid")

    if not config.skip_threshold_panels:
        print("Generating reproducibility/stability/coverage panels...")
        generate_reproducibility_panels(config)

    if not config.skip_heatmaps:
        print("Generating TP-support heatmaps...")
        generate_heatmaps(config)

    if not config.skip_rank:
        print("Generating rank-stability outputs...")
        run_rank_section(config)

    print("Unified reproducibility analysis complete.")


if __name__ == "__main__":
    main()
