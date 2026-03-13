"""
Shared utilities for script-based rank-stability analysis.

This module holds the implementation used by `reproducibility_analysis.py`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy.stats import spearmanr
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

try:
    from .analysis_utils import save_top1_samples_image_grid
    from .plot_style import configure_popets_style
except ImportError:
    from analysis_utils import save_top1_samples_image_grid
    from plot_style import configure_popets_style


DEFAULT_CIFAR_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass
class RankStabilityConfig:
    base_dir: Path
    runs: list[str]
    score_file: str = "online_scores_leave_one_out.npy"
    labels_file: str = "membership_labels.npy"
    topq_pcts: list[float] | None = None
    bin_edges: list[int] | None = None
    max_sample_id_excl: int | None = 50000
    out_fig_dir: str = "figures_rank_stability"
    out_tex_dir: str = "latex_rank_stability"
    tag: str = "rank_scores"
    vuln_mode: str = "all"
    base_fontsize: int = 12
    cifar_root: Path = DEFAULT_CIFAR_ROOT
    include_all_runs_metrics: bool = False
    include_inrun_displacement: bool = False
    inrun_deltas: list[float] | None = None
    inrun_q_print: list[float] | None = None
    save_inrun_plot: bool = True

    def __post_init__(self) -> None:
        if self.topq_pcts is None:
            self.topq_pcts = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5] + list(range(10, 101, 10))
        if self.bin_edges is None:
            self.bin_edges = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
        if self.inrun_deltas is None:
            self.inrun_deltas = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        if self.inrun_q_print is None:
            self.inrun_q_print = [0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.0, 20.0]

    @property
    def run_dirs(self) -> dict[str, Path]:
        return {run: self.base_dir / run for run in self.runs}

    @property
    def out_top1_dir(self) -> str:
        return os.path.join(self.out_fig_dir, "top1_across_runs")

    @property
    def out_inrun_dir(self) -> str:
        return os.path.join(self.out_fig_dir, "inside_run_displacement_simple")


def _now() -> float:
    return time.time()


def _fmt_s(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.2f}h"


def _q_count(n: int, q_pct: float) -> int:
    return max(1, int(np.ceil(n * (q_pct / 100.0))))


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return np.nan
    return float(spearmanr(a, b).correlation)


def mean_std(arr: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    return (float(np.mean(arr)), float(np.std(arr)))


def robust_stats_1d(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(mean=np.nan, median=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan, std=np.nan, min=np.nan, max=np.nan, n=0)
    mean = float(np.mean(x))
    std = float(np.std(x))
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    return {
        "mean": mean,
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "std": std,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n": int(x.size),
    }


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def fmt_pm_percent_latex(mean: float, std: float, decimals: int = 2) -> str:
    if not np.isfinite(mean):
        return "--"
    m = 100.0 * float(mean)
    if not np.isfinite(std):
        return f"{m:.{decimals}f}\\%"
    s = 100.0 * float(std)
    return f"{m:.{decimals}f}\\% $\\pm$ {s:.{decimals}f}\\%"


def fmt_pm_int_latex(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    m = int(np.rint(float(mean)))
    if not np.isfinite(std):
        return f"{m:d}"
    s = int(np.rint(float(std)))
    return f"{m:d} $\\pm$ {s:d}"


def fmt_pm_percent(mean: float, std: float, decimals: int = 2) -> str:
    if not np.isfinite(mean):
        return "--"
    m = 100.0 * float(mean)
    if not np.isfinite(std):
        return f"{m:.{decimals}f}%"
    s = 100.0 * float(std)
    return f"{m:.{decimals}f}% +/- {s:.{decimals}f}%"


def fmt_pm_int(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    m = int(np.rint(float(mean)))
    if not np.isfinite(std):
        return f"{m:d}"
    s = int(np.rint(float(std)))
    return f"{m:d} +/- {s:d}"


def load_run_arrays(run_dir: Path, labels_file: str, score_file: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.load(run_dir / labels_file).astype(bool, copy=False)
    scores = np.load(run_dir / score_file).astype(np.float64, copy=False)
    if labels.shape != scores.shape:
        raise RuntimeError(f"Shape mismatch in {run_dir}: labels={labels.shape}, scores={scores.shape}")
    return labels, scores


def compute_vulnerability(labels_mn: np.ndarray, scores_mn: np.ndarray, mode: str) -> np.ndarray:
    labels_mn = np.asarray(labels_mn, dtype=bool)
    scores_mn = np.asarray(scores_mn, dtype=np.float64)
    if labels_mn.shape != scores_mn.shape:
        raise ValueError(f"Shape mismatch: labels {labels_mn.shape}, scores {scores_mn.shape}")

    members = np.where(labels_mn, scores_mn, np.nan)
    nonmembers = np.where(~labels_mn, scores_mn, np.nan)

    if mode == "mean_gap":
        return np.nanmean(members, axis=0) - np.nanmean(nonmembers, axis=0)
    if mode == "median_gap":
        return np.nanmedian(members, axis=0) - np.nanmedian(nonmembers, axis=0)
    if mode != "all":
        raise ValueError(f"Unknown mode: {mode}")

    n_items = labels_mn.shape[1]
    vulnerability = np.empty(n_items, dtype=np.float64)
    for idx in range(n_items):
        lab = labels_mn[:, idx]
        sc = scores_mn[:, idx]
        s_in = sc[lab]
        s_out = sc[~lab]
        if s_in.size == 0 or s_out.size == 0:
            vulnerability[idx] = np.nan
        else:
            vulnerability[idx] = (np.mean(s_in) - np.mean(s_out) + np.median(s_in) - np.median(s_out)) / 2.0
    return vulnerability


def compute_rank_positions(vulnerability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((np.arange(vulnerability.size, dtype=np.int32), -vulnerability))
    positions = np.empty_like(order)
    positions[order] = np.arange(order.size, dtype=order.dtype)
    return order, positions


def global_spearman_pairs(v_by_run: dict[str, np.ndarray]) -> np.ndarray:
    values = []
    for run_a, run_b in combinations(v_by_run.keys(), 2):
        values.append(safe_spearman(v_by_run[run_a], v_by_run[run_b]))
    return np.asarray(values, dtype=float)


def topq_pairwise_exact_from_pos(pos_by_run: dict[str, np.ndarray], q_pcts: list[float], n: int) -> pd.DataFrame:
    rows = []
    runs = list(pos_by_run.keys())
    for q_pct in q_pcts:
        k = _q_count(n, float(q_pct))
        for run_a, run_b in combinations(runs, 2):
            pos_a = pos_by_run[run_a]
            pos_b = pos_by_run[run_b]
            inter = int(np.sum(np.maximum(pos_a, pos_b) < k))
            union = int(2 * k - inter)
            rows.append(
                {
                    "topq_pct": float(q_pct),
                    "intersection": inter,
                    "union": union,
                    "jaccard": inter / max(1, union),
                }
            )
    return pd.DataFrame(rows)


def tail_spearman_pairwise_topq_intersection_only(
    v_by_run: dict[str, np.ndarray],
    pos_by_run: dict[str, np.ndarray],
    q_pcts: list[float],
    n: int,
) -> pd.DataFrame:
    rows = []
    runs = list(v_by_run.keys())
    for q_pct in q_pcts:
        k = _q_count(n, float(q_pct))
        for run_a, run_b in combinations(runs, 2):
            pos_a = pos_by_run[run_a]
            pos_b = pos_by_run[run_b]
            idx = np.where(np.maximum(pos_a, pos_b) < k)[0]
            rho = safe_spearman(v_by_run[run_a][idx], v_by_run[run_b][idx]) if idx.size >= 2 else np.nan
            rows.append({"topq_pct": float(q_pct), "tail_inter_spearman": rho})
    return pd.DataFrame(rows)


def allruns_intersection_union_jaccard_from_pos(
    pos_by_run: dict[str, np.ndarray],
    q_pcts: list[float],
    n: int,
) -> pd.DataFrame:
    pos_stack = np.stack([pos_by_run[run] for run in pos_by_run], axis=0)
    maxpos = np.max(pos_stack, axis=0)
    minpos = np.min(pos_stack, axis=0)
    rows = []
    for q_pct in q_pcts:
        k = _q_count(n, float(q_pct))
        all_intersection = int(np.sum(maxpos < k))
        all_union = int(np.sum(minpos < k))
        rows.append(
            {
                "topq_pct": float(q_pct),
                "all_intersection": all_intersection,
                "all_union": all_union,
                "all_jaccard": (all_intersection / all_union) if all_union > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def allruns_tail_spearman_on_global_intersection(
    v_by_run: dict[str, np.ndarray],
    pos_by_run: dict[str, np.ndarray],
    q_pcts: list[float],
    n: int,
) -> pd.DataFrame:
    runs = list(v_by_run.keys())
    pos_stack = np.stack([pos_by_run[run] for run in runs], axis=0)
    maxpos = np.max(pos_stack, axis=0)
    rows = []
    for q_pct in q_pcts:
        k = _q_count(n, float(q_pct))
        idx = np.where(maxpos < k)[0]
        if idx.size < 2:
            rows.append({"topq_pct": float(q_pct), "all_tail_spearman_mean": np.nan})
            continue
        values = [safe_spearman(v_by_run[run_a][idx], v_by_run[run_b][idx]) for run_a, run_b in combinations(runs, 2)]
        mu, _ = mean_std(np.asarray(values, dtype=float))
        rows.append({"topq_pct": float(q_pct), "all_tail_spearman_mean": mu})
    return pd.DataFrame(rows)


def agg_pairs_for_q(df_q_pairs: pd.DataFrame, df_t_pairs: pd.DataFrame) -> pd.DataFrame:
    agg_q = df_q_pairs.groupby("topq_pct", as_index=False).agg(
        inter_mean=("intersection", "mean"),
        inter_std=("intersection", "std"),
        union_mean=("union", "mean"),
        union_std=("union", "std"),
        jacc_mean=("jaccard", "mean"),
        jacc_std=("jaccard", "std"),
    )
    agg_t = df_t_pairs.groupby("topq_pct", as_index=False).agg(
        tail_inter_mean=("tail_inter_spearman", "mean"),
        tail_inter_std=("tail_inter_spearman", "std"),
    )
    return pd.merge(agg_q, agg_t, on="topq_pct", how="outer").sort_values("topq_pct")


def _pos_to_percentile(pos: np.ndarray, n: int) -> np.ndarray:
    return (100.0 * pos.astype(np.float64)) / float(n)


def inside_run_displacement_simple(
    pos_by_run: dict[str, np.ndarray],
    q_pcts: list[float],
    deltas: list[float],
    runs_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = next(iter(pos_by_run.values())).size
    pos_stack = np.stack([pos_by_run[run] for run in runs_order], axis=0)
    minpos = np.min(pos_stack, axis=0)
    rows = []
    for q_pct in q_pcts:
        q = float(q_pct)
        kq = _q_count(n, q)
        in_union = minpos < kq
        for run in runs_order:
            pos_r = pos_by_run[run]
            displaced = in_union & ~(pos_r < kq)
            idx = np.where(displaced)[0]
            row = {"run": run, "topq_pct": q, "n_displaced": int(idx.size)}
            if idx.size == 0:
                for delta in deltas:
                    row[f"frac_leq_q_plus_{delta:g}"] = np.nan
                rows.append(row)
                continue
            percentiles = _pos_to_percentile(pos_r[idx], n)
            for delta in deltas:
                row[f"frac_leq_q_plus_{delta:g}"] = float(np.mean(percentiles <= (q + float(delta))))
            rows.append(row)

    df_per_run = pd.DataFrame(rows)
    agg_rows = []
    for q_pct in sorted(df_per_run["topq_pct"].unique()):
        subset = df_per_run[df_per_run["topq_pct"] == q_pct]
        base = {
            "topq_pct": float(q_pct),
            "n_displaced_mean": float(np.mean(subset["n_displaced"])),
            "n_displaced_std": float(np.std(subset["n_displaced"])),
        }
        for delta in deltas:
            mu, sd = mean_std(subset[f"frac_leq_q_plus_{delta:g}"].to_numpy(float))
            base[f"frac_leq_q_plus_{delta:g}_mean"] = mu
            base[f"frac_leq_q_plus_{delta:g}_std"] = sd
        agg_rows.append(base)
    return df_per_run, pd.DataFrame(agg_rows).sort_values("topq_pct")


def save_inside_run_displacement_plot(out_pdf: str, df_agg: pd.DataFrame, q_list: list[float], deltas: list[float]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.2))
    for q_pct in q_list:
        row = df_agg[df_agg["topq_pct"] == float(q_pct)]
        if row.empty:
            continue
        ys = [float(row[f"frac_leq_q_plus_{delta:g}_mean"].iloc[0]) for delta in deltas]
        ax.plot(deltas, ys, marker="o", linewidth=1.3, markersize=3.5, label=f"q={q_pct:g}%")
    ax.set_xlabel("Delta (quantile shift)")
    ax.set_ylabel("Fraction <= q + Delta")
    ax.set_xticks(np.arange(0, max(deltas) + 0.5, 1))
    ax.set_xlim(0, max(deltas))
    ax.set_ylim(0, 1.01)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=400)
    plt.close(fig)


def format_agg_for_console(agg: pd.DataFrame, n_total: int) -> pd.DataFrame:
    df = agg.copy()
    df.insert(1, "k", [_q_count(n_total, float(q_pct)) for q_pct in df["topq_pct"].tolist()])
    df["tail_inter"] = [fmt_pm_percent(m, s, decimals=2) for m, s in zip(df["tail_inter_mean"], df["tail_inter_std"])]
    df["cap"] = [fmt_pm_int(m, s) for m, s in zip(df["inter_mean"], df["inter_std"])]
    df["cup"] = [fmt_pm_int(m, s) for m, s in zip(df["union_mean"], df["union_std"])]
    df["jaccard"] = [fmt_pm_percent(m, s, decimals=2) for m, s in zip(df["jacc_mean"], df["jacc_std"])]
    out = df[["topq_pct", "k", "tail_inter", "cap", "cup", "jaccard"]].copy()
    out = out.rename(columns={"topq_pct": "Top-q%", "tail_inter": "Tail rho (|cap|)", "cap": "|cap|", "cup": "|cup|", "jaccard": "Jaccard"})
    if "all_intersection" in df.columns and "all_union" in df.columns:
        out["All-runs |int|/|uni|"] = [
            f"{int(a)}/{int(u)}" if np.isfinite(a) and np.isfinite(u) else "--"
            for a, u in zip(df["all_intersection"], df["all_union"])
        ]
    if "all_jaccard" in df.columns:
        out["All-runs Jaccard"] = [f"{100.0 * float(j):.2f}%" if np.isfinite(j) else "--" for j in df["all_jaccard"].to_numpy(float)]
    if "all_tail_spearman_mean" in df.columns:
        out["All-runs Tail rho"] = [f"{100.0 * float(rho):.2f}%" if np.isfinite(rho) else "--" for rho in df["all_tail_spearman_mean"].to_numpy(float)]
    return out


def save_latex_table_topq(path_tex: str, agg: pd.DataFrame, topq_list: list[float], n_total: int, caption: str, label: str) -> None:
    has_all_runs = {"all_intersection", "all_union", "all_jaccard", "all_tail_spearman_mean"}.issubset(set(agg.columns))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        "\\begin{tabular}{rrccccccc}" if has_all_runs else "\\begin{tabular}{rrcccc}",
        "\\toprule",
    ]
    if has_all_runs:
        lines.append(
            "Top-$q\\%$ & $k$ & Tail $\\rho$ ($|\\cap|$) & $|\\cap|$ & $|\\cup|$ & Jaccard & "
            "All-runs $|\\cap|/|\\cup|$ & All-runs Jac. & All-runs Tail $\\rho$\\\\"
        )
    else:
        lines.append("Top-$q\\%$ & $k$ & Tail $\\rho$ ($|\\cap|$) & $|\\cap|$ & $|\\cup|$ & Jaccard\\\\")
    lines.append("\\midrule")

    by_q = agg.set_index("topq_pct")
    for q_pct in topq_list:
        row = by_q.loc[float(q_pct)]
        k = _q_count(n_total, float(q_pct))
        ti = fmt_pm_percent_latex(float(row["tail_inter_mean"]), float(row["tail_inter_std"]), decimals=2)
        jac = fmt_pm_percent_latex(float(row["jacc_mean"]), float(row["jacc_std"]), decimals=2)
        inter = fmt_pm_int_latex(float(row["inter_mean"]), float(row["inter_std"]))
        uni = fmt_pm_int_latex(float(row["union_mean"]), float(row["union_std"]))
        if has_all_runs:
            all_cell = (
                f"{int(row['all_intersection'])}/{int(row['all_union'])}"
                if np.isfinite(row.get("all_intersection", np.nan)) and np.isfinite(row.get("all_union", np.nan))
                else "--"
            )
            all_jac = f"{100.0 * float(row['all_jaccard']):.2f}\\%" if np.isfinite(row.get("all_jaccard", np.nan)) else "--"
            all_tail = f"{100.0 * float(row['all_tail_spearman_mean']):.2f}\\%" if np.isfinite(row.get("all_tail_spearman_mean", np.nan)) else "--"
            lines.append(f"{q_pct:g} & {k:d} & {ti} & {inter} & {uni} & {jac} & {all_cell} & {all_jac} & {all_tail}\\\\")
        else:
            lines.append(f"{q_pct:g} & {k:d} & {ti} & {inter} & {uni} & {jac}\\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{{latex_escape(label)}}}",
            "\\end{table}",
        ]
    )
    with open(path_tex, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _pct_to_start_end(n: int, lo_pct: float, hi_pct: float) -> tuple[int, int]:
    start = 0 if lo_pct == 1 else int(np.floor(n * (lo_pct / 100.0)))
    end = int(np.ceil(n * (hi_pct / 100.0)))
    return start, max(start + 1, end)


def collect_scores_by_percentile_bin(vulnerability: np.ndarray, bin_edges: list[int]) -> pd.DataFrame:
    n = vulnerability.size
    order = np.lexsort((np.arange(n, dtype=np.int32), -vulnerability))
    rows = []
    for lo_pct, hi_pct in zip(bin_edges[:-1], bin_edges[1:]):
        start, end = _pct_to_start_end(n, lo_pct, hi_pct)
        idx = order[start:end]
        label = f"{lo_pct}-{hi_pct}%"
        for value in vulnerability[idx]:
            rows.append({"bin": label, "value": float(value)})
    return pd.DataFrame(rows)


def save_bin_boxplot(path_pdf: str, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    sns.boxplot(data=df, x="bin", y="value", ax=ax, color="#d9d9d9", linewidth=0.8, fliersize=1.6)
    ax.set_xlabel("Rank-percentile bin")
    ax.set_ylabel("Vulnerability score")
    ax.tick_params(axis="x", which="both", direction="out", length=3, width=0.6, labelbottom=True)
    ax.tick_params(axis="y", which="both", direction="out", length=3, width=0.6)
    offset = mtransforms.ScaledTranslation(10 / 72.0, 0, fig.dpi_scale_trans)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_horizontalalignment("right")
        tick.set_transform(tick.get_transform() + offset)
    fig.tight_layout()
    fig.savefig(path_pdf, bbox_inches="tight", dpi=400)
    plt.close(fig)


def _top1_indices_per_run(v_by_run: dict[str, np.ndarray], runs_order: list[str]) -> list[dict[str, int]]:
    rows = []
    for run in runs_order:
        order = np.lexsort((np.arange(v_by_run[run].size, dtype=np.int32), -v_by_run[run]))
        rows.append({"run": run, "sample_id": int(order[0])})
    return rows


def _build_top1_across_runs_df(v_by_run: dict[str, np.ndarray], top1_list: list[dict[str, int]], runs_order: list[str]) -> pd.DataFrame:
    rows = []
    for item in top1_list:
        top1_run = item["run"]
        sample_id = item["sample_id"]
        scores = [float(v_by_run[run][sample_id]) for run in runs_order]
        stats = robust_stats_1d(np.asarray(scores, dtype=float))
        for run, value in zip(runs_order, scores):
            rows.append({"top1_run": top1_run, "sample_id": sample_id, "eval_run": run, "value": value, **stats})
    return pd.DataFrame(rows)


def save_top1_across_runs_boxplots(out_pdf: str, df: pd.DataFrame, runs_order: list[str]) -> None:
    top1_runs = list(dict.fromkeys(df["top1_run"].tolist()))
    n_plots = len(top1_runs)
    ncols = 4
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 2.4 * nrows), squeeze=False)
    for ax in axes.flat[n_plots:]:
        ax.axis("off")
    for ax, top1_run in zip(axes.flat, top1_runs):
        sub = df[df["top1_run"] == top1_run]
        sample_id = int(sub["sample_id"].iloc[0])
        values = sub["value"].to_numpy(float)
        ax.boxplot(values, widths=0.4, showfliers=True)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_xticks([])
        ax.set_ylabel("")
        ax.set_title(f"{top1_run.replace('seed', 'Seed ')}: Sample {sample_id}", fontsize=9.5)
        text = f"Mean {sub['mean'].iloc[0]:.3f}\nMedian {sub['median'].iloc[0]:.3f}\nIQR {sub['iqr'].iloc[0]:.3f}"
        ax.text(
            0.02,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=7.0,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.85, linewidth=0.5),
        )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", dpi=400)
    plt.close(fig)


def run_one_mode(
    mode: str,
    labels_scores_by_run: dict[str, tuple[np.ndarray, np.ndarray]],
    viz_dataset: CIFAR10,
    config: RankStabilityConfig,
) -> None:
    t0 = _now()
    print(f"\n==================== Mode: {mode} ====================")

    v_by_run = {run: compute_vulnerability(labels, scores, mode=mode) for run, (labels, scores) in labels_scores_by_run.items()}
    n_total = next(iter(v_by_run.values())).size
    rhos = global_spearman_pairs(v_by_run)
    mu100, sd100 = mean_std(rhos)
    print(f"[{mode}] Global Spearman (all samples): {mu100:.4f} +/- {sd100:.4f} over {rhos.size} run pairs")

    top1_list = _top1_indices_per_run(v_by_run, config.runs)
    df_top1 = _build_top1_across_runs_df(v_by_run, top1_list, config.runs)
    grid_pdf = os.path.join(config.out_top1_dir, f"top1_samples_grid_{mode}_{config.tag}.pdf")
    save_top1_samples_image_grid(out_path_pdf=grid_pdf, top1_list=top1_list, dataset=viz_dataset, nrows=3, ncols=4, dpi=400)
    csv_top1 = os.path.join(config.out_top1_dir, f"top1_samples_across_runs_{mode}_{config.tag}.csv")
    df_top1.to_csv(csv_top1, index=False)
    pdf_top1 = os.path.join(config.out_top1_dir, f"top1_across_runs_boxplots_{mode}_{config.tag}.pdf")
    save_top1_across_runs_boxplots(out_pdf=pdf_top1, df=df_top1, runs_order=config.runs)
    mapping = ", ".join(f"{item['run']}->{item['sample_id']}" for item in top1_list)
    print(f"[{mode}] Saved top-1 samples image grid (PDF): {grid_pdf}")
    print(f"[{mode}] Top-1 per run (run->sample_id): {mapping}")
    print(f"[{mode}] Saved top-1 across-run CSV: {csv_top1}")
    print(f"[{mode}] Saved top-1 across-run figure: {pdf_top1}")

    pos_by_run = {run: compute_rank_positions(values)[1] for run, values in v_by_run.items()}
    if config.include_inrun_displacement:
        df_inrun_per, df_inrun_agg = inside_run_displacement_simple(pos_by_run, config.topq_pcts, config.inrun_deltas, config.runs)
        csv_per = os.path.join(config.out_inrun_dir, f"inside_run_displacement_per_run_{mode}_{config.tag}.csv")
        csv_agg = os.path.join(config.out_inrun_dir, f"inside_run_displacement_agg_over_runs_{mode}_{config.tag}.csv")
        df_inrun_per.to_csv(csv_per, index=False)
        df_inrun_agg.to_csv(csv_agg, index=False)
        print(f"[{mode}] Inside-run displacement per-run CSV: {csv_per}")
        print(f"[{mode}] Inside-run displacement aggregated CSV: {csv_agg}")
        if config.save_inrun_plot:
            out_pdf = os.path.join(config.out_inrun_dir, f"inside_run_displacement_curve_{mode}_{config.tag}.pdf")
            save_inside_run_displacement_plot(out_pdf, df_inrun_agg, config.inrun_q_print, config.inrun_deltas)
            print(f"[{mode}] Inside-run displacement curve: {out_pdf}")

    df_q = topq_pairwise_exact_from_pos(pos_by_run, config.topq_pcts, n_total)
    df_t = tail_spearman_pairwise_topq_intersection_only(v_by_run, pos_by_run, config.topq_pcts, n_total)
    agg = agg_pairs_for_q(df_q, df_t)
    if config.include_all_runs_metrics:
        agg = pd.merge(agg, allruns_intersection_union_jaccard_from_pos(pos_by_run, config.topq_pcts, n_total), on="topq_pct", how="left")
        agg = pd.merge(agg, allruns_tail_spearman_on_global_intersection(v_by_run, pos_by_run, config.topq_pcts, n_total), on="topq_pct", how="left")

    print(f"\n[{mode}] Top-q% agreement summary:")
    print(format_agg_for_console(agg, n_total=n_total).to_string(index=False))

    tex_path = os.path.join(config.out_tex_dir, f"topq_tail_table_{mode}_{config.tag}.tex")
    if config.include_all_runs_metrics:
        caption = (
            f"Top-q% stability and agreement for vulnerability score {mode}. Pairwise columns are mean +/- std over "
            f"all C({len(config.runs)},2) run pairs. Pairwise Tail rho is Spearman on the intersection. "
            f"All-runs columns summarize the strict multi-run core. (Global Spearman: {mu100:.3f} +/- {sd100:.3f}.)"
        )
    else:
        caption = (
            f"Top-q% stability and agreement for vulnerability score {mode}. Tail rho is Spearman on the pairwise "
            f"intersection. Values are mean +/- std over all C({len(config.runs)},2) run pairs. "
            f"(Global Spearman: {mu100:.3f} +/- {sd100:.3f}.)"
        )
    save_latex_table_topq(tex_path, agg, config.topq_pcts, n_total, caption=caption, label=f"tab:topq_tail_{mode}")
    print(f"[{mode}] LaTeX (Top-q%) table: {tex_path}")

    pooled = []
    for run, values in v_by_run.items():
        df_run = collect_scores_by_percentile_bin(values, config.bin_edges)
        df_run["run"] = run
        pooled.append(df_run)
    df_pooled = pd.concat(pooled, ignore_index=True)
    bin_labels = [f"{lo}-{hi}%" for lo, hi in zip(config.bin_edges[:-1], config.bin_edges[1:])]
    df_pooled["bin"] = pd.Categorical(df_pooled["bin"], categories=bin_labels, ordered=True)
    fig_bins = os.path.join(config.out_fig_dir, f"bin_boxplot_scores_{mode}_{config.tag}.pdf")
    save_bin_boxplot(fig_bins, df_pooled)
    print(f"[{mode}] Figure (bin boxplot, no annotations): {fig_bins}")
    print(f"[{mode}] Done. Total: {_fmt_s(_now() - t0)}")


def run_rank_stability_analysis(config: RankStabilityConfig) -> None:
    os.makedirs(config.out_fig_dir, exist_ok=True)
    os.makedirs(config.out_tex_dir, exist_ok=True)
    os.makedirs(config.out_top1_dir, exist_ok=True)
    if config.include_inrun_displacement:
        os.makedirs(config.out_inrun_dir, exist_ok=True)

    configure_popets_style(base_fontsize=config.base_fontsize, family="serif")
    sns.set_theme(context="paper", style="whitegrid")

    t_all = _now()
    print("Loading runs...")
    labels_scores_by_run: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_used = None
    for run, run_dir in config.run_dirs.items():
        t_run = _now()
        labels, scores = load_run_arrays(run_dir, labels_file=config.labels_file, score_file=config.score_file)
        if config.max_sample_id_excl is not None:
            keep = np.arange(labels.shape[1]) < int(config.max_sample_id_excl)
            labels = labels[:, keep]
            scores = scores[:, keep]
        if n_used is None:
            n_used = labels.shape[1]
        elif labels.shape[1] != n_used:
            raise RuntimeError(f"Inconsistent N after filtering: {run} has {labels.shape[1]} vs {n_used}")
        labels_scores_by_run[run] = (labels, scores)
        print(f"  - {run}: (M,N)={labels.shape}  ({_fmt_s(_now() - t_run)})")

    print(f"\nAll runs loaded. N_used={n_used}. Total load: {_fmt_s(_now() - t_all)}")
    viz_dataset = CIFAR10(root=str(config.cifar_root), train=True, download=False, transform=ToTensor())
    modes = ["mean_gap", "median_gap"] if config.vuln_mode == "all" else [config.vuln_mode]
    for mode in modes:
        run_one_mode(mode, labels_scores_by_run, viz_dataset, config)

    print("\n" + "=" * 80)
    print(f"Rank-stability analysis complete! Total wall time: {_fmt_s(_now() - t_all)}")
    print("=" * 80)
