"""Per-run LiRA post analysis.

This script writes the analysis outputs under
``comprehensive_analysis/analysis_results``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

try:
    from .analysis_utils import (
        compute_confusion_matrix,
        compute_precision_from_rates,
        compute_shadow_thresholds,
        display_top_k_vulnerable_samples,
        find_threshold_at_fpr,
        load_dataset,
        load_experiment_data,
    )
    from .plot_style import configure_popets_style
except ImportError:
    from analysis_utils import (
        compute_confusion_matrix,
        compute_precision_from_rates,
        compute_shadow_thresholds,
        display_top_k_vulnerable_samples,
        find_threshold_at_fpr,
        load_dataset,
        load_experiment_data,
    )
    from plot_style import configure_popets_style


DEFAULT_SCORE_FILES = {
    "LiRA (online)": "online_scores_leave_one_out.npy",
    "LiRA (online, fixed var)": "online_fixed_scores_leave_one_out.npy",
    "LiRA (offline)": "offline_scores_leave_one_out.npy",
    "LiRA (offline, fixed var)": "offline_fixed_scores_leave_one_out.npy",
    "Global threshold": "global_scores_leave_one_out.npy",
}


def fpr_to_tag(fpr: float) -> str:
    pct = fpr * 100.0
    value = f"{pct}".replace(".", "p")
    return value + "pct"


def _prior_tag(prior: float) -> str:
    return f"{prior}".replace(".", "p")


def save_latex_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Save one booktabs .tex table per (mode, prior) from the summary statistics DataFrame.

    Columns: Attack | AUC | TPR±std | Prec±std for each target FPR.
    """
    attack_order = [
        "LiRA (online)",
        "LiRA (online, fixed var)",
        "LiRA (offline)",
        "LiRA (offline, fixed var)",
        "Global threshold",
    ]

    def fmt(mean: float, std: float) -> str:
        return f"${mean:.2f} \\pm {std:.2f}$"

    modes = summary_df["mode"].unique()
    priors = sorted(summary_df["prior"].unique())
    fprs = sorted(summary_df["Target FPR (%)"].unique())

    for mode in modes:
        for prior in priors:
            df_slice = summary_df[(summary_df["mode"] == mode) & (summary_df["prior"] == prior)]
            if df_slice.empty:
                continue

            n_fpr = len(fprs)
            col_spec = "l" + "c" * (1 + n_fpr * 2)  # Attack + AUC + 2 per FPR

            fpr_spans = " & ".join(
                f"\\multicolumn{{2}}{{c}}{{FPR = {fpr:.4g}\\%}}" for fpr in fprs
            )
            cmidrules = " ".join(
                f"\\cmidrule(lr){{{3 + i * 2}-{4 + i * 2}}}" for i in range(n_fpr)
            )
            sub_headers = " & ".join(["TPR (\\%)", "Prec (\\%)"] * n_fpr)

            lines = [
                f"\\begin{{tabular}}{{{col_spec}}}",
                "\\toprule",
                f" &  & {fpr_spans} \\\\",
                cmidrules,
                f"Attack & AUC & {sub_headers} \\\\",
                "\\midrule",
            ]

            for attack in attack_order:
                rows = df_slice[df_slice["attack"] == attack]
                if rows.empty:
                    continue
                auc_str = fmt(rows.iloc[0]["AUC_Mean"], rows.iloc[0]["AUC_Std"])
                cells: list[str] = [attack, auc_str]
                for fpr in fprs:
                    fpr_rows = rows[np.isclose(rows["Target FPR (%)"].to_numpy(dtype=float), fpr)]
                    if fpr_rows.empty:
                        cells += ["--", "--"]
                    else:
                        r = fpr_rows.iloc[0]
                        cells += [fmt(r["TPR_Mean"], r["TPR_Std"]), fmt(r["Precision_Mean"], r["Precision_Std"])]
                lines.append(" & ".join(cells) + " \\\\")

            lines += ["\\bottomrule", "\\end{tabular}"]

            tex_path = out_dir / f"summary_{mode}_prior{_prior_tag(prior)}.tex"
            tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Saved: {tex_path}")


def create_output_directory(base_dir: Path, exp_path: Path) -> Path:
    parts = exp_path.parts
    if len(parts) >= 4:
        dataset, model, run_name = parts[-3], parts[-2], parts[-1]
    else:
        dataset, model, run_name = "unknown_dataset", "unknown_model", exp_path.name
    out_dir = base_dir / "analysis_results" / dataset / model / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def evaluate_two_modes(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    target_fprs: list[float],
    priors: list[float],
) -> pd.DataFrame:
    num_models, _ = labels.shape
    rows: list[dict[str, float | int | str]] = []

    for attack_name, score_array in scores.items():
        print(f"Evaluating {attack_name}...")
        aucs = np.full(num_models, np.nan)
        for model_idx in range(num_models):
            try:
                aucs[model_idx] = roc_auc_score(labels[model_idx].astype(int), score_array[model_idx])
            except ValueError:
                pass

        for target_fpr in target_fprs:
            target_taus = np.empty(num_models)
            for model_idx in range(num_models):
                tau, _, _ = find_threshold_at_fpr(score_array[model_idx], labels[model_idx], target_fpr)
                target_taus[model_idx] = tau

            shadow_taus = np.array([compute_shadow_thresholds(target_taus, model_idx) for model_idx in range(num_models)])

            for mode, taus in [("target", target_taus), ("shadow", shadow_taus)]:
                if mode == "shadow" and attack_name == "Global threshold":
                    continue

                for model_idx in range(num_models):
                    tau = taus[model_idx]
                    if not np.isfinite(tau):
                        tp, fp = 0, 0
                        tn = int(np.sum(~labels[model_idx]))
                        fn = int(np.sum(labels[model_idx]))
                        tpr, fpr_achieved = 0.0, 0.0
                    else:
                        tp, fp, tn, fn, tpr, fpr_achieved = compute_confusion_matrix(
                            score_array[model_idx], labels[model_idx], tau
                        )

                    for prior in priors:
                        rows.append(
                            {
                                "mode": mode,
                                "attack": attack_name,
                                "target_fpr": target_fpr,
                                "achieved_fpr": fpr_achieved,
                                "prior": prior,
                                "model_idx": model_idx,
                                "threshold": tau,
                                "tp": tp,
                                "fp": fp,
                                "tn": tn,
                                "fn": fn,
                                "tpr": tpr,
                                "precision": compute_precision_from_rates(tpr, fpr_achieved, prior),
                                "auc": aucs[model_idx],
                            }
                        )

    return pd.DataFrame(rows)


def create_summary_statistics(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        detail_df.groupby(["mode", "attack", "target_fpr", "prior"], as_index=False)
        .agg(
            TPR_Mean=("tpr", "mean"),
            TPR_Std=("tpr", "std"),
            FPR_Achieved_Mean=("achieved_fpr", "mean"),
            FPR_Achieved_Std=("achieved_fpr", "std"),
            Precision_Mean=("precision", "mean"),
            Precision_Std=("precision", "std"),
            AUC_Mean=("auc", "mean"),
            AUC_Std=("auc", "std"),
        )
    )

    percentage_cols = [
        "TPR_Mean",
        "TPR_Std",
        "FPR_Achieved_Mean",
        "FPR_Achieved_Std",
        "Precision_Mean",
        "Precision_Std",
        "AUC_Mean",
        "AUC_Std",
    ]
    for column in percentage_cols:
        summary[column] = (summary[column] * 100).round(3)

    summary["Target FPR (%)"] = (summary["target_fpr"] * 100).round(4)
    summary = summary.drop(columns=["target_fpr"])
    return summary[
        [
            "mode",
            "attack",
            "Target FPR (%)",
            "prior",
            "TPR_Mean",
            "TPR_Std",
            "FPR_Achieved_Mean",
            "FPR_Achieved_Std",
            "Precision_Mean",
            "Precision_Std",
            "AUC_Mean",
            "AUC_Std",
        ]
    ]


def compute_sample_vulnerability(
    detail_df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    labels: np.ndarray,
    attack_name: str,
    target_fpr: float,
) -> pd.DataFrame:
    mask = (
        (detail_df["mode"] == "shadow")
        & (detail_df["attack"] == attack_name)
        & np.isclose(detail_df["target_fpr"].to_numpy(dtype=float), float(target_fpr), rtol=1e-12, atol=1e-12)
    )
    shadow_info = (
        detail_df.loc[mask, ["model_idx", "threshold"]]
        .dropna(subset=["model_idx", "threshold"])
        .drop_duplicates(subset=["model_idx"])
    )
    if shadow_info.empty:
        raise ValueError(f"No shadow thresholds found for {attack_name} @ {target_fpr}")

    num_models, num_samples = scores[attack_name].shape
    thresholds = np.full(num_models, np.inf)
    for _, row in shadow_info.iterrows():
        model_idx = int(row["model_idx"])
        if 0 <= model_idx < num_models:
            thresholds[model_idx] = float(row["threshold"])

    predictions = scores[attack_name] >= thresholds[:, None]
    labels_bool = labels.astype(bool)
    tp = np.sum(predictions & labels_bool, axis=0).astype(int)
    fp = np.sum(predictions & ~labels_bool, axis=0).astype(int)
    tn = np.sum(~predictions & ~labels_bool, axis=0).astype(int)
    fn = np.sum(~predictions & labels_bool, axis=0).astype(int)
    return pd.DataFrame({"sample_id": np.arange(num_samples), "tp": tp, "fp": fp, "tn": tn, "fn": fn})


def rank_vulnerable_samples(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = sample_df.sort_values(by=["fp", "tp"], ascending=[True, False], kind="mergesort")
    highly_vulnerable = sample_df[(sample_df["fp"] == 0) & (sample_df["tp"] > 0)]
    return ranked, highly_vulnerable


def maybe_generate_visualizations(
    exp_path: Path,
    out_dir: Path,
    vuln_results: dict[float, tuple[pd.DataFrame, pd.DataFrame]],
    vuln_fprs: list[float],
    top_k: int,
    nrow: int,
    data_dir: Path,
    skip_visualization: bool,
) -> None:
    if skip_visualization:
        print("Skipping vulnerability visualizations.")
        return

    cfg_path = exp_path / "attack_config.yaml"
    if not cfg_path.exists():
        print("Config not found, skipping visualization.")
        return

    with cfg_path.open("r", encoding="utf-8") as handle:
        exp_config = yaml.safe_load(handle)

    try:
        full_dataset, _ = load_dataset(exp_config, data_dir=str(data_dir))
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Failed to load visualization dataset: {exc}")
        return

    for fpr_val in vuln_fprs:
        if fpr_val not in vuln_results:
            continue
        tag = fpr_to_tag(fpr_val)
        vuln_ranked, _ = vuln_results[fpr_val]
        save_name = f"top{top_k}_vulnerable_online_shadow_{tag}.png"
        display_top_k_vulnerable_samples(
            vulnerable_samples=vuln_ranked,
            full_dataset=full_dataset,
            k=top_k,
            nrow=nrow,
            out_dir=out_dir,
            save_name=save_name,
            font_size=8,
            badge_margin=1,
            overhang_left=1,
            overhang_up=1,
        )
        print(f"Saved vulnerability grid: {out_dir / save_name}")


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent  # noqa: F841
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Run per-experiment LiRA post analysis")
    parser.add_argument("--exp-path", required=True, type=Path, help="Experiment directory produced by train/attack stages")
    parser.add_argument("--target-fprs", nargs="+", type=float, default=[0.00001, 0.001])
    parser.add_argument("--priors", nargs="+", type=float, default=[0.01, 0.1, 0.5])
    parser.add_argument("--labels-file", default="membership_labels.npy")
    parser.add_argument("--vuln-attack", default="LiRA (online)")
    parser.add_argument("--vuln-fprs", nargs="+", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=9)
    parser.add_argument("--nrow", type=int, default=3, help="Images per row in vulnerability grid (default 3; use 4 for --top-k 16)")
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data")
    parser.add_argument("--out-root", type=Path, default=repo_root)
    parser.add_argument("--skip-visualization", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_popets_style(base_fontsize=12, family="serif")

    exp_path = args.exp_path.resolve()
    out_dir = create_output_directory(args.out_root.resolve(), exp_path)
    vuln_fprs = args.vuln_fprs or list(args.target_fprs)

    print(f"Output directory: {out_dir}")
    print("Loading experiment data...")
    labels, scores = load_experiment_data(exp_path, DEFAULT_SCORE_FILES, args.labels_file)
    num_models, num_samples = labels.shape
    print(f"Loaded {num_models} models x {num_samples} samples")
    print(f"Attacks: {list(scores.keys())}")

    print("Evaluating target and shadow modes...")
    detail_df = evaluate_two_modes(labels, scores, list(args.target_fprs), list(args.priors))
    detail_path = out_dir / "per_model_metrics_two_modes.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Saved: {detail_path}")

    print("Generating summary statistics...")
    summary_df = create_summary_statistics(detail_df)
    summary_path = out_dir / "summary_statistics_two_modes.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    save_latex_summary(summary_df, out_dir)

    print("Analyzing per-sample vulnerability...")
    vuln_results: dict[float, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fpr_val in vuln_fprs:
        sample_vuln = compute_sample_vulnerability(detail_df, scores, labels, args.vuln_attack, fpr_val)
        vuln_ranked, highly_vuln = rank_vulnerable_samples(sample_vuln)
        tag = fpr_to_tag(fpr_val)
        vuln_path = out_dir / f"samples_vulnerability_ranked_online_shadow_{tag}.csv"
        high_vuln_path = out_dir / f"samples_highly_vulnerable_online_shadow_{tag}.csv"
        vuln_ranked.to_csv(vuln_path, index=False)
        highly_vuln.to_csv(high_vuln_path, index=False)
        print(f"Saved: {vuln_path}")
        print(f"Saved: {high_vuln_path}")
        vuln_results[fpr_val] = (vuln_ranked, highly_vuln)

    maybe_generate_visualizations(
        exp_path=exp_path,
        out_dir=out_dir,
        vuln_results=vuln_results,
        vuln_fprs=vuln_fprs,
        top_k=args.top_k,
        nrow=args.nrow,
        data_dir=args.data_dir.resolve(),
        skip_visualization=args.skip_visualization,
    )


if __name__ == "__main__":
    main()
