"""Generate Figure 8: sample score distributions across benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib import ticker

try:
    from .plot_style import configure_popets_style
except ImportError:
    from plot_style import configure_popets_style


def freedman_diaconis_bins(values: np.ndarray, max_bins: int = 80, min_bins: int = 50) -> int:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return min_bins
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    if iqr <= 0:
        return min(max(min_bins, int(np.sqrt(values.size))), max_bins)
    bin_width = 2 * iqr * (values.size ** (-1 / 3))
    if bin_width <= 0:
        return min(max(min_bins, int(np.sqrt(values.size))), max_bins)
    num_bins = int(np.ceil((values.max() - values.min()) / bin_width))
    return int(np.clip(num_bins, min_bins, max_bins))


def parse_panel(panel_arg: str) -> tuple[str, Path]:
    if "=" not in panel_arg:
        raise ValueError(f"Expected TITLE=PATH, got {panel_arg!r}")
    title, path_str = panel_arg.split("=", 1)
    return title, Path(path_str)


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def resolve_config_paths(
    config: dict,
    *,
    config_path: Path,
    repo_root: Path,
    cli_experiments_root: Path | None = None,
) -> tuple[list[tuple[str, Path]], Path]:
    experiments_root = cli_experiments_root
    if experiments_root is None and config.get("experiments_root"):
        experiments_root = _resolve_path(config["experiments_root"], config_path.parent)

    panel_base = experiments_root if experiments_root is not None else config_path.parent
    panels = [(item["title"], _resolve_path(item["path"], panel_base)) for item in config["panels"]]
    out_path = _resolve_path(config.get("out", repo_root / "analysis_results" / "figures" / "sample_inout_score.pdf"), repo_root)
    return panels, out_path


def plot_sample_inout_distributions(
    panels: list[tuple[str, Path]],
    sample_idx: int,
    scores_fname: str,
    labels_fname: str,
    save_path: Path,
    single_sample: bool = True,
    share_xlim: bool = True,
    color_member: str = "#F53030",
    color_nonmember: str = "#1C9452",
) -> None:
    bench_data: dict[str, dict[str, np.ndarray]] = {}
    xmins: list[float] = []
    xmaxs: list[float] = []

    for title, exp_dir in panels:
        if not exp_dir.exists():
            raise FileNotFoundError(f"Panel directory does not exist for {title}: {exp_dir}")
        labels = np.load(exp_dir / labels_fname)
        scores = np.load(exp_dir / scores_fname)
        if scores.shape != labels.shape:
            raise ValueError(f"Shape mismatch for {title}: scores {scores.shape}, labels {labels.shape}")

        _, num_samples = labels.shape
        if not 0 <= sample_idx < num_samples:
            raise ValueError(f"sample_idx {sample_idx} out of range [0, {num_samples - 1}] for {title}")

        if single_sample:
            y_values = labels[:, sample_idx].astype(bool)
            score_values = scores[:, sample_idx].astype(np.float64)
        else:
            y_values = labels.astype(bool)
            score_values = scores.astype(np.float64)

        bench_data[title] = {"scores": score_values, "labels": y_values}
        finite_scores = score_values[np.isfinite(score_values)]
        if finite_scores.size:
            xmins.append(float(finite_scores.min()))
            xmaxs.append(float(finite_scores.max()))

    if share_xlim and xmins and xmaxs:
        xlo, xhi = min(xmins), max(xmaxs)
        pad = 0.05 * (xhi - xlo + 1e-12)
    else:
        xlo = xhi = pad = None

    num_panels = len(panels)
    ncols = 2
    nrows = int(np.ceil(num_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6, max(2.2, 1.65 * nrows)))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax in axes_flat[num_panels:]:
        ax.axis("off")

    for ax, (title, _) in zip(axes_flat, panels):
        data = bench_data[title]
        scores = data["scores"]
        labels = data["labels"]
        member_scores = scores[labels]
        nonmember_scores = scores[~labels]
        all_finite = np.concatenate([member_scores[np.isfinite(member_scores)], nonmember_scores[np.isfinite(nonmember_scores)]])
        if all_finite.size == 0:
            ax.set_title(f"{title} (no finite scores)")
            ax.axis("off")
            continue

        bins = freedman_diaconis_bins(all_finite)
        if share_xlim and xlo is not None and xhi is not None and pad is not None:
            x_range = (xlo - pad, xhi + pad)
        else:
            lo, hi = float(all_finite.min()), float(all_finite.max())
            pad_local = 0.05 * (hi - lo + 1e-12)
            x_range = (lo - pad_local, hi + pad_local)

        ax.hist(nonmember_scores, bins=bins, range=x_range, density=True, color=color_nonmember, alpha=0.6, label="Non-member", edgecolor="none")
        ax.hist(member_scores, bins=bins, range=x_range, density=True, color=color_member, alpha=0.6, label="Member", edgecolor="none")
        ax.set_title(title, pad=3)
        ax.grid(axis="y", alpha=0.25, ls="--", lw=0.4)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.tick_params(width=0.6)
        if share_xlim:
            ax.set_xlim(x_range)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=0.8, w_pad=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    png_path = save_path.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate sample IN/OUT score distributions across benchmark runs")
    parser.add_argument("--config", type=Path, help="YAML config with panels, score file, sample index, and output path")
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Base directory used to resolve relative panel paths from a config file.",
    )
    parser.add_argument(
        "--panel",
        action="append",
        default=[],
        help="Panel definition in TITLE=EXPERIMENT_DIR form. Repeat for each subplot.",
    )
    parser.add_argument("--sample-idx", type=int, default=21)
    parser.add_argument("--labels-file", default="membership_labels.npy")
    parser.add_argument("--score-file", default="global_scores_leave_one_out.npy")
    parser.add_argument("--out", type=Path, default=script_dir.parent / "analysis_results" / "figures" / "sample_inout_score.pdf")
    parser.add_argument("--all-samples", action="store_true", help="Aggregate all samples instead of plotting one sample index")
    parser.add_argument("--no-share-xlim", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    configure_popets_style(base_fontsize=12, family="serif")
    sns.set_theme(context="paper", style="whitegrid")

    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        panels, out_path = resolve_config_paths(
            config,
            config_path=args.config.resolve(),
            repo_root=repo_root,
            cli_experiments_root=args.experiments_root.resolve() if args.experiments_root is not None else None,
        )
        sample_idx = int(config.get("sample_idx", args.sample_idx))
        labels_file = str(config.get("labels_file", args.labels_file))
        score_file = str(config.get("score_file", args.score_file))
        share_xlim = bool(config.get("share_xlim", not args.no_share_xlim))
        single_sample = bool(config.get("single_sample", not args.all_samples))
    else:
        if not args.panel:
            raise SystemExit("At least one --panel TITLE=EXPERIMENT_DIR argument or --config is required.")
        panels = [parse_panel(value) for value in args.panel]
        sample_idx = args.sample_idx
        labels_file = args.labels_file
        score_file = args.score_file
        out_path = args.out
        share_xlim = not args.no_share_xlim
        single_sample = not args.all_samples

    plot_sample_inout_distributions(
        panels=panels,
        sample_idx=sample_idx,
        scores_fname=score_file,
        labels_fname=labels_file,
        save_path=out_path,
        single_sample=single_sample,
        share_xlim=share_xlim,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
