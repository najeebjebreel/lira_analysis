# --- Headless-safe backend ---
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------ Okabe–Ito palette (colorblind-safe) ------------------
OI_BLUE   = "#0072B2"
OI_ORANGE = "#E69F00"
OI_VERMIL = "#D55E00"
OI_GREY   = "#888888"

# ------------------ Styling ------------------
def _style_for_papers():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # vector-friendly fonts
        # subtle boxplot defaults
        "boxplot.flierprops.marker": "o",
        "boxplot.flierprops.markersize": 2.0,
        "boxplot.flierprops.markerfacecolor": OI_GREY,
        "boxplot.flierprops.markeredgecolor": OI_GREY,
        "boxplot.medianprops.color": OI_VERMIL,
        "boxplot.medianprops.linewidth": 1.2,
        "boxplot.whiskerprops.linewidth": 0.9,
        "boxplot.capprops.linewidth": 0.9,
    })

# ------------------ Data loading ------------------
def _load_thresholds(csv_path, target_fpr, attack_contains="online"):
    df = pd.read_csv(csv_path)
    df["target_fpr"] = pd.to_numeric(df["target_fpr"], errors="coerce")
    m_attack = df["attack"].astype(str).str.lower().str.contains(attack_contains.lower())
    m_tfpr   = np.isclose(df["target_fpr"].to_numpy(), float(target_fpr), rtol=1e-6, atol=1e-12)
    vals = pd.to_numeric(df.loc[m_attack & m_tfpr, "threshold"], errors="coerce").to_numpy()
    vals = vals[np.isfinite(vals)]
    return vals

def _load_thresholds_many(csv_paths, target_fpr, attack_contains="online"):
    series = []
    for p in csv_paths:
        v = _load_thresholds(p, target_fpr, attack_contains)
        if v.size:
            series.append(v)
    return np.concatenate(series) if series else np.array([])

# ------------------ Robust dispersion (median-centric) ------------------
def _median_rmad(vals):
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    rmad = 100.0 * 1.4826 * mad / med if med != 0 else np.nan
    return med, rmad

# ------------------ Main: side-by-side BOX with Q3-aligned labels ------------------
def plot_thresholds_box_q3labels(
    single_csv,
    pooled_csvs,       # list[str]: CSVs to pool for the right box
    out_path,
    target_fpr=1e-5,
    labels=(r"(a) Single run ($M{=}256$)", r"(b) Five runs ($5{\times}256$)"),
    show_fliers=False,
    whisker_mode="tukey"  # "tukey" (1.5*IQR) or "p05p95" (5–95% whiskers)
):
    vals_single = _load_thresholds(single_csv, target_fpr)
    vals_pooled = _load_thresholds_many(pooled_csvs, target_fpr)
    if vals_single.size == 0:
        raise ValueError("No thresholds in single_csv for the given filters.")
    if vals_pooled.size == 0:
        raise ValueError("No thresholds found across pooled_csvs for the given filters.")

    _style_for_papers()

    whis = (5, 95) if whisker_mode == "p05p95" else 1.5
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    plt.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.24)

    data = [vals_single, vals_pooled]
    bp = ax.boxplot(
        data,
        vert=True,
        patch_artist=True,
        widths=0.55,
        whis=whis,
        showfliers=show_fliers
    )

    # Colors
    edgecolors = [OI_BLUE, OI_ORANGE]
    facecolors = [OI_BLUE + "33", OI_ORANGE + "33"]
    for patch, fc, ec in zip(bp['boxes'], facecolors, edgecolors):
        patch.set_facecolor(fc)
        patch.set_edgecolor(ec)
        patch.set_linewidth(1.0)
    for med in bp['medians']:
        med.set_color(OI_VERMIL); med.set_linewidth(1.2)
    for part in ["whiskers", "caps"]:
        for i, line in enumerate(bp[part]):
            ec = edgecolors[0] if i < 2 else edgecolors[1]
            line.set_color(ec); line.set_linewidth(0.9)

    # Axes labels
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(r"Threshold $\tau$")

    # ---- Q3-aligned annotations (placed just above Q3) ----
    for xpos, vals, color in zip([1, 2], data, edgecolors):
        med, rmad = _median_rmad(vals)
        q3 = float(np.percentile(vals, 75))
        ax.annotate(
            f"Median = {med:.2f}\n"
            f"rMAD = {rmad:.1f}%",
            xy=(xpos, q3),                    # anchor at top of box
            xytext=(5, 12),                   # shift right & upward (points)
            textcoords="offset points",
            ha="left", va="bottom",           # position above box
            fontsize=6.5, color="#1A2732",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.4),
            clip_on=False
        )


    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return str(out_path)

# ------------------ Example ------------------
plot_thresholds_box_q3labels(
    single_csv="thresholds/seed42.csv",
    pooled_csvs=[
        "thresholds/seed0.csv", "thresholds/seed7.csv", "thresholds/seed42.csv",
        "thresholds/seed123.csv", "thresholds/seed1234.csv",
    ],
    out_path="thresholds/thresh_sidebyside_box_q3labels.pdf",
    target_fpr=1e-5,
    show_fliers=False,
    whisker_mode="tukey"
)
