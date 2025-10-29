"""
Visualization utilities for LiRA attack analysis.

This module provides functions for:
- ROC curve plotting
- Score distribution visualization
- Threshold distribution plots
- Sample vulnerability grids
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Optional, List, Tuple, Union
import torch
import torchvision.utils as vutils

# Configure matplotlib for publication-quality figures
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Okabe-Ito colorblind-safe palette
COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'vermillion': '#D55E00',
    'sky_blue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'purple': '#CC79A7',
    'grey': '#888888',
}


def setup_paper_style():
    """Configure matplotlib for paper-quality figures."""
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def plot_roc_curves(roc_data: dict,
                    save_path: Optional[Union[str, Path]] = None,
                    title: Optional[str] = None,
                    figsize: Tuple[float, float] = (6, 5),
                    xlim: Tuple[float, float] = (1e-5, 1),
                    ylim: Tuple[float, float] = (1e-5, 1)) -> plt.Figure:
    """
    Plot ROC curves for multiple attacks.

    Args:
        roc_data: Dictionary mapping attack names to (fpr, tpr, auc) tuples
        save_path: Path to save figure (optional)
        title: Plot title (optional)
        figsize: Figure size
        xlim: X-axis limits
        ylim: Y-axis limits

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each attack
    for i, (attack_name, (fpr, tpr, auc_score)) in enumerate(roc_data.items()):
        color = list(COLORS.values())[i % len(COLORS)]
        ax.plot(fpr, tpr, label=f'{attack_name} (AUC={auc_score:.3f})',
               color=color, linewidth=1.5)

    # Diagonal reference line
    ax.plot([1e-5, 1], [1e-5, 1], 'k--', alpha=0.3, linewidth=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    if title:
        ax.set_title(title)

    ax.legend(loc='best', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved ROC plot to {save_path}")

    return fig


def plot_score_distributions(member_scores: np.ndarray,
                             non_member_scores: np.ndarray,
                             attack_name: str = "Attack",
                             save_path: Optional[Union[str, Path]] = None,
                             bins: int = 50,
                             figsize: Tuple[float, float] = (8, 5)) -> plt.Figure:
    """
    Plot score distributions for members vs non-members.

    Args:
        member_scores: Scores for members
        non_member_scores: Scores for non-members
        attack_name: Name of the attack
        save_path: Path to save figure (optional)
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histograms
    ax.hist(non_member_scores, bins=bins, alpha=0.6, color=COLORS['blue'],
           label='Non-members', density=True, edgecolor='none')
    ax.hist(member_scores, bins=bins, alpha=0.6, color=COLORS['orange'],
           label='Members', density=True, edgecolor='none')

    ax.set_xlabel('Attack Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{attack_name}: Score Distribution')
    ax.legend(frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved distribution plot to {save_path}")

    return fig


def plot_threshold_boxplots(threshold_data: dict,
                            save_path: Optional[Union[str, Path]] = None,
                            title: str = "Threshold Distribution",
                            ylabel: str = r"Threshold $\tau$",
                            figsize: Tuple[float, float] = (8, 5),
                            show_fliers: bool = False) -> plt.Figure:
    """
    Plot box plots of thresholds for different conditions.

    Args:
        threshold_data: Dictionary mapping labels to threshold arrays
        save_path: Path to save figure (optional)
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
        show_fliers: Whether to show outliers

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(threshold_data.keys())
    data = [threshold_data[label] for label in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showfliers=show_fliers, widths=0.6)

    # Color boxes
    colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]
    for patch, color in zip(bp['boxes'], colors * (len(labels) // len(colors) + 1)):
        patch.set_facecolor(color + '33')  # Add transparency
        patch.set_edgecolor(color)

    # Color medians
    for median in bp['medians']:
        median.set_color(COLORS['vermillion'])
        median.set_linewidth(1.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved threshold boxplot to {save_path}")

    return fig


def _to_chw_float_tensor(img):
    """Convert image to CHW float tensor for visualization."""
    if isinstance(img, torch.Tensor):
        t = img.clone()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1)
        t = t.float()
        if t.numel() and t.max() > 1.0:
            t = t / 255.0
    else:
        arr = np.array(img)
        t = torch.from_numpy(arr)
        if t.ndim == 2:
            t = t.unsqueeze(-1)
        if t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        t = t.float()
        if t.numel() and t.max() > 1.0:
            t = t / 255.0

    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] > 3:
        t = t[:3]

    return t


def plot_vulnerable_samples_grid(sample_indices: np.ndarray,
                                 dataset: object,
                                 confusion_stats: np.ndarray,
                                 save_path: Optional[Union[str, Path]] = None,
                                 k: int = 20,
                                 nrow: int = 5,
                                 padding: int = 2,
                                 figsize: Optional[Tuple[float, float]] = None,
                                 title: str = "Most Vulnerable Samples") -> plt.Figure:
    """
    Create a grid visualization of vulnerable samples with TP/FP annotations.

    Args:
        sample_indices: Indices of samples to visualize
        dataset: PyTorch dataset object
        confusion_stats: Array with shape [N, 2] containing (TP, FP) for each sample
        save_path: Path to save figure (optional)
        k: Number of samples to display
        nrow: Number of samples per row
        padding: Padding between images
        figsize: Figure size (auto-calculated if None)
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    k = min(k, len(sample_indices))
    indices = sample_indices[:k]

    # Load images
    images = []
    for idx in indices:
        img, _ = dataset[int(idx)]
        img_tensor = _to_chw_float_tensor(img)
        images.append(img_tensor)

    tensor = torch.stack(images)  # [k, 3, H, W]

    # Create grid
    grid = vutils.make_grid(tensor, nrow=nrow, padding=padding,
                           normalize=True, pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()

    # Calculate figure size
    if figsize is None:
        rows = (k + nrow - 1) // nrow
        figsize = (max(4, nrow * 1.2), max(4, rows * 1.2))

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.imshow(grid_np, aspect='equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=10)

    # Annotate with TP/FP counts
    H, W = tensor.shape[-2:]
    stride_x, stride_y = W + padding, H + padding
    base_x = base_y = padding

    for i, idx in enumerate(indices):
        r, c = divmod(i, nrow)
        x0 = base_x + c * stride_x
        y0 = base_y + r * stride_y

        tp_val, fp_val = confusion_stats[idx]
        text = f"TP:{int(tp_val)}  FP:{int(fp_val)}"

        ax.text(x0 + 5, y0 + 5, text,
               ha='left', va='top', fontsize=6.5, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.9, edgecolor='black', linewidth=0.7),
               color='black', clip_on=False)

    plt.tight_layout(pad=0.1)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        print(f"Saved vulnerable samples grid to {save_path}")

    return fig


def plot_comparison_bars(data: dict,
                         save_path: Optional[Union[str, Path]] = None,
                         ylabel: str = "Metric Value",
                         title: str = "Attack Comparison",
                         figsize: Tuple[float, float] = (10, 6)) -> plt.Figure:
    """
    Plot bar chart comparing metrics across attacks.

    Args:
        data: Dictionary mapping attack names to values
        save_path: Path to save figure (optional)
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    attacks = list(data.keys())
    values = list(data.values())
    x_pos = np.arange(len(attacks))

    colors_list = list(COLORS.values())
    bar_colors = [colors_list[i % len(colors_list)] for i in range(len(attacks))]

    bars = ax.bar(x_pos, values, color=bar_colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved comparison bars to {save_path}")

    return fig
