# plot_style_popets.py
import matplotlib as mpl
import matplotlib.pyplot as plt

def configure_popets_style(
    base_fontsize: float = 9.0,
    family: str = "serif",
) -> None:
    """
    Unified Matplotlib style for PoPETs:
      - TrueType fonts (Type 42) only, no Type 3.
      - Embedded, subset fonts.
      - Compact, camera-ready sizes.
      - Works without LaTeX (text.usetex=False), using mathtext instead.
    """

    # Choose font family
    if family == "serif":
        font_rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "CMU Serif", "Nimbus Roman"],
            "mathtext.fontset": "stix",  # Times-like math
        }
    else:  # e.g. "sans"
        font_rc = {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
        }

    rc = {
        # --- FONT / PDF SETTINGS (PoPETs-relevant) ---
        "text.usetex": False,        # avoid LaTeX; use mathtext (no Type 3 surprises)
        "pdf.fonttype": 42,          # TrueType / Type 42, no Type 3
        "ps.fonttype": 42,
        "pdf.use14corefonts": False, # force embedding instead of core-only fonts

        # --- DPI (vector anyway, but helpful for PNGs/previews) ---
        "figure.dpi": 300,
        "savefig.dpi": 400,

        # --- BASE SIZES (good for 1- and 2-column figures) ---
        "font.size": base_fontsize,
        "axes.labelsize": base_fontsize,
        "axes.titlesize": base_fontsize,
        "legend.fontsize": base_fontsize - 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # --- Tick aesthetics ---
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # --- Grid defaults (light, print-friendly) ---
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
    }

    rc.update(font_rc)
    mpl.rcParams.update(rc)
