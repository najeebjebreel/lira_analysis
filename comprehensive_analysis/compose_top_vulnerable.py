"""Collect per-run top-vulnerable sample grids into a single browsable folder.

Globs all top*_vulnerable*.png files under analysis_results/ (excluding
anything already inside analysis_results/figures/) and copies them to
analysis_results/figures/topk_vulnerable_images/ preserving the
dataset/arch/run subfolder hierarchy.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def collect_topk_images(repo_root: Path, dest: Path) -> list[Path]:
    src_root = repo_root / "analysis_results"
    figures_root = src_root / "figures"
    dest.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for src in sorted(src_root.rglob("top*vulnerable*.png")):
        # Skip anything already under figures/
        try:
            src.relative_to(figures_root)
            continue
        except ValueError:
            pass

        # Preserve relative path from analysis_results/ → dest/
        rel = src.relative_to(src_root)
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Collect per-run top-vulnerable images into analysis_results/figures/topk_vulnerable_images/"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo_root / "analysis_results" / "figures" / "topk_vulnerable_images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    copied = collect_topk_images(repo_root, args.dest)
    if copied:
        for p in copied:
            print(f"  {p.relative_to(repo_root)}")
        print(f"\nCollected {len(copied)} image(s) → {args.dest.relative_to(repo_root)}")
    else:
        print("No top-vulnerable images found (run run_analysis.py first).")


if __name__ == "__main__":
    main()
