"""Importable benchmark entrypoint used by the unified CLI."""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "configs" / "benchmarks"
GENERATED_DIR = ROOT / "configs" / "generated"


def load_manifest(benchmark_id: str) -> dict:
    """Load a benchmark manifest by ID."""
    manifest_path = BENCHMARK_DIR / f"{benchmark_id}.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Unknown benchmark '{benchmark_id}'. Expected {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    manifest["_manifest_path"] = manifest_path
    return manifest


def list_benchmarks() -> list[str]:
    """List available benchmark manifests."""
    return sorted(path.stem for path in BENCHMARK_DIR.glob("*.yaml") if path.name != "README.md")


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply dot-path overrides (e.g. 'training.epochs=1') to a config dict in place."""
    for override in overrides:
        key_path, _, raw_value = override.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        # Try to cast to int/float/bool, fall back to string
        for cast in (int, float):
            try:
                raw_value = cast(raw_value)
                break
            except ValueError:
                pass
        if raw_value in ("true", "True"):
            raw_value = True
        elif raw_value in ("false", "False"):
            raw_value = False
        node[keys[-1]] = raw_value


def write_generated_configs(manifest: dict, overrides: list[str] | None = None) -> tuple[Path, Path]:
    """Write generated train/attack configs for a benchmark run."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_id = manifest["benchmark"]["id"]
    experiment_dir = manifest["experiment_dir"]

    train_cfg = copy.deepcopy(manifest["train"])
    train_cfg.setdefault("experiment", {})
    train_cfg["experiment"]["checkpoint_dir"] = experiment_dir

    attack_cfg = copy.deepcopy(manifest["attack"])
    attack_cfg.setdefault("experiment", {})
    attack_cfg["experiment"]["checkpoint_dir"] = experiment_dir

    if overrides:
        _apply_overrides(train_cfg, overrides)
        _apply_overrides(attack_cfg, overrides)

    train_path = GENERATED_DIR / f"{benchmark_id}.train.yaml"
    attack_path = GENERATED_DIR / f"{benchmark_id}.attack.yaml"

    with train_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(train_cfg, handle, sort_keys=False)
    with attack_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(attack_cfg, handle, sort_keys=False)

    return train_path, attack_path


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    """Run or print a command in a consistent benchmark-friendly format."""
    print(f"[run] cwd={cwd}")
    print("[run] " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def analysis_output_marker(experiment_dir: Path) -> Path:
    """Return the analysis summary path used to detect completed runs."""
    parts = experiment_dir.parts
    if len(parts) >= 4:
        dataset, model, run_name = parts[-3], parts[-2], parts[-1]
    else:
        dataset, model, run_name = "unknown_dataset", "unknown_model", experiment_dir.name
    return ROOT / "comprehensive_analysis" / "analysis_results" / dataset / model / run_name / "summary_statistics_two_modes.csv"


def maybe_skip(stage: str, experiment_dir: Path, skip_existing: bool) -> bool:
    """Skip a stage when its output marker already exists."""
    if not skip_existing:
        return False

    markers = {
        "train": experiment_dir / "train_config.yaml",
        "attack": experiment_dir / "attack_results_leave_one_out_summary.csv",
        "analysis": analysis_output_marker(experiment_dir),
    }
    marker = markers[stage]
    if marker.exists():
        print(f"[skip] {stage}: found {marker}")
        return True
    return False


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for benchmark runs."""
    parser = argparse.ArgumentParser(description="Run a named LiRA paper benchmark")
    parser.add_argument("--benchmark", help="Benchmark ID from configs/benchmarks")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["train", "attack", "analysis"],
        default=["train", "attack", "analysis"],
        help="Stages to execute",
    )
    parser.add_argument("--list", action="store_true", help="List available benchmark IDs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--skip-existing", action="store_true", help="Skip stages with existing outputs")
    parser.add_argument(
        "--override",
        nargs="+",
        metavar="KEY=VALUE",
        default=[],
        help="Override config values (e.g. training.epochs=1)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a benchmark manifest end to end."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for benchmark_id in list_benchmarks():
            print(benchmark_id)
        return

    if not args.benchmark:
        parser.error("--benchmark is required unless --list is used")

    manifest = load_manifest(args.benchmark)
    experiment_dir = ROOT / manifest["experiment_dir"]
    train_cfg_path, attack_cfg_path = write_generated_configs(manifest, overrides=args.override)

    if "train" in args.stages and not maybe_skip("train", experiment_dir, args.skip_existing):
        run_command([sys.executable, "train.py", "--config", str(train_cfg_path)], cwd=ROOT, dry_run=args.dry_run)

    if "attack" in args.stages and not maybe_skip("attack", experiment_dir, args.skip_existing):
        run_command([sys.executable, "attack.py", "--config", str(attack_cfg_path)], cwd=ROOT, dry_run=args.dry_run)

    if "analysis" in args.stages and not maybe_skip("analysis", experiment_dir, args.skip_existing):
        analysis_cfg = manifest.get("analysis", {})
        analysis_script = ROOT / analysis_cfg.get("script", "comprehensive_analysis/run_analysis.py")
        run_command(
            [
                sys.executable,
                str(analysis_script),
                "--exp-path",
                str(experiment_dir),
                "--target-fprs",
                *[str(value) for value in analysis_cfg.get("target_fprs", [0.00001, 0.001])],
                "--priors",
                *[str(value) for value in analysis_cfg.get("priors", [0.01, 0.1, 0.5])],
            ],
            cwd=ROOT,
            dry_run=args.dry_run,
        )
