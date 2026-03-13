from pathlib import Path

from comprehensive_analysis.plot_benchmark_distribution import resolve_config_paths


def test_resolve_config_paths_uses_cli_experiments_root(tmp_path):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "configs" / "figure_panels"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "figure8.yaml"
    config_path.write_text("placeholder", encoding="utf-8")

    config = {
        "panels": [{"title": "Baseline", "path": "cifar10/resnet18/run1"}],
        "out": "comprehensive_analysis/figures/out.pdf",
    }
    experiments_root = tmp_path / "external_experiments"

    panels, out_path = resolve_config_paths(
        config,
        config_path=config_path,
        repo_root=repo_root,
        cli_experiments_root=experiments_root,
    )

    assert panels == [("Baseline", experiments_root / "cifar10" / "resnet18" / "run1")]
    assert out_path == repo_root / "comprehensive_analysis" / "figures" / "out.pdf"
