import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import TensorDataset

import attack
import train
from comprehensive_analysis import run_analysis


def test_cli_pipeline_smoke(tmp_path, monkeypatch):
    experiment_dir = tmp_path / "experiments" / "toy" / "linear" / "smoke"
    train_config_path = tmp_path / "train_config.yaml"
    attack_config_path = tmp_path / "attack_config.yaml"

    train_config = {
        "seed": 123,
        "use_cuda": False,
        "dataset": {"name": "cifar10", "num_classes": 2, "pkeep": 0.5},
        "model": {"architecture": "resnet18"},
        "training": {
            "num_shadow_models": 2,
            "start_shadow_model_idx": 0,
            "end_shadow_model_idx": 1,
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
        },
        "experiment": {"checkpoint_dir": str(experiment_dir), "log_level": "info", "deterministic": True},
    }
    attack_config = {
        "experiment": {"checkpoint_dir": str(experiment_dir), "log_level": "info", "deterministic": True},
        "attack": {"method": ["lira"], "evaluation_mode": "leave_one_out"},
    }

    train_config_path.write_text(yaml.safe_dump(train_config, sort_keys=False), encoding="utf-8")
    attack_config_path.write_text(yaml.safe_dump(attack_config, sort_keys=False), encoding="utf-8")

    features = torch.randn(6, 3, 4, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(features, labels)
    keep_indices = np.array(
        [
            [True, False, True, False, True, False],
            [False, True, False, True, False, True],
        ],
        dtype=bool,
    )

    def fake_load_dataset(config, mode="training"):
        del config, mode
        return dataset, keep_indices.copy(), dataset, dataset

    def fake_train_target_model(config, train_dataset, test_dataset, train_dataset_eval, device, shadow_model_dir, logger):
        del config, train_dataset, test_dataset, train_dataset_eval, device, logger
        shadow_model_dir = Path(shadow_model_dir)
        shadow_model_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {"weight": torch.ones(1)}}, shadow_model_dir / "best_model.pth")
        pd.DataFrame(
            [{"Best_epoch": 1, "Train_loss": 0.2, "Test_loss": 0.3, "Train_acc (%)": 90.0, "Test_acc (%)": 85.0}]
        ).to_csv(shadow_model_dir / "metrics.csv", index=False)
        return object()

    class FakeLiRA:
        def __init__(self, config, logger):
            del logger
            self.config = config
            self.experiment_dir = Path(config["experiment"]["checkpoint_dir"])

        def generate_logits(self):
            return None

        def compute_scores(self):
            return None

        def plot(self, ntest=1, metric="auc"):
            del ntest, metric
            labels_array = keep_indices.copy()
            base_scores = np.array(
                [
                    [0.9, 0.2, 0.8, 0.1, 0.85, 0.05],
                    [0.1, 0.85, 0.15, 0.9, 0.2, 0.95],
                ],
                dtype=float,
            )
            np.save(self.experiment_dir / "membership_labels.npy", labels_array)
            np.save(self.experiment_dir / "online_scores_leave_one_out.npy", base_scores)
            np.save(self.experiment_dir / "online_fixed_scores_leave_one_out.npy", base_scores - 0.05)
            np.save(self.experiment_dir / "offline_scores_leave_one_out.npy", base_scores - 0.1)
            np.save(self.experiment_dir / "offline_fixed_scores_leave_one_out.npy", base_scores - 0.15)
            np.save(self.experiment_dir / "global_scores_leave_one_out.npy", base_scores - 0.2)
            pd.DataFrame([{"Attack": "LiRA (online)", "AUC Mean": 100.0, "Acc Mean": 100.0}]).to_csv(
                self.experiment_dir / "attack_results_leave_one_out_summary.csv",
                index=False,
            )

    monkeypatch.setattr(train, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(train, "train_target_model", fake_train_target_model)
    monkeypatch.setattr(attack, "LiRA", FakeLiRA)

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", str(train_config_path)],
    )
    train.main()

    monkeypatch.setattr(
        sys,
        "argv",
        ["attack.py", "--config", str(attack_config_path)],
    )
    attack.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_analysis.py",
            "--exp-path",
            str(experiment_dir),
            "--target-fprs",
            "0.5",
            "--priors",
            "0.5",
            "--out-root",
            str(tmp_path),
            "--skip-visualization",
        ],
    )
    run_analysis.main()

    analysis_dir = tmp_path / "analysis_results" / "toy" / "linear" / "smoke"
    assert (experiment_dir / "train_config.yaml").exists()
    assert (experiment_dir / "attack_config.yaml").exists()
    assert (experiment_dir / "membership_labels.npy").exists()
    assert (analysis_dir / "summary_statistics_two_modes.csv").exists()
