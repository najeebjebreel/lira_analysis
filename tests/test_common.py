import numpy as np
import torch
import yaml

from utils.common import save_config, set_seed, setup_logger


def test_setup_logger_replaces_existing_handlers(tmp_path):
    log_path = tmp_path / "train.log"

    logger = setup_logger("unit-test-logger", str(log_path))
    assert len(logger.handlers) == 2

    logger = setup_logger("unit-test-logger", str(log_path))
    assert len(logger.handlers) == 2


def test_set_seed_repeats_numpy_and_torch_sequences():
    set_seed(123, deterministic=True)
    first_numpy = np.random.rand(4)
    first_torch = torch.rand(4)

    set_seed(123, deterministic=True)
    second_numpy = np.random.rand(4)
    second_torch = torch.rand(4)

    assert np.allclose(first_numpy, second_numpy)
    assert torch.allclose(first_torch, second_torch)


def test_save_config_writes_yaml_roundtrip(tmp_path):
    config = {"seed": 7, "experiment": {"deterministic": True}}
    out_path = tmp_path / "config.yaml"

    save_config(config, str(out_path))

    with out_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    assert loaded == config
