import numpy as np

from utils.data_utils import get_keep_indices


def test_get_keep_indices_single_model_has_expected_shape():
    keep_none = get_keep_indices(dataset_size=8, num_shadow_models=1, pkeep=0.0, seed=0)
    keep_all = get_keep_indices(dataset_size=8, num_shadow_models=1, pkeep=1.0, seed=0)

    assert keep_none.shape == (1, 8)
    assert keep_all.shape == (1, 8)
    assert not keep_none.any()
    assert keep_all.all()


def test_get_keep_indices_multi_model_is_reproducible_and_balanced():
    keep_a = get_keep_indices(dataset_size=11, num_shadow_models=4, pkeep=0.5, seed=42)
    keep_b = get_keep_indices(dataset_size=11, num_shadow_models=4, pkeep=0.5, seed=42)

    assert np.array_equal(keep_a, keep_b)
    assert np.all(keep_a.sum(axis=0) == 2)
