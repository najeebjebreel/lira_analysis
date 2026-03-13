import numpy as np

from comprehensive_analysis.metrics import compute_roc_metrics, compute_tpr_at_fpr


def test_compute_roc_metrics_and_target_operating_point():
    scores = np.array([0.95, 0.90, 0.20, 0.10], dtype=float)
    labels = np.array([1, 1, 0, 0], dtype=bool)

    fpr, tpr, thresholds, auc_score, accuracy = compute_roc_metrics(scores, labels)
    tpr_at, actual_fpr, threshold = compute_tpr_at_fpr(fpr, tpr, thresholds, target_fpr=0.0)

    assert auc_score == 1.0
    assert accuracy == 1.0
    assert tpr_at == 1.0
    assert actual_fpr == 0.0
    assert np.isfinite(threshold)
