import numpy as np

from src.utils import normalized_gini, rmse


def test_rmse_zero_for_identical_arrays():
    y = np.array([1, 2, 3])
    assert rmse(y, y) == 0.0


def test_normalized_gini_runs():
    y_true = np.array([0, 1, 0, 3, 5])
    y_score = np.array([0.1, 0.4, 0.2, 0.8, 0.9])
    value = normalized_gini(y_true, y_score)
    assert -1 <= value <= 1
