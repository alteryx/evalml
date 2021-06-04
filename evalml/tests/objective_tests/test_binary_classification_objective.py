from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import pytest

from evalml import AutoMLSearch
from evalml.objectives.standard_metrics import AUC, F1


def test_optimize_threshold():
    ypred_proba = np.array([0.2, 0.4])
    y_true = np.array([0, 1])
    obj = F1()
    np.random.seed(
        42
    )  # unfortunately scipy.optimize.minimize_scalar has no ability to accept seed as input
    threshold = obj.optimize_threshold(ypred_proba, y_true)
    assert 0.2 < threshold and threshold < 0.4


def test_optimize_threshold_neg():
    ypred_proba = np.array([0.2, 0.4])
    y_true = np.array([0, 1])
    obj = AUC()
    np.random.seed(0)
    with pytest.raises(
        RuntimeError, match="Trying to optimize objective that can't be optimized!"
    ):
        obj.optimize_threshold(ypred_proba, y_true)


def test_can_optimize_threshold():
    assert F1().can_optimize_threshold
    assert not AUC().can_optimize_threshold


def test_decision_function():
    ypred_proba = np.arange(6) / 5.0
    obj = F1()
    pd.testing.assert_series_equal(
        obj.decision_function(ypred_proba),
        pd.Series(np.array([0] * 3 + [1] * 3), dtype=bool),
    )
    pd.testing.assert_series_equal(
        obj.decision_function(ypred_proba, threshold=0.5),
        pd.Series(np.array([0] * 3 + [1] * 3), dtype=bool),
    )
    pd.testing.assert_series_equal(
        obj.decision_function(ypred_proba, threshold=0.0),
        pd.Series(np.array([0] + [1] * 5, dtype=int), dtype=bool),
    )
    pd.testing.assert_series_equal(
        obj.decision_function(ypred_proba, threshold=1.0),
        pd.Series(np.array([0] * 6, dtype=int), dtype=bool),
    )


def test_decision_function_neg():
    ypred_proba = np.arange(6) / 5.0
    y_true = pd.Series(np.array([0] * 3 + [1] * 3), dtype=bool)
    obj = F1()
    pd.testing.assert_series_equal(obj.decision_function(ypred_proba), y_true)
    pd.testing.assert_series_equal(
        obj.decision_function(pd.Series(ypred_proba, dtype=float)), y_true
    )


class TestBinaryObjective(metaclass=ABCMeta):
    __test__ = False

    def assign_problem_type(self):
        self.problem_type = "binary"

    @abstractmethod
    def assign_objective(self, **kwargs):
        """Get objective object using specified parameters"""

    def run_pipeline(self, X_y_binary, **kwargs):
        self.X, self.y = X_y_binary
        automl = AutoMLSearch(
            X_train=self.X,
            y_train=self.y,
            problem_type=self.problem_type,
            objective=self.objective,
            max_iterations=1,
        )
        automl.search()

        pipeline = automl.best_pipeline
        pipeline.fit(self.X, self.y)
        pipeline.predict(self.X, self.objective)
        pipeline.predict_proba(self.X)
        pipeline.score(self.X, self.y, [self.objective])

    @abstractmethod
    def test_score(self, y_true, y_predicted, expected_score):
        """Objective score matches expected score

        Args:
            y_true (pd.Series): true classes
            y_predicted (pd.Series): predicted classes
            expected_score (float): expected output from objective.objective_function()
        """

    @abstractmethod
    def test_all_base_tests(self):
        """Run all relevant tests from the base class"""

    @pytest.fixture(scope="class")
    def fix_y_pred_na(self):
        return np.array([np.nan, 0, 0])

    @pytest.fixture(scope="class")
    def fix_y_true(self):
        return np.array([1, 2, 1])

    @pytest.fixture(scope="class")
    def fix_y_pred_diff_len(self):
        return np.array([0, 1])

    @pytest.fixture(scope="class")
    def fix_empty_array(self):
        return np.array([])

    @pytest.fixture(scope="class")
    def fix_y_pred_multi(self):
        return np.array([0, 1, 2])

    def input_contains_nan_inf(self, fix_y_pred_na, fix_y_true):
        with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
            self.objective.score(fix_y_true, fix_y_pred_na)

    def different_input_lengths(self, fix_y_pred_diff_len, fix_y_true):
        with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
            self.objective.score(fix_y_true, fix_y_pred_diff_len)

    def zero_input_lengths(self, fix_empty_array):
        with pytest.raises(ValueError, match="Length of inputs is 0"):
            self.objective.score(fix_empty_array, fix_empty_array)

    def binary_more_than_two_unique_values(self, fix_y_pred_multi, fix_y_true):
        with pytest.raises(
            ValueError, match="y_predicted contains more than two unique values"
        ):
            self.objective.score(fix_y_true, fix_y_pred_multi)
