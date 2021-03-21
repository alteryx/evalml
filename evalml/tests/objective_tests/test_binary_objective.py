from abc import ABCMeta, abstractmethod

import numpy as np
import pytest

from evalml import AutoMLSearch


@pytest.fixture()
def fix_y_pred_na():
    return np.array([np.nan, 0, 0])


@pytest.fixture()
def fix_y_true():
    return np.array([1, 2, 1])


@pytest.fixture()
def fix_y_pred_diff_len():
    return np.array([0, 1])


@pytest.fixture()
def fix_empty_array():
    np.array([])


@pytest.fixture()
def fix_y_pred_multi():
    return np.array([0, 1, 2])


class TestBinaryObjective(metaclass=ABCMeta):
    __test__ = False

    def assign_problem_type(self):
        self.problem_type = 'binary'

    @abstractmethod
    def assign_objective(self, **kwargs):
        """Get objective object using specified parameters
        """

    def run_pipeline(self, X_y_binary, **kwargs):
        self.X, self.y = X_y_binary
        automl = AutoMLSearch(X_train=self.X, y_train=self.y, problem_type=self.problem_type, objective=self.objective, max_iterations=1)
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
            y_true (ww.DataColumn, pd.Series): true classes
            y_predicted (ww.DataColumn, pd.Series): predicted classes
            expected_score (float): expected output from objective.objective_function()
        """

    @abstractmethod
    def test_all_base_tests(self):
        """Run all relevant tests from the base class
        """

    def test_input_contains_nan_inf(self, y_predicted, y_true):
        with pytest.raises(ValueError, match="y_predicted contains NaN or infinity"):
            self.objective.score(y_true, y_predicted)

    def test_different_input_lengths(self, y_predicted, y_true):
        with pytest.raises(ValueError, match="Inputs have mismatched dimensions"):
            self.objective.score(y_true, y_predicted)

    def test_zero_input_lengths(self, y_predicted, y_true):
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'shape'"):
            self.objective.score(y_true, y_predicted)

    def test_binary_more_than_two_unique_values(self, y_predicted, y_true):
        with pytest.raises(ValueError, match="y_true contains more than two unique values"):
            self.objective.score(y_true, y_predicted)
