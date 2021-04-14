import numpy as np
import pandas as pd
import pytest

from evalml.objectives import SensitivityLowAlert
from evalml.tests.objective_tests.test_binary_classification_objective import (
    TestBinaryObjective
)


class TestSLA(TestBinaryObjective):
    __test__ = True

    def assign_objective(self, alert_rate):
        self.objective = SensitivityLowAlert(alert_rate)

    def test_sla_objective(self, X_y_binary):
        self.assign_problem_type()
        self.assign_objective(0.1)
        self.run_pipeline(X_y_binary)

    @pytest.mark.parametrize("alert_rate", [0.01, 0.99])
    def test_valid_alert_rate(self, alert_rate):
        obj = SensitivityLowAlert(alert_rate)
        assert obj.alert_rate == alert_rate

    @pytest.mark.parametrize("alert_rate", [-1, 1.5])
    def test_invalid_alert_rate(self, alert_rate):
        with pytest.raises(ValueError):
            SensitivityLowAlert(alert_rate)

    @pytest.mark.parametrize("alert_rate, ypred_proba, high_risk", [
        (0.1, pd.Series([0.5, 0.5, 0.5]), [True, True, True]),
        (0.1, list(range(10)), [False if i != 9 else True for i in range(10)])])
    def test_high_risk_output(self, alert_rate, ypred_proba, high_risk):
        self.assign_objective(alert_rate)
        assert self.objective.decision_function(ypred_proba).tolist() == high_risk

    @pytest.mark.parametrize("y_true, y_predicted, expected_score", [
        (pd.Series([False, False, False]), pd.Series([True, True, False]), np.nan),
        (pd.Series([True, True, True, True]), pd.Series([True, True, False, False]), 0.5)])
    def test_score(self, y_true, y_predicted, expected_score):
        sensitivity = SensitivityLowAlert(0.1).objective_function(y_true, y_predicted)
        assert (sensitivity is expected_score) or (sensitivity == expected_score)

    def test_all_base_tests(self, fix_y_pred_na, fix_y_true, fix_y_pred_diff_len, fix_empty_array, fix_y_pred_multi):
        self.assign_objective(0.1)
        self.input_contains_nan_inf(fix_y_pred_na, fix_y_true)
        self.different_input_lengths(fix_y_pred_diff_len, fix_y_true)
        self.zero_input_lengths(fix_empty_array)
        self.binary_more_than_two_unique_values(fix_y_pred_multi, fix_y_true)
