import pytest

from evalml import AutoMLSearch
from evalml.objectives import SLA
from evalml.tests.objective_tests.test_binary_objective import TestBinaryObjective

class TestSLA(TestBinaryObjective):
    __test__ = True

    def assign_objective(self, alert_rate):
        self.objective = SLA(alert_rate)

    def test_sla_objective(self, X_y_binary):
        self.assign_problem_type()
        self.assign_objective(0.1)
        self.run_pipeline(X_y_binary)

    @pytest.mark.parametrize("alert_rate", [0.01, 0.99])
    def test_valid_alert_rate(self, alert_rate):
        object = SLA(alert_rate)
        assert object.alert_rate == alert_rate

    @pytest.mark.parametrize("alert_rate", [-1, 1.5])
    def test_invalid_alert_rate(self, alert_rate):
        with pytest.raises(Exception):
            SLA(alert_rate)

    # def test_single_pred_proba():

    # def test_extreme_threshold():

    # def test_sla_score():