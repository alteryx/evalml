import numpy as np

from .binary_classification_objective import BinaryClassificationObjective

from evalml.utils.logger import get_logger

logger = get_logger(__file__)


class SensitivityLowAlert(BinaryClassificationObjective):
    name = "Sensitivity at Low Alert Rates"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True

    def __init__(self, alert_rate=0.01):
        """Create instance of SensitivityLowAlert

        Arguments:
            alert_rate (float): percentage of top scores to classify as high risk

        """
        if (alert_rate > 1) or (alert_rate < 0):
            raise ValueError("Alert rate is outside of valid range [0,1]")

        self.alert_rate = alert_rate

    def decision_function(self, ypred_proba, **kwargs):
        """Determine if an observation is high risk given an alert rate

        Arguments:
            ypred_proba (ww.DataColumn, pd.Series): Predicted probabilities
        """

        ypred_proba = self._standardize_input_type(ypred_proba)
        if len(ypred_proba.unique()) == 1:
            logger.debug(f"All predicted probabilities have the same value: {ypred_proba.unique()}")

        prob_thresh = np.quantile(ypred_proba, 1 - self.alert_rate)
        if (prob_thresh == 0) or (prob_thresh == 1):
            logger.debug(f"Extreme threshold of {prob_thresh}")

        return ypred_proba.astype(float) >= prob_thresh

    def objective_function(self, y_true, y_predicted, **kwargs):
        """Calculate sensitivity across all predictions, using the top alert_rate percent of observations as the predicted positive class

        Arguments:
            y_true (ww.DataColumn, pd.Series): True labels
            y_predicted (ww.DataColumn, pd.Series): Predicted labels based on alert_rate

        Returns:
            float: sensitivity using the observations with the top scores as the predicted positive class
        """
        y_true = self._standardize_input_type(y_true)
        y_predicted = self._standardize_input_type(y_predicted)
        self.validate_inputs(y_true, y_predicted)

        tp = y_true & y_predicted
        fn = y_true & (~y_predicted)

        if (tp.sum() + fn.sum()) > 0:
            sensitivity = tp.sum() / (tp.sum() + fn.sum())
        else:
            sensitivity = np.nan

        return sensitivity
