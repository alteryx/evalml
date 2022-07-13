"""Sensitivity at Low Alert Rates objective."""
import logging

import numpy as np

from evalml.objectives.binary_classification_objective import (
    BinaryClassificationObjective,
)

logger = logging.getLogger(__name__)


class SensitivityLowAlert(BinaryClassificationObjective):
    """Create instance of SensitivityLowAlert.

    Args:
        alert_rate (float): percentage of top scores to classify as high risk.
    """

    name = "Sensitivity at Low Alert Rates"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = 1.0
    is_bounded_like_percentage = True
    expected_range = [0, 1]

    def __init__(self, alert_rate=0.01):
        if (alert_rate > 1) or (alert_rate < 0):
            raise ValueError("Alert rate is outside of valid range [0,1]")

        self.alert_rate = alert_rate

    def decision_function(self, ypred_proba, **kwargs):
        """Determine if an observation is high risk given an alert rate.

        Args:
            ypred_proba (pd.Series): Predicted probabilities.
            **kwargs: Additional abritrary parameters.

        Returns:
            pd.Series: Whether or not an observation is high risk given an alert rate.
        """
        ypred_proba = self._standardize_input_type(ypred_proba)
        if len(ypred_proba.unique()) == 1:
            logger.debug(
                f"All predicted probabilities have the same value: {ypred_proba.unique()}",
            )

        prob_thresh = np.quantile(ypred_proba, 1 - self.alert_rate)
        if (prob_thresh == 0) or (prob_thresh == 1):
            logger.debug(f"Extreme threshold of {prob_thresh}")

        return ypred_proba.astype(float) >= prob_thresh

    def objective_function(self, y_true, y_predicted, **kwargs):
        """Calculate sensitivity across all predictions, using the top alert_rate percent of observations as the predicted positive class.

        Args:
            y_true (pd.Series): True labels.
            y_predicted (pd.Series): Predicted labels based on alert_rate.
            **kwargs: Additional abritrary parameters.

        Returns:
            float: sensitivity using the observations with the top scores as the predicted positive class.
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
