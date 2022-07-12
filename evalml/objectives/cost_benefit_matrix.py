"""Cost-benefit matrix objective."""
import numpy as np

import evalml
from evalml.objectives.binary_classification_objective import (
    BinaryClassificationObjective,
)


class CostBenefitMatrix(BinaryClassificationObjective):
    """Score using a cost-benefit matrix. Scores quantify the benefits of a given value, so greater numeric scores represents a better score. Costs and scores can be negative, indicating that a value is not beneficial. For example, in the case of monetary profit, a negative cost and/or score represents loss of cash flow.

    Args:
        true_positive (float): Cost associated with true positive predictions.
        true_negative (float): Cost associated with true negative predictions.
        false_positive (float): Cost associated with false positive predictions.
        false_negative (float): Cost associated with false negative predictions.
    """

    name = "Cost Benefit Matrix"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = np.inf
    is_bounded_like_percentage = False  # Range (-Inf, Inf)
    expected_range = [float("-inf"), float("inf")]

    def __init__(self, true_positive, true_negative, false_positive, false_negative):
        if None in {true_positive, true_negative, false_positive, false_negative}:
            raise ValueError(
                "Parameters to CostBenefitMatrix must all be numeric values.",
            )

        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

    def objective_function(self, y_true, y_predicted, X=None, sample_weight=None):
        """Calculates cost-benefit of the using the predicted and true values.

        Args:
            y_predicted (pd.Series): Predicted labels.
            y_true (pd.Series): True labels.
            X (pd.DataFrame): Ignored.
            sample_weight (pd.DataFrame): Ignored.

        Returns:
            float: Cost-benefit matrix score
        """
        conf_matrix = evalml.model_understanding.metrics.confusion_matrix(
            y_true,
            y_predicted,
            normalize_method="all",
        )
        cost_matrix = np.array(
            [
                [self.true_negative, self.false_positive],
                [self.false_negative, self.true_positive],
            ],
        )

        total_cost = np.multiply(conf_matrix.values, cost_matrix).sum()
        return total_cost
