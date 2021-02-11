
import numpy as np

from .binary_classification_objective import BinaryClassificationObjective

from evalml.model_understanding.graphs import confusion_matrix


class CostBenefitMatrix(BinaryClassificationObjective):
    """Score using a cost-benefit matrix. Scores quantify the benefits of a given value, so greater numeric
        scores represents a better score. Costs and scores can be negative, indicating that a value is not beneficial.
        For example, in the case of monetary profit, a negative cost and/or score represents loss of cash flow."""
    name = "Cost Benefit Matrix"
    greater_is_better = True
    score_needs_proba = False
    perfect_score = np.inf
    is_bounded_like_percentage = False  # Range (-Inf, Inf)

    def __init__(self, true_positive, true_negative, false_positive, false_negative):
        """Create instance of CostBenefitMatrix.

        Arguments:
            true_positive (float): Cost associated with true positive predictions
            true_negative (float): Cost associated with true negative predictions
            false_positive (float): Cost associated with false positive predictions
            false_negative (float): Cost associated with false negative predictions
        """
        if None in {true_positive, true_negative, false_positive, false_negative}:
            raise ValueError("Parameters to CostBenefitMatrix must all be numeric values.")

        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

    def objective_function(self, y_true, y_predicted, X=None):
        """Calculates cost-benefit of the using the predicted and true values.

        Arguments:
            y_predicted (pd.Series, ww.DataColumn): Predicted labels
            y_true (pd.Series, ww.DataColumn): True labels
            X (pd.DataFrame, ww.DataTable): Ignored.

        Returns:
            float: Cost-benefit matrix score
        """
        conf_matrix = confusion_matrix(y_true, y_predicted, normalize_method='all')
        cost_matrix = np.array([[self.true_negative, self.false_positive],
                                [self.false_negative, self.true_positive]])

        total_cost = np.multiply(conf_matrix.values, cost_matrix).sum()
        return total_cost
