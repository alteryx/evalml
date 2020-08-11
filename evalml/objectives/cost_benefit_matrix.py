
import numpy as np

from .binary_classification_objective import BinaryClassificationObjective

from evalml.utils.graph_utils import confusion_matrix


class CostBenefitMatrix(BinaryClassificationObjective):
    """Score using a cost-benefit matrix"""
    name = "Cost Benefit Matrix"
    greater_is_better = True
    score_needs_proba = False

    def __init__(self, true_positive, true_negative, false_positive, false_negative):
        """Create instance of CostBenefitMatrix.

        Arguments:
            true_positive (float): Cost associated with true positive predictions
            true_negative (float): Cost associated with true negative predictions
            false_positive (float): Cost associated with false positive predictions
            false_negative (float): Cost associated with false negative predictions
        """
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

    def objective_function(self, y_true, y_predicted, X=None):
        """Calculates cost-benefit of the using the predicted and true values.

        Arguments:
            y_predicted (pd.Series): Predicted labels
            y_true (pd.Series): True labels
            X (pd.DataFrame): Ignored.

        Returns:
            float: score
        """
        conf_matrix = confusion_matrix(y_true, y_predicted, normalize_method=None)
        cost_matrix = np.array([[self.true_negative, self.false_positive],
                                [self.false_negative, self.true_positive]])

        total_cost = np.multiply(conf_matrix.values, cost_matrix).sum()
        return total_cost
