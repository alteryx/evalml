import pandas as pd

from .binary_classification_objective import BinaryClassificationObjective


class CostBenefitMatrix(BinaryClassificationObjective):
    """Score using a cost-benefit matrix"""
    name = "Cost Benefit Matrix"
    greater_is_better = True
    score_needs_proba = False

    def __init__(self, true_positive, true_negative, false_positive, false_negative):
        """Create instance of CostBenefitMatrix

        Arguments:

        """
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

    def objective_function(self, y_true, y_predicted, X=None):
        """
            Arguments:
                y_predicted (pd.Series): predicted labels
                y_true (pd.Series): true labels
                X (pd.DataFrame): Ignored.
            Returns:
                float: score
        """
