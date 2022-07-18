"""Base class for all binary classification objectives."""
import numpy as np
from scipy.optimize import differential_evolution

from evalml.objectives.objective_base import ObjectiveBase
from evalml.problem_types import ProblemTypes


class BinaryClassificationObjective(ObjectiveBase):
    """Base class for all binary classification objectives."""

    problem_types = [ProblemTypes.BINARY, ProblemTypes.TIME_SERIES_BINARY]
    """[ProblemTypes.BINARY, ProblemTypes.TIME_SERIES_BINARY]"""

    @property
    def can_optimize_threshold(cls):
        """Returns a boolean determining if we can optimize the binary classification objective threshold.

        This will be false for any objective that works directly with
        predicted probabilities, like log loss and AUC. Otherwise, it
        will be true.

        Returns:
            bool: Whether or not an objective can be optimized.
        """
        return not cls.score_needs_proba

    def optimize_threshold(self, ypred_proba, y_true, X=None):
        """Learn a binary classification threshold which optimizes the current objective.

        Args:
            ypred_proba (pd.Series): The classifier's predicted probabilities
            y_true (pd.Series): The ground truth for the predictions.
            X (pd.DataFrame, optional): Any extra columns that are needed from training data.

        Returns:
            Optimal threshold for this objective.

        Raises:
            RuntimeError: If objective cannot be optimized.
        """
        ypred_proba = self._standardize_input_type(ypred_proba)
        y_true = self._standardize_input_type(y_true)
        if X is not None:
            X = self._standardize_input_type(X)

        if not self.can_optimize_threshold:
            raise RuntimeError("Trying to optimize objective that can't be optimized!")

        def cost(threshold):
            y_predicted = self.decision_function(
                ypred_proba=ypred_proba,
                threshold=threshold[0],
                X=X,
            )
            cost = self.objective_function(y_true, y_predicted, X=X)
            return -cost if self.greater_is_better else cost

        optimal = differential_evolution(cost, bounds=[(0, 1)], seed=0, maxiter=250)

        return optimal.x[0]

    def decision_function(self, ypred_proba, threshold=0.5, X=None):
        """Apply a learned threshold to predicted probabilities to get predicted classes.

        Args:
            ypred_proba (pd.Series, np.ndarray): The classifier's predicted probabilities
            threshold (float, optional): Threshold used to make a prediction. Defaults to 0.5.
            X (pd.DataFrame, optional): Any extra columns that are needed from training data.

        Returns:
            predictions
        """
        ypred_proba = self._standardize_input_type(ypred_proba)
        return ypred_proba > threshold

    def validate_inputs(self, y_true, y_predicted):
        """Validate inputs for scoring."""
        super().validate_inputs(y_true, y_predicted)
        if len(np.unique(y_true)) > 2:
            raise ValueError("y_true contains more than two unique values")
        if len(np.unique(y_predicted)) > 2 and not self.score_needs_proba:
            raise ValueError("y_predicted contains more than two unique values")
