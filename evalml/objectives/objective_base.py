from abc import ABC, abstractmethod

import numpy as np

from evalml.exceptions import DimensionMismatchError


class ObjectiveBase(ABC):
    """Base class for all objectives."""

    @property
    @classmethod
    @abstractmethod
    def name(cls):
        """Returns a name describing the objective."""

    @property
    @classmethod
    @abstractmethod
    def greater_is_better(cls):
        """Returns a boolean determining if a greater score indicates better model performance."""

    @property
    @classmethod
    @abstractmethod
    def score_needs_proba(cls):
        """Returns a boolean determining if the score() method needs probability estimates. This should be true for objectives which work with predicted probabilities, like log loss or AUC, and false for objectives which compare predicted class labels to the actual labels, like F1 or correlation.
        """

    @classmethod
    @abstractmethod
    def objective_function(cls, y_true, y_predicted, X=None):
        """Computes the relative value of the provided predictions compared to the actual labels, according a specified metric
         Arguments:
            y_predicted (pd.Series) : predicted values of length [n_samples]
            y_true (pd.Series) : actual class labels of length [n_samples]
            X (pd.DataFrame or np.array) : extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            numerical value used to calculate score
        """

    def score(self, y_true, y_predicted, X=None):
        """Returns a numerical score indicating performance based on the differences between the predicted and actual values.

        Arguments:
            y_predicted (pd.Series) : predicted values of length [n_samples]
            y_true (pd.Series) : actual class labels of length [n_samples]
            X (pd.DataFrame or np.array) : extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            score
        """
        return self.objective_function(y_true, y_predicted, X=X)

    def standard_checks(self, y_true, y_predicted):
        if len(y_true) == 0 or len(y_predicted) == 0:
            raise ValueError("Length of inputs is 0")
        if len(y_predicted) != len(y_true):
            raise DimensionMismatchError("Inputs have mismatched dimensions: y_predicted has shape {}, y_true has shape {}".format(len(y_predicted), len(y_true)))
        if self.score_needs_proba and (np.any([(y_predicted < 0) | (y_predicted > 1)])):
            raise ValueError("y_predicted contains probability estimates not within [0, 1]")
