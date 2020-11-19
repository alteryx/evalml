from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import woodwork as ww

from evalml.utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


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

    @property
    @classmethod
    @abstractmethod
    def perfect_score(cls):
        """Returns the score obtained by evaluating this objective on a perfect model."""

    @classmethod
    @abstractmethod
    def objective_function(cls, y_true, y_predicted, X=None):
        """Computes the relative value of the provided predictions compared to the actual labels, according a specified metric
         Arguments:
            y_predicted (pd.Series): Predicted values of length [n_samples]
            y_true (pd.Series): Actual class labels of length [n_samples]
            X (pd.DataFrame or np.ndarray): Extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            Numerical value used to calculate score
        """

    def score(self, y_true, y_predicted, X=None):
        """Returns a numerical score indicating performance based on the differences between the predicted and actual values.

        Arguments:
            y_predicted (pd.Series): Predicted values of length [n_samples]
            y_true (pd.Series): Actual class labels of length [n_samples]
            X (pd.DataFrame or np.ndarray): Extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            score
        """
        y_true = self._standardize_input_type(y_true)
        y_predicted = self._standardize_input_type(y_predicted)
        self.validate_inputs(y_true, y_predicted)
        return self.objective_function(y_true, y_predicted, X=X)

    @staticmethod
    def _standardize_input_type(y_in):
        """Standardize np or pd input to np for scoring

        Arguments:
            y_in (np.ndarray or pd.Series): A matrix of predictions or predicted probabilities

        Returns:
            np.ndarray: a 1d np array, or a 2d np array if predicted probabilities were provided.
        """
        if isinstance(y_in, (pd.Series, pd.DataFrame)):
            return y_in.to_numpy()
        if isinstance(y_in, ww.DataColumn):
            return y_in.to_series().to_numpy()
        if isinstance(y_in, ww.DataTable):
            return y_in.to_dataframe().to_numpy()
        return y_in

    def validate_inputs(self, y_true, y_predicted):
        """Validates the input based on a few simple checks.

        Arguments:
            y_predicted (pd.Series): Predicted values of length [n_samples]
            y_true (pd.Series): Actual class labels of length [n_samples]

        Returns:
            None
        """
        if isinstance(y_true, ww.DataColumn):
            y_true = y_true.to_series()
        if isinstance(y_predicted, ww.DataColumn):
            y_predicted = y_predicted.to_series()
        if y_predicted.shape[0] != y_true.shape[0]:
            raise ValueError("Inputs have mismatched dimensions: y_predicted has shape {}, y_true has shape {}".format(len(y_predicted), len(y_true)))
        if len(y_true) == 0:
            raise ValueError("Length of inputs is 0")
        if np.isnan(y_true).any() or np.isinf(y_true).any():
            raise ValueError("y_true contains NaN or infinity")
        if np.isnan(y_predicted).any() or np.isinf(y_predicted).any():
            raise ValueError("y_predicted contains NaN or infinity")
        if self.score_needs_proba and np.any([(y_predicted < 0) | (y_predicted > 1)]):
            raise ValueError("y_predicted contains probability estimates not within [0, 1]")

    @classmethod
    def calculate_percent_difference(cls, score, baseline_score):
        """Calculate the percent difference between scores.

        Arguments:
            score (float): A score. Output of the score method of this objective.
            baseline_score (float): A score. Output of the score method of this objective. In practice,
                this is the score achieved on this objective with a baseline estimator.

        Returns:
            float: The percent difference between the scores. This will be the difference normalized by the
                baseline score.
        """

        if pd.isna(score) or pd.isna(baseline_score):
            return np.nan

        if baseline_score == 0:
            return np.nan

        if baseline_score == score:
            return 0

        decrease = False
        if (baseline_score > score and cls.greater_is_better) or (baseline_score < score and not cls.greater_is_better):
            decrease = True

        difference = (baseline_score - score)
        change = difference / baseline_score
        return 100 * (-1) ** (decrease) * np.abs(change)

    @classmethod
    def is_defined_for_problem_type(cls, problem_type):
        return problem_type in cls.problem_types
