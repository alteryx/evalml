from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import woodwork as ww

from evalml.problem_types import handle_problem_types
from evalml.utils import _convert_woodwork_types_wrapper, classproperty


class ObjectiveBase(ABC):
    """Base class for all objectives."""

    problem_types = None

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

    @property
    @classmethod
    @abstractmethod
    def is_bounded_like_percentage(cls):
        """Returns whether this objective is bounded between 0 and 1, inclusive."""

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

    @classproperty
    def positive_only(cls):
        """If True, this objective is only valid for positive data. Default False."""
        return False

    def score(self, y_true, y_predicted, X=None):
        """Returns a numerical score indicating performance based on the differences between the predicted and actual values.

        Arguments:
            y_predicted (pd.Series): Predicted values of length [n_samples]
            y_true (pd.Series): Actual class labels of length [n_samples]
            X (pd.DataFrame or np.ndarray): Extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            score
        """
        if X is not None:
            X = self._standardize_input_type(X)
        y_true = self._standardize_input_type(y_true)
        y_predicted = self._standardize_input_type(y_predicted)
        self.validate_inputs(y_true, y_predicted)
        return self.objective_function(y_true, y_predicted, X=X)

    @staticmethod
    def _standardize_input_type(input_data):
        """Standardize input to pandas for scoring.

        Arguments:
            input_data (list, ww.DataTable, ww.DataColumn, pd.DataFrame, pd.Series, or np.ndarray): A matrix of predictions or predicted probabilities

        Returns:
            pd.DataFrame or pd.Series: a pd.Series, or pd.DataFrame object if predicted probabilities were provided.
        """
        if isinstance(input_data, (pd.Series, pd.DataFrame)):
            return _convert_woodwork_types_wrapper(input_data)
        if isinstance(input_data, ww.DataTable):
            return _convert_woodwork_types_wrapper(input_data.to_dataframe())
        if isinstance(input_data, ww.DataColumn):
            return _convert_woodwork_types_wrapper(input_data.to_series())
        if isinstance(input_data, list):
            if isinstance(input_data[0], list):
                return pd.DataFrame(input_data)
            return pd.Series(input_data)
        if isinstance(input_data, np.ndarray):
            if len(input_data.shape) == 1:
                return pd.Series(input_data)
            return pd.DataFrame(input_data)

    def validate_inputs(self, y_true, y_predicted):
        """Validates the input based on a few simple checks.

        Arguments:
            y_predicted (ww.DataColumn, ww.DataTable, pd.Series, or pd.DataFrame): Predicted values of length [n_samples]
            y_true (ww.DataColumn, pd.Series): Actual class labels of length [n_samples]

        Returns:
            None
        """
        if y_predicted.shape[0] != y_true.shape[0]:
            raise ValueError("Inputs have mismatched dimensions: y_predicted has shape {}, y_true has shape {}".format(len(y_predicted), len(y_true)))
        if len(y_true) == 0:
            raise ValueError("Length of inputs is 0")
        if np.isnan(y_true).any() or np.isinf(y_true).any():
            raise ValueError("y_true contains NaN or infinity")
        # y_predicted could be a 1d vector (predictions) or a 2d vector (classifier predicted probabilities)
        y_pred_flat = y_predicted.to_numpy().flatten()
        if np.isnan(y_pred_flat).any() or np.isinf(y_pred_flat).any():
            raise ValueError("y_predicted contains NaN or infinity")
        if self.score_needs_proba and np.any([(y_pred_flat < 0) | (y_pred_flat > 1)]):
            raise ValueError("y_predicted contains probability estimates not within [0, 1]")

    @classmethod
    def calculate_percent_difference(cls, score, baseline_score):
        """Calculate the percent difference between scores.

        Arguments:
            score (float): A score. Output of the score method of this objective.
            baseline_score (float): A score. Output of the score method of this objective. In practice,
                this is the score achieved on this objective with a baseline estimator.

        Returns:
            float: The percent difference between the scores. Note that for objectives that can be interpreted
                as percentages, this will be the difference between the reference score and score. For all other
                objectives, the difference will be normalized by the reference score.
        """

        if pd.isna(score) or pd.isna(baseline_score):
            return np.nan

        if np.isclose(baseline_score - score, 0, atol=1e-10):
            return 0

        # Return inf when dividing by 0
        if np.isclose(baseline_score, 0, atol=1e-10) and not cls.is_bounded_like_percentage:
            return np.inf

        decrease = False
        if (baseline_score > score and cls.greater_is_better) or (baseline_score < score and not cls.greater_is_better):
            decrease = True

        difference = (baseline_score - score)
        change = difference if cls.is_bounded_like_percentage else difference / baseline_score
        return 100 * (-1) ** (decrease) * np.abs(change)

    @classmethod
    def is_defined_for_problem_type(cls, problem_type):
        return handle_problem_types(problem_type) in cls.problem_types
