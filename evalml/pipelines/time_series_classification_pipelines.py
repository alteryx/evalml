import pandas as pd

from .binary_classification_pipeline_mixin import (
    BinaryClassificationPipelineMixin,
)

from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.pipelines.time_series_pipeline_base import TimeSeriesPipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils import (
    drop_rows_with_nans,
    infer_feature_types,
    pad_with_nans,
)


class TimeSeriesClassificationPipeline(TimeSeriesPipelineBase, ClassificationPipeline):
    """Pipeline base class for time series classification problems.

    Arguments:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as date_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"date_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def fit(self, X, y):
        """Fit a time series classification pipeline.

        Arguments:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): The target training targets of length [n_samples]

        Returns:
            self
        """
        X, y = self._convert_to_woodwork(X, y)
        self._encoder.fit(y)
        y = self._encode_targets(y)
        self._fit(X, y)
        return self

    def _estimator_predict_proba(self, features, y):
        """Get estimator predicted probabilities.

        This helper passes y as an argument if needed by the estimator.
        """
        y_arg = None
        if self.estimator.predict_uses_y:
            y_arg = y
        return self.estimator.predict_proba(features, y=y_arg)

    def _predict(self, X, y, objective=None, pad=False):
        features = self.compute_estimator_features(X, y)
        features_no_nan, y_no_nan = drop_rows_with_nans(features, y)
        predictions = self._estimator_predict(features_no_nan, y_no_nan)
        if pad:
            padded = pad_with_nans(
                predictions, max(0, features.shape[0] - predictions.shape[0])
            )
            return infer_feature_types(padded)
        return predictions

    def predict(self, X, y=None, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series, np.ndarray, None): The target training targets of length [n_samples].
            objective (Object or string): The objective to use to make predictions.

        Returns:
            pd.Series: Predicted values.
        """
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict() on a component graph because the final component is not an Estimator."
            )
        X, y = self._convert_to_woodwork(X, y)
        y = self._encode_targets(y)
        n_features = max(len(y), X.shape[0])
        predictions = self._predict(X, y, objective=objective, pad=False)
        # In case gap is 0 and this is a baseline pipeline, we drop the nans in the
        # predictions before decoding them
        predictions = pd.Series(
            self._decode_targets(predictions.dropna()), name=self.input_target_name
        )
        padded = pad_with_nans(predictions, max(0, n_features - predictions.shape[0]))
        return infer_feature_types(padded)

    def predict_proba(self, X, y=None):
        """Make probability estimates for labels.

        Arguments:
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features].

        Returns:
            pd.DataFrame: Probability estimates.
        """
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict_proba() on a component graph because the final component is not an Estimator."
            )
        X, y = self._convert_to_woodwork(X, y)
        y = self._encode_targets(y)
        features = self.compute_estimator_features(X, y)
        features_no_nan, y_no_nan = drop_rows_with_nans(features, y)
        proba = self._estimator_predict_proba(features_no_nan, y_no_nan)
        proba.columns = self._encoder.classes_
        padded = pad_with_nans(proba, max(0, features.shape[0] - proba.shape[0]))
        return infer_feature_types(padded)

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives.

        Arguments:
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series): True labels of length [n_samples].
            objectives (list): Non-empty list of objectives to score on.

        Returns:
            dict: Ordered dictionary of objective scores.
        """
        X, y = self._convert_to_woodwork(X, y)
        objectives = self.create_objectives(objectives)
        y_encoded = self._encode_targets(y)
        y_shifted = y_encoded.shift(-self.gap)
        y_predicted, y_predicted_proba = self._compute_predictions(
            X, y, objectives, time_series=True
        )
        y_shifted, y_predicted, y_predicted_proba = drop_rows_with_nans(
            y_shifted, y_predicted, y_predicted_proba
        )
        return self._score_all_objectives(
            X,
            y_shifted,
            y_predicted,
            y_pred_proba=y_predicted_proba,
            objectives=objectives,
        )


class TimeSeriesBinaryClassificationPipeline(
    BinaryClassificationPipelineMixin,
    TimeSeriesClassificationPipeline,
):
    """Pipeline base class for time series binary classification problems.

    Arguments:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as date_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"date_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    problem_type = ProblemTypes.TIME_SERIES_BINARY

    def _predict(self, X, y, objective=None, pad=False):
        features = self.compute_estimator_features(X, y)
        features_no_nan, y_no_nan = drop_rows_with_nans(features, y)

        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if not objective.is_defined_for_problem_type(self.problem_type):
                raise ValueError(
                    f"Objective {objective.name} is not defined for time series binary classification."
                )

        if self.threshold is None:
            predictions = self._estimator_predict(features_no_nan, y_no_nan)
        else:
            proba = self._estimator_predict_proba(features_no_nan, y_no_nan)
            proba = proba.iloc[:, 1]
            if objective is None:
                predictions = proba > self.threshold
            else:
                predictions = objective.decision_function(
                    proba, threshold=self.threshold, X=features_no_nan
                )
        if pad:
            predictions = pad_with_nans(
                predictions, max(0, features.shape[0] - predictions.shape[0])
            )
        return infer_feature_types(predictions)

    @staticmethod
    def _score(X, y, predictions, objective):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score."""
        if predictions.ndim > 1:
            predictions = predictions.iloc[:, 1]
        return TimeSeriesClassificationPipeline._score(X, y, predictions, objective)


class TimeSeriesMulticlassClassificationPipeline(TimeSeriesClassificationPipeline):
    """Pipeline base class for time series multiclass classification problems.

    Arguments:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as date_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"date_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    problem_type = ProblemTypes.TIME_SERIES_MULTICLASS
    """ProblemTypes.TIME_SERIES_MULTICLASS"""
