import pandas as pd
import woodwork as ww

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

    def predict_proba_in_sample(self, X_holdout, y_holdout, X_train, y_train):
        y_holdout = self._encode_targets(y_holdout)
        y_train = self._encode_targets(y_train)
        features, target = self._compute_holdout_features_and_target(X_holdout, y_holdout, X_train, y_train)
        proba = self._estimator_predict_proba(features, target)
        proba.index = y_holdout.index
        proba = proba.ww.rename(
            columns={
                col: new_col
                for col, new_col in zip(proba.columns, self._encoder.classes_)
            }
        )
        return infer_feature_types(proba)

    def predict_in_sample(self, X, y, X_train, y_train, objective=None):
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict_in_sample() on a component graph because the final component is not an Estimator."
            )

        y = self._encode_targets(y)
        y_train = self._encode_targets(y_train)
        features, target = self._compute_holdout_features_and_target(X, y, X_train, y_train)
        predictions = self._estimator_predict(features, target)
        predictions.index = y.index
        predictions = self.inverse_transform(predictions)
        predictions = pd.Series(
            self._decode_targets(predictions), name=self.input_target_name, index=y.index
        )
        return infer_feature_types(predictions)

    def predict_proba(self, X, X_train=None, y_train=None):
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
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        y_holdout = self._create_empty_series(y_train)
        X, y_holdout = self._convert_to_woodwork(X, y_holdout)
        y_holdout.index = X.index
        return self.predict_proba_in_sample(X, y_holdout, X_train, y_train)

    def _compute_predictions(self, X, y, X_train, y_train, objectives):
        y_predicted = None
        y_predicted_proba = None
        if any(o.score_needs_proba for o in objectives):
            y_predicted_proba = self.predict_proba_in_sample(X, y, X_train, y_train)
        if any(not o.score_needs_proba for o in objectives):
            y_predicted = self.predict_in_sample(X, y, X_train, y_train)
            y_predicted = self._encode_targets(y_predicted)
        return y_predicted, y_predicted_proba

    def score(self, X, y, objectives, X_train=None, y_train=None):
        """Evaluate model performance on current and additional objectives.

        Arguments:
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series): True labels of length [n_samples].
            objectives (list): Non-empty list of objectives to score on.

        Returns:
            dict: Ordered dictionary of objective scores.
        """
        X, y = self._convert_to_woodwork(X, y)
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        objectives = self.create_objectives(objectives)
        y_predicted, y_predicted_proba = self._compute_predictions(
            X, y, X_train, y_train, objectives,
        )
        return self._score_all_objectives(
            X,
            self._encode_targets(y),
            y_predicted,
            y_pred_proba=y_predicted_proba,
            objectives=objectives,
        )


class TimeSeriesBinaryClassificationPipeline(
    TimeSeriesClassificationPipeline,
    BinaryClassificationPipelineMixin,
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

    def predict_in_sample(self, X, y, X_train, y_train, objective=None):

        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if not objective.is_defined_for_problem_type(self.problem_type):
                raise ValueError(
                    f"Objective {objective.name} is not defined for time series binary classification."
                )

        predictions = super().predict_in_sample(X, y, X_train, y_train)

        if self.threshold is not None:
            proba = self.predict_proba_in_sample(X, y, X_train, y_train)
            proba = proba.iloc[:, 1]
            if objective is None:
                predictions = proba > self.threshold
            else:
                predictions = objective.decision_function(
                    proba, threshold=self.threshold, X=X
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
