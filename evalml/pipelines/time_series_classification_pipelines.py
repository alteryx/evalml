"""Pipeline base class for time-series classification problems."""
import pandas as pd
from woodwork.statistics_utils import infer_frequency

from evalml.objectives import get_objective
from evalml.pipelines.binary_classification_pipeline_mixin import (
    BinaryClassificationPipelineMixin,
)
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.pipelines.time_series_pipeline_base import TimeSeriesPipelineBase
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


class TimeSeriesClassificationPipeline(TimeSeriesPipelineBase, ClassificationPipeline):
    """Pipeline base class for time series classification problems.

    Args:
        component_graph (ComponentGraph, list, dict): ComponentGraph instance, list of components in order, or dictionary of components.
            Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as time_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"time_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def _estimator_predict_proba(self, features):
        """Get estimator predicted probabilities.

        This helper passes y as an argument if needed by the estimator.
        """
        return self.estimator.predict_proba(features)

    def fit(self, X, y):
        """Fit a time series classification model.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): The target training labels of length [n_samples]

        Returns:
            self

        Raises:
            ValueError: If the number of unique classes in y are not appropriate for the type of pipeline.
        """
        self.frequency = infer_frequency(X[self.time_index])
        X, y = self._drop_time_index(X, y)
        return super().fit(X, y)

    def predict_proba_in_sample(self, X_holdout, y_holdout, X_train, y_train):
        """Predict on future data where the target is known, e.g. cross validation.

        Args:
            X_holdout (pd.DataFrame or np.ndarray): Future data of shape [n_samples, n_features].
            y_holdout (pd.Series, np.ndarray): Future target of shape [n_samples].
            X_train (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_features].
            y_train (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train].

        Returns:
            pd.Series: Estimated probabilities.

        Raises:
            ValueError: If the final component is not an Estimator.
        """
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict_proba_in_sample() on a component graph because the final component is not an Estimator.",
            )
        features = self.transform_all_but_final(X_holdout, y_holdout, X_train, y_train)
        proba = self._estimator_predict_proba(features)
        proba.index = y_holdout.index
        proba = proba.ww.rename(
            columns={
                col: new_col for col, new_col in zip(proba.columns, self.classes_)
            },
        )
        return infer_feature_types(proba)

    def predict_in_sample(self, X, y, X_train, y_train, objective=None):
        """Predict on future data where the target is known, e.g. cross validation.

        Note: we cast y as ints first to address boolean values that may be returned from
        calculating predictions which we would not be able to otherwise transform if we
        originally had integer targets.

        Args:
            X (pd.DataFrame or np.ndarray): Future data of shape [n_samples, n_features].
            y (pd.Series, np.ndarray): Future target of shape [n_samples].
            X_train (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_features].
            y_train (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train].
            objective (ObjectiveBase, str, None): Objective used to threshold predicted probabilities, optional.

        Returns:
            pd.Series: Estimated labels.

        Raises:
            ValueError: If final component is not an Estimator.
        """
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict_in_sample() on a component graph because the final component is not an Estimator.",
            )
        features = self.transform_all_but_final(X, y, X_train, y_train)
        predictions = self._estimator_predict(features)
        predictions.index = y.index
        predictions = self.inverse_transform(predictions.astype(int))
        predictions = pd.Series(predictions, name=self.input_target_name)

        predictions = predictions.rename(index=dict(zip(predictions.index, y.index)))
        return infer_feature_types(predictions)

    def predict_proba(self, X, X_train=None, y_train=None):
        """Predict on future data where the target is unknown.

        Args:
            X (pd.DataFrame or np.ndarray): Future data of shape [n_samples, n_features].
            X_train (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_features].
            y_train (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train].

        Returns:
            pd.Series: Estimated probabilities.

        Raises:
            ValueError: If final component is not an Estimator.
        """
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict_proba() on a component graph because the final component is not an Estimator.",
            )
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        X = infer_feature_types(X)
        X.index = self._move_index_forward(
            X_train.index[-X.shape[0] :],
            self.gap + X.shape[0],
        )
        y_holdout = self._create_empty_series(y_train, X.shape[0])
        y_holdout = infer_feature_types(y_holdout)
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

        Args:
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series): True labels of length [n_samples].
            objectives (list): Non-empty list of objectives to score on.
            X_train (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_features].
            y_train (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train].

        Returns:
            dict: Ordered dictionary of objective scores.
        """
        X, y = self._convert_to_woodwork(X, y)
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        X, y = self._drop_time_index(X, y)
        X_train, y_train = self._drop_time_index(X_train, y_train)
        objectives = self.create_objectives(objectives)
        y_predicted, y_predicted_proba = self._compute_predictions(
            X,
            y,
            X_train,
            y_train,
            objectives,
        )
        if self._encoder is not None:
            y = self._encode_targets(y)
        return self._score_all_objectives(
            X,
            y,
            y_predicted,
            y_pred_proba=y_predicted_proba,
            objectives=objectives,
        )


class TimeSeriesBinaryClassificationPipeline(
    TimeSeriesClassificationPipeline,
    BinaryClassificationPipelineMixin,
):
    """Pipeline base class for time series binary classification problems.

    Args:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as time_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"time_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> pipeline = TimeSeriesBinaryClassificationPipeline(component_graph=["Simple Imputer", "Logistic Regression Classifier"],
        ...                                                   parameters={"Logistic Regression Classifier": {"penalty": "elasticnet",
        ...                                                                                                  "solver": "liblinear"},
        ...                                                               "pipeline": {"gap": 1, "max_delay": 1, "forecast_horizon": 1, "time_index": "date"}},
        ...                                                   custom_name="My TimeSeriesBinary Pipeline")
        ...
        >>> assert pipeline.custom_name == "My TimeSeriesBinary Pipeline"
        >>> assert pipeline.component_graph.component_dict.keys() == {'Simple Imputer', 'Logistic Regression Classifier'}
        ...
        >>> assert pipeline.parameters == {
        ...     'Simple Imputer': {'impute_strategy': 'most_frequent', 'fill_value': None},
        ...     'Logistic Regression Classifier': {'penalty': 'elasticnet',
        ...                                         'C': 1.0,
        ...                                         'n_jobs': -1,
        ...                                         'multi_class': 'auto',
        ...                                         'solver': 'liblinear'},
        ...     'pipeline': {'gap': 1, 'max_delay': 1, 'forecast_horizon': 1, 'time_index': "date"}}
    """

    problem_type = ProblemTypes.TIME_SERIES_BINARY

    def _select_y_pred_for_score(self, X, y, y_pred, y_pred_proba, objective):
        y_pred_to_use = y_pred
        if self.threshold is not None and not objective.score_needs_proba:
            y_pred_to_use = self._predict_with_objective(X, y_pred_proba, objective)
        return y_pred_to_use

    def predict_in_sample(self, X, y, X_train, y_train, objective=None):
        """Predict on future data where the target is known, e.g. cross validation.

        Args:
            X (pd.DataFrame): Future data of shape [n_samples, n_features].
            y (pd.Series): Future target of shape [n_samples].
            X_train (pd.DataFrame): Data the pipeline was trained on of shape [n_samples_train, n_feautures].
            y_train (pd.Series): Targets used to train the pipeline of shape [n_samples_train].
            objective (ObjectiveBase, str): Objective used to threshold predicted probabilities, optional. Defaults to None.

        Returns:
            pd.Series: Estimated labels.

        Raises:
            ValueError: If objective is not defined for time-series binary classification problems.
        """
        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if not objective.is_defined_for_problem_type(self.problem_type):
                raise ValueError(
                    f"Objective {objective.name} is not defined for time series binary classification.",
                )

        if self.threshold is not None:
            proba = self.predict_proba_in_sample(X, y, X_train, y_train)
            proba = proba.iloc[:, 1]
            if objective is None:
                predictions = proba > self.threshold
                predictions = predictions.astype(int)
            else:
                predictions = objective.decision_function(
                    proba,
                    threshold=self.threshold,
                    X=X,
                )
            predictions = pd.Series(
                predictions,
                name=self.input_target_name,
                index=y.index,
            )
        else:
            predictions = super().predict_in_sample(X, y, X_train, y_train)

        return infer_feature_types(predictions)

    @staticmethod
    def _score(X, y, predictions, objective):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score."""
        if predictions.ndim > 1:
            predictions = predictions.iloc[:, 1]
        return TimeSeriesClassificationPipeline._score(X, y, predictions, objective)


class TimeSeriesMulticlassClassificationPipeline(TimeSeriesClassificationPipeline):
    """Pipeline base class for time series multiclass classification problems.

    Args:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary {} implies using all default values for component parameters. Pipeline-level
             parameters such as time_index, gap, and max_delay must be specified with the "pipeline" key. For example:
             Pipeline(parameters={"pipeline": {"time_index": "Date", "max_delay": 4, "gap": 2}}).
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> pipeline = TimeSeriesMulticlassClassificationPipeline(component_graph=["Simple Imputer", "Logistic Regression Classifier"],
        ...                                                       parameters={"Logistic Regression Classifier": {"penalty": "elasticnet",
        ...                                                                                                      "solver": "liblinear"},
        ...                                                                   "pipeline": {"gap": 1, "max_delay": 1, "forecast_horizon": 1, "time_index": "date"}},
        ...                                                       custom_name="My TimeSeriesMulticlass Pipeline")
        >>> assert pipeline.custom_name == "My TimeSeriesMulticlass Pipeline"
        >>> assert pipeline.component_graph.component_dict.keys() == {'Simple Imputer', 'Logistic Regression Classifier'}
        >>> assert pipeline.parameters == {
        ...  'Simple Imputer': {'impute_strategy': 'most_frequent', 'fill_value': None},
        ...  'Logistic Regression Classifier': {'penalty': 'elasticnet',
        ...                                     'C': 1.0,
        ...                                     'n_jobs': -1,
        ...                                     'multi_class': 'auto',
        ...                                     'solver': 'liblinear'},
        ...     'pipeline': {'gap': 1, 'max_delay': 1, 'forecast_horizon': 1, 'time_index': "date"}}
    """

    problem_type = ProblemTypes.TIME_SERIES_MULTICLASS
    """ProblemTypes.TIME_SERIES_MULTICLASS"""
