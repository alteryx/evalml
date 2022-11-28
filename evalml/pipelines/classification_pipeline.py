"""Pipeline subclass for all classification pipelines."""
import numpy as np
import pandas as pd
import woodwork as ww

from evalml.pipelines import PipelineBase
from evalml.problem_types import is_binary, is_multiclass
from evalml.utils import infer_feature_types


class ClassificationPipeline(PipelineBase):
    """Pipeline subclass for all classification pipelines.

    Args:
        component_graph (list or dict): List of components in order. Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"]
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters. Defaults to None.
        custom_name (str): Custom name for the pipeline. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def __init__(
        self,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        self._classes_ = None
        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )
        try:
            self._encoder = self.component_graph.get_component("Label Encoder")
        except ValueError:
            self._encoder = None

    def fit(self, X, y):
        """Build a classification model. For string and categorical targets, classes are sorted by sorted(set(y)) and then are mapped to values between 0 and n_classes-1.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): The target training labels of length [n_samples]

        Returns:
            self

        Raises:
            ValueError: If the number of unique classes in y are not appropriate for the type of pipeline.
            TypeError: If the dtype is boolean but pd.NA exists in the series.
            Exception: For all other exceptions.
        """
        X = infer_feature_types(X)
        y = infer_feature_types(y)

        if is_binary(self.problem_type) and y.nunique() != 2:
            raise ValueError("Binary pipelines require y to have 2 unique classes!")
        elif is_multiclass(self.problem_type) and y.nunique() in [1, 2]:
            raise ValueError(
                "Multiclass pipelines require y to have 3 or more unique classes!",
            )

        self._fit(X, y)

        # TODO: Added this in because numpy's unique() does not support pandas.NA
        try:
            self._classes_ = list(ww.init_series(np.unique(y)))
        except TypeError as e:
            if "boolean value of NA is ambiguous" in str(e):
                self._classes_ = y.unique()

        return self

    def _encode_targets(self, y):
        """Converts target values from their original values to integer values that can be processed."""
        if self._encoder is not None:
            try:
                return pd.Series(
                    self._encoder.transform(None, y)[1],
                    index=y.index,
                    name=y.name,
                ).astype(int)
            except ValueError as e:
                raise ValueError(str(e))
        return y

    @property
    def classes_(self):
        """Gets the class names for the pipeline. Will return None before pipeline is fit."""
        return self._classes_

    def _predict(self, X, objective=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            objective (Object or string): The objective to use to make predictions.

        Returns:
            pd.Series: Estimated labels
        """
        return self.component_graph.predict(X)

    def predict(self, X, objective=None, X_train=None, y_train=None):
        """Make predictions using selected features.

        Note: we cast y as ints first to address boolean values that may be returned from
        calculating predictions which we would not be able to otherwise transform if we
        originally had integer targets.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            objective (Object or string): The objective to use to make predictions.
            X_train (pd.DataFrame): Training data. Ignored. Only used for time series.
            y_train (pd.Series): Training labels. Ignored. Only used for time series.

        Returns:
            pd.Series: Estimated labels.
        """
        _predictions = self._predict(X, objective=objective)
        predictions = self.inverse_transform(_predictions.astype(int))
        predictions = pd.Series(
            predictions,
            name=self.input_target_name,
            index=_predictions.index,
        )
        return infer_feature_types(predictions)

    def predict_proba(self, X, X_train=None, y_train=None):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            X_train (pd.DataFrame or np.ndarray or None): Training data. Ignored. Only used for time series.
            y_train (pd.Series or None): Training labels. Ignored. Only used for time series.

        Returns:
            pd.DataFrame: Probability estimates

        Raises:
            ValueError: If final component is not an estimator.
        """
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict_proba() on a component graph because the final component is not an Estimator.",
            )
        X = self.transform_all_but_final(X, y=None)
        proba = self.estimator.predict_proba(X)
        proba = proba.ww.rename(
            columns={
                col: new_col for col, new_col in zip(proba.columns, self.classes_)
            },
        )
        return infer_feature_types(proba)

    def score(self, X, y, objectives, X_train=None, y_train=None):
        """Evaluate model performance on objectives.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            y (pd.Series): True labels of length [n_samples]
            objectives (list): List of objectives to score
            X_train (pd.DataFrame): Training data. Ignored. Only used for time series.
            y_train (pd.Series): Training labels. Ignored. Only used for time series.

        Returns:
            dict: Ordered dictionary of objective scores.
        """
        y = infer_feature_types(y)
        objectives = self.create_objectives(objectives)
        if self._encoder is not None:
            y = self._encode_targets(y)
        y_predicted, y_predicted_proba = self._compute_predictions(X, y, objectives)
        return self._score_all_objectives(
            X,
            y,
            y_predicted,
            y_predicted_proba,
            objectives,
        )

    def _compute_predictions(self, X, y, objectives):
        """Compute predictions/probabilities based on objectives."""
        y_predicted = None
        y_predicted_proba = None
        if any(o.score_needs_proba for o in objectives):
            y_predicted_proba = self.predict_proba(X)
        if any(not o.score_needs_proba for o in objectives):
            y_predicted = self._predict(X)
        return y_predicted, y_predicted_proba
