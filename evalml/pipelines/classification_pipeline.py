"""Pipeline subclass for all classification pipelines."""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evalml.pipelines import PipelineBase
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
        self._encoder = LabelEncoder()
        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )

    def fit(self, X, y):
        """Build a classification model. For string and categorical targets, classes are sorted by sorted(set(y)) and then are mapped to values between 0 and n_classes-1.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): The target training labels of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        self._encoder.fit(y)
        y = self._encode_targets(y)
        self._fit(X, y)
        return self

    def _encode_targets(self, y):
        """Converts target values from their original values to integer values that can be processed."""
        try:
            return pd.Series(self._encoder.transform(y), index=y.index, name=y.name)
        except ValueError as e:
            raise ValueError(str(e))

    def _decode_targets(self, y):
        """Converts encoded numerical values to their original target values.

        Note: we cast y as ints first to address boolean values that may be returned from
        calculating predictions which we would not be able to otherwise transform if we
        originally had integer targets.
        """
        return self._encoder.inverse_transform(y.astype(int))

    @property
    def classes_(self):
        """Gets the class names for the problem."""
        if not hasattr(self._encoder, "classes_"):
            raise AttributeError(
                "Cannot access class names before fitting the pipeline."
            )
        return self._encoder.classes_

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

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            objective (Object or string): The objective to use to make predictions.
            X_train (pd.DataFrame): Training data. Ignored. Only used for time series.
            y_train (pd.Series): Training labels. Ignored. Only used for time series.

        Returns:
            pd.Series: Estimated labels.
        """
        predictions = self._predict(X, objective=objective)
        predictions = pd.Series(
            self._decode_targets(predictions), name=self.input_target_name
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
                "Cannot call predict_proba() on a component graph because the final component is not an Estimator."
            )
        X = self.compute_estimator_features(X, y=None)
        proba = self.estimator.predict_proba(X)
        proba = proba.ww.rename(
            columns={
                col: new_col
                for col, new_col in zip(proba.columns, self._encoder.classes_)
            }
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
        y = self._encode_targets(y)
        y_predicted, y_predicted_proba = self._compute_predictions(X, y, objectives)
        return self._score_all_objectives(
            X, y, y_predicted, y_predicted_proba, objectives
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
