import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    _get_rows_without_nans,
    drop_rows_with_nans,
    pad_with_nans
)


class TimeSeriesClassificationPipeline(ClassificationPipeline):
    """Pipeline base class for time series classifcation problems."""

    def __init__(self, parameters, random_state=0):
        """Machine learning pipeline for time series classification problems made out of transformers and a classifier.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase subclasses in the list

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters. Pipeline-level
                 parameters such as gap and max_delay must be specified with the "pipeline" key. For example:
                 Pipeline(parameters={"pipeline": {"max_delay": 4, "gap": 2}}).
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        if "pipeline" not in parameters:
            raise ValueError("gap and max_delay parameters cannot be omitted from the parameters dict. "
                             "Please specify them as a dictionary with the key 'pipeline'.")
        pipeline_params = parameters["pipeline"]
        self.gap = pipeline_params['gap']
        self.max_delay = pipeline_params['max_delay']
        super().__init__(parameters, random_state)

    @staticmethod
    def _convert_to_woodwork(X, y):
        if X is None:
            X = pd.DataFrame()
        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
        return X, y

    def fit(self, X, y):
        """Fit a time series classification pipeline.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training targets of length [n_samples]

        Returns:
            self
        """
        X, y = self._convert_to_woodwork(X, y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        self._encoder.fit(y)
        y = self._encode_targets(y)

        X_t = self._compute_features_during_fit(X, y)
        if X_t.empty:
            raise RuntimeError("Pipeline computed empty features during call to .fit. This means "
                               "that either 1) you passed in X=None to fit and don't have a DelayFeatureTransformer "
                               "in your pipeline or 2) you do have a DelayFeatureTransformer but gap=0 and max_delay=0. "
                               "Please add a DelayFeatureTransformer or change the values of gap and max_delay")

        y_shifted = y.shift(-self.gap)
        X_t, y_shifted = drop_rows_with_nans(X_t, y_shifted)
        self.estimator.fit(X_t, y_shifted)
        return self

    def _predict(self, X, y, objective=None, pad=False):
        y_encoded = self._encode_targets(y)
        features = self.compute_estimator_features(X, y_encoded)
        predictions = self.estimator.predict(features.dropna(axis=0, how="any"))
        if pad:
            return pad_with_nans(predictions, max(0, features.shape[0] - predictions.shape[0]))
        return predictions

    def predict(self, X, y=None, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray, None): The target training targets of length [n_samples]
            objective (Object or string): The objective to use to make predictions

        Returns:
            pd.Series: Predicted values.
        """
        X, y = self._convert_to_woodwork(X, y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        n_features = max(len(y), X.shape[0])
        predictions = self._predict(X, y, objective=objective, pad=False)
        predictions = pd.Series(self._decode_targets(predictions))
        return pad_with_nans(predictions, max(0, n_features - predictions.shape[0]))

    def predict_proba(self, X, y=None):
        """Make probability estimates for labels.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame: Probability estimates
        """
        X, y = self._convert_to_woodwork(X, y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        y_encoded = self._encode_targets(y)
        features = self.compute_estimator_features(X, y_encoded)
        proba = self.estimator.predict_proba(features.dropna(axis=0, how="any"))
        proba.columns = self._encoder.classes_
        return pad_with_nans(proba, max(0, features.shape[0] - proba.shape[0]))

    def _compute_predictions(self, X, y, objectives):
        """Compute predictions/probabilities based on objectives."""
        y_predicted = None
        y_predicted_proba = None
        if any(o.score_needs_proba for o in objectives):
            y_predicted_proba = self.predict_proba(X, y)
        if any(not o.score_needs_proba for o in objectives):
            y_predicted = self._predict(X, y, pad=True)
        return y_predicted, y_predicted_proba

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (pd.Series, ww.DataColumn): True labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """
        X, y = self._convert_to_woodwork(X, y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        objectives = [get_objective(o, return_instance=True) for o in objectives]

        y_encoded = self._encode_targets(y)
        y_shifted = y_encoded.shift(-self.gap)
        y_pred, y_pred_proba = self._compute_predictions(X, y, objectives)
        non_nan_mask = _get_rows_without_nans(y_shifted, y_pred, y_pred_proba)
        if y_pred is not None:
            y_pred = y_pred.iloc[non_nan_mask]
        if y_pred_proba is not None:
            y_pred_proba = y_pred_proba.iloc[non_nan_mask]
        y_labels = y_shifted.iloc[non_nan_mask]
        return self._score_all_objectives(X, y_labels, y_pred,
                                          y_pred_proba=y_pred_proba,
                                          objectives=objectives)


class TimeSeriesBinaryClassificationPipeline(TimeSeriesClassificationPipeline):
    problem_type = ProblemTypes.TIME_SERIES_BINARY
    _threshold = None

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def _predict(self, X, y, objective=None, pad=False):
        y_encoded = self._encode_targets(y)
        features = self.compute_estimator_features(X, y_encoded)
        features_no_nan = features.dropna(axis=0, how="any")

        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if not objective.is_defined_for_problem_type(self.problem_type):
                raise ValueError(f"Objective {objective.name} is not defined for time series binary classification.")

        if self.threshold is None:
            predictions = self.estimator.predict(features_no_nan)
        else:
            ypred_proba = self.estimator.predict_proba(features_no_nan).iloc[:, 1]
            if objective is None:
                predictions = ypred_proba > self.threshold
            else:
                predictions = objective.decision_function(ypred_proba, threshold=self.threshold, X=features_no_nan)
        if pad:
            return pad_with_nans(predictions, max(0, features.shape[0] - predictions.shape[0]))
        return predictions

    @staticmethod
    def _score(X, y, predictions, objective):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score.
        """
        if predictions.ndim > 1:
            predictions = predictions.iloc[:, 1]
        return TimeSeriesClassificationPipeline._score(X, y, predictions, objective)


class TimeSeriesMulticlassClassificationPipeline(TimeSeriesClassificationPipeline):
    problem_type = ProblemTypes.TIME_SERIES_MULTICLASS
