import pandas as pd

from evalml.objectives import get_objective
from evalml.pipelines.classification_pipeline import ClassificationPipeline
from evalml.pipelines.binary_classification_pipeline import BinaryClassificationPipeline
from evalml.pipelines.multiclass_classification_pipeline import MulticlassClassificationPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    drop_rows_with_nans,
    pad_with_nans
)


class TimeSeriesClassificationPipeline(ClassificationPipeline):
    """Pipeline base class for time series classifcation problems."""

    def __init__(self, parameters, random_state=0):
        """Machine learning pipeline for time series regression problems made out of transformers and a classifier.

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

    def fit(self, X, y):
        """Fit a time series regression pipeline.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training targets of length [n_samples]

        Returns:
            self
        """
        if X is None:
            X = pd.DataFrame()

        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
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

    def _predict(self, X, y):
        features = self.compute_estimator_features(X, y).dropna(axis=0, how="any")
        predictions = self.estimator.predict(features)
        return pad_with_nans(predictions, max(0, features.shape[0] - predictions.shape[0]))

    def predict(self, X, y=None, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray, None): The target training targets of length [n_samples]
            objective (Object or string): The objective to use to make predictions

        Returns:
            pd.Series: Predicted values.
        """
        if X is None:
            X = pd.DataFrame()
        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        return self._predict(X, y)

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives.

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (pd.Series, ww.DataColumn): True labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """
        # Only converting X for the call to _score_all_objectives
        if X is None:
            X = pd.DataFrame()
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())

        y_predicted = self.predict(X, y)
        y_shifted = y.shift(-self.gap)
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        y_shifted, y_predicted = drop_rows_with_nans(y_shifted, y_predicted)
        return self._score_all_objectives(X, y_shifted,
                                          y_predicted,
                                          y_pred_proba=None,
                                          objectives=objectives)


class TimeSeriesBinaryClassificationPipeline(TimeSeriesClassificationPipeline):
    problem_type = ProblemTypes.TimeSeriesBinaryClassification
    _threshold = None

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def _predict(self, X, y, objective):
        X_t = self.compute_estimator_features(X, y).dropna(axis=0, how="any")

        if objective is not None:
            objective = get_objective(objective, return_instance=True)
            if not any(p in objective.problem_types for p in [ProblemTypes.BINARY, ProblemTypes.TIME_SERIES_BINARY]):
                raise ValueError(
                    "You can only use a binary classification objective to make predictions for a binary classification pipeline.")


        if self.threshold is None:
            predictions = self.estimator.predict(X_t)
        else:
            ypred_proba = self.estimator.predict_proba(X_t)
            ypred_proba = ypred_proba.iloc[:, 1]
            if objective is None:
                return ypred_proba > self.threshold
            predictions = objective.decision_function(ypred_proba, threshold=self.threshold, X=X_t)
        return pad_with_nans(predictions, max(0, X_t.shape[0] - predictions.shape[0]))


class TimeSeriesMulticlassClassificationPipeline(TimeSeriesClassificationPipeline):
    problem_type = ProblemTypes.TimeSeriesMulticlassClassification