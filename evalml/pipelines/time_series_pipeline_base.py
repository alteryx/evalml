import pandas as pd
import woodwork as ww

from evalml.pipelines import PipelineBase
from evalml.pipelines.pipeline_meta import PipelineBaseMeta
from evalml.utils import drop_rows_with_nans, infer_feature_types


class TimeSeriesPipelineBase(PipelineBase, metaclass=PipelineBaseMeta):

    """Pipeline base class for time series problems.

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

    def __init__(
        self,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        if not parameters or "pipeline" not in parameters:
            raise ValueError(
                "date_index, gap, and max_delay parameters cannot be omitted from the parameters dict. "
                "Please specify them as a dictionary with the key 'pipeline'."
            )
        pipeline_params = parameters["pipeline"]
        self.date_index = pipeline_params["date_index"]
        self.gap = pipeline_params["gap"]
        self.max_delay = pipeline_params["max_delay"]
        self.forecast_horizon = pipeline_params["forecast_horizon"]
        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )

    @staticmethod
    def _convert_to_woodwork(X, y):
        if X is None:
            X = pd.DataFrame()
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        return X, y

    def fit(self, X, y):
        """Fit a time series pipeline.

        Arguments:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features].
            y (pd.Series, np.ndarray): The target training targets of length [n_samples].

        Returns:
            self
        """
        X, y = self._convert_to_woodwork(X, y)
        self._fit(X, y)
        return self

    @staticmethod
    def _move_index_forward(index, gap):
        if gap == 0:
            return index
        elif isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
            return index.shift(gap)
        else:
            return index + gap

    def _compute_holdout_features_and_target(
        self, X_holdout, y_holdout, X_train, y_train
    ):
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        X_holdout, y_holdout = self._convert_to_woodwork(X_holdout, y_holdout)
        last_row_of_training_needed_for_features = (
            self.forecast_horizon + self.max_delay
        )
        gap_features = pd.DataFrame()
        gap_target = pd.Series()
        if self.gap:
            gap_features = X_train.iloc[[-1] * self.gap]
            gap_features.index = self._move_index_forward(
                X_train.index[-self.gap :], self.gap
            )
            gap_target = y_train.iloc[[-1] * self.gap]
            gap_target.index = self._move_index_forward(
                y_train.index[-self.gap :], self.gap
            )
        padded_features = pd.concat(
            [
                X_train.iloc[-last_row_of_training_needed_for_features:],
                gap_features,
                X_holdout,
            ],
            axis=0,
        )
        padded_target = pd.concat(
            [
                y_train.iloc[-last_row_of_training_needed_for_features:],
                gap_target,
                y_holdout,
            ],
            axis=0,
        )
        padded_features.ww.init(schema=X_train.ww.schema)
        padded_target = ww.init_series(
            padded_target, logical_type=y_train.ww.logical_type
        )
        features = self.compute_estimator_features(padded_features, padded_target)
        features_holdout = features.iloc[-len(y_holdout) :]
        return features_holdout, y_holdout

    def predict_in_sample(self, X, y, X_train, y_train, objective=None):
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict_in_sample() on a component graph because the final component is not an Estimator."
            )
        features, target = self._compute_holdout_features_and_target(
            X, y, X_train, y_train
        )
        predictions = self._estimator_predict(features, target)
        predictions.index = y.index
        predictions = self.inverse_transform(predictions)
        predictions = predictions.rename(self.input_target_name)
        return infer_feature_types(predictions)

    def _create_empty_series(self, y_train):
        return ww.init_series(
            pd.Series([y_train.iloc[0]] * self.forecast_horizon),
            logical_type=y_train.ww.logical_type,
        )

    def predict(self, X, objective=None, X_train=None, y_train=None):
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict() on a component graph because the final component is not an Estimator."
            )
        y_holdout = self._create_empty_series(y_train)
        X, y_holdout = self._convert_to_woodwork(X, y_holdout)
        y_holdout.index = X.index
        return self.predict_in_sample(
            X, y_holdout, X_train, y_train, objective=objective
        )

    def _fit(self, X, y):
        self.input_target_name = y.name
        X_t = self.component_graph.fit_features(X, y)
        X_t, y_shifted = drop_rows_with_nans(X_t, y)

        if self.estimator is not None:
            self.estimator.fit(X_t, y_shifted)
        else:
            self.component_graph.get_last_component().fit(X_t, y)

        self.input_feature_names = self.component_graph.input_feature_names

    def _estimator_predict(self, features, y):
        """Get estimator predictions.

        This helper passes y as an argument if needed by the estimator.
        """
        y_arg = None
        if self.estimator.predict_uses_y:
            y_arg = y
        return self.estimator.predict(features, y=y_arg)
