"""Pipeline base class for time-series problems."""
import pandas as pd
import woodwork as ww

from evalml.pipelines import PipelineBase
from evalml.pipelines.pipeline_meta import PipelineBaseMeta
from evalml.utils import infer_feature_types
from evalml.utils.gen_utils import are_datasets_separated_by_gap_time_index


class TimeSeriesPipelineBase(PipelineBase, metaclass=PipelineBaseMeta):
    """Pipeline base class for time series problems.

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

    def __init__(
        self,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        if not parameters or "pipeline" not in parameters:
            raise ValueError(
                "time_index, gap, max_delay, and forecast_horizon parameters cannot be omitted from the parameters dict. "
                "Please specify them as a dictionary with the key 'pipeline'."
            )
        self.pipeline_params = parameters["pipeline"]
        self.gap = self.pipeline_params["gap"]
        self.max_delay = self.pipeline_params["max_delay"]
        self.forecast_horizon = self.pipeline_params["forecast_horizon"]
        self.time_index = self.pipeline_params["time_index"]
        if self.time_index is None:
            raise ValueError("Parameter time_index cannot be None!")
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

    @staticmethod
    def _move_index_forward(index, gap):
        """Fill in the index of the gap features and values with the right values."""
        if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
            return index.shift(gap)
        else:
            return index + gap

    def _add_training_data_to_X_Y(self, X, y, X_train, y_train):
        """Append the training data to the holdout data.

        Need to do this so that we have all the data we need to compute lagged features on the holdout set.
        """
        last_row_of_training = self.forecast_horizon + self.max_delay + self.gap
        gap_features = pd.DataFrame()
        gap_target = pd.Series()
        if (
            are_datasets_separated_by_gap_time_index(X_train, X, self.pipeline_params)
            and self.gap
        ):
            # The training data does not have the gap dates so don't need to include them
            last_row_of_training -= self.gap

            # Instead, we'll create some dummy data to represent the missing gap dates
            # These do not show up in the features used for prediction
            gap_features = X_train.iloc[[-1] * self.gap]
            gap_features.index = self._move_index_forward(
                X_train.index[-self.gap :], self.gap
            )
            gap_target = y_train.iloc[[-1] * self.gap]
            gap_target.index = self._move_index_forward(
                y_train.index[-self.gap :], self.gap
            )

        features_to_concat = [
            X_train.iloc[-last_row_of_training:],
            gap_features,
            X,
        ]
        targets_to_concat = [
            y_train.iloc[-last_row_of_training:],
            gap_target,
            y,
        ]
        padded_features = pd.concat(features_to_concat, axis=0).fillna(method="ffill")
        padded_target = pd.concat(targets_to_concat, axis=0).fillna(method="ffill")
        padded_features.ww.init(schema=X_train.ww.schema)
        padded_target = ww.init_series(
            padded_target, logical_type=y_train.ww.logical_type
        )
        return padded_features, padded_target

    def transform_all_but_final(self, X, y=None, X_train=None, y_train=None):
        """Transforms the data by applying all pre-processing components.

        Args:
            X (pd.DataFrame): Input data to the pipeline to transform.
            y (pd.Series): Targets corresponding to the pipeline targets.
            X_train (pd.DataFrame): Training data used to generate generates from past observations.
            y_train (pd.Series): Training targets used to generate features from past observations.

        Returns:
            pd.DataFrame: New transformed features.
        """
        if y_train is None:
            y_train = pd.Series()
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        X, y = self._convert_to_woodwork(X, y)

        empty_training_data = X_train.empty or y_train.empty
        if empty_training_data:
            features_holdout = super().transform_all_but_final(X, y)
        else:
            padded_features, padded_target = self._add_training_data_to_X_Y(
                X, y, X_train, y_train
            )
            features = super().transform_all_but_final(padded_features, padded_target)
            features_holdout = features.iloc[-len(y) :]
        return features_holdout

    def predict_in_sample(self, X, y, X_train, y_train, objective=None):
        """Predict on future data where the target is known, e.g. cross validation.

        Args:
            X (pd.DataFrame or np.ndarray): Future data of shape [n_samples, n_features]
            y (pd.Series, np.ndarray): Future target of shape [n_samples]
            X_train (pd.DataFrame, np.ndarray): Data the pipeline was trained on of shape [n_samples_train, n_feautures]
            y_train (pd.Series, np.ndarray): Targets used to train the pipeline of shape [n_samples_train]
            objective (ObjectiveBase, str, None): Objective used to threshold predicted probabilities, optional.

        Returns:
            pd.Series: Estimated labels.

        Raises:
            ValueError: If final component is not an Estimator.
        """
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict_in_sample() on a component graph because the final component is not an Estimator."
            )
        target = infer_feature_types(y)
        features = self.transform_all_but_final(X, target, X_train, y_train)
        predictions = self._estimator_predict(features)
        predictions.index = y.index
        predictions = self.inverse_transform(predictions)
        predictions = predictions.rename(self.input_target_name)
        return infer_feature_types(predictions)

    def _create_empty_series(self, y_train, size):
        return ww.init_series(
            pd.Series([y_train.iloc[0]] * size),
            logical_type=y_train.ww.logical_type,
        )

    def predict(self, X, objective=None, X_train=None, y_train=None):
        """Predict on future data where target is not known.

        Args:
            X (pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features].
            objective (Object or string): The objective to use to make predictions.
            X_train (pd.DataFrame or np.ndarray or None): Training data.
            y_train (pd.Series or None): Training labels.

        Raises:
            ValueError: If final component is not an Estimator.

        Returns:
            Predictions.
        """
        X_train, y_train = self._convert_to_woodwork(X_train, y_train)
        if self.estimator is None:
            raise ValueError(
                "Cannot call predict() on a component graph because the final component is not an Estimator."
            )
        X = infer_feature_types(X)
        X.index = self._move_index_forward(
            X_train.index[-X.shape[0] :], self.gap + X.shape[0]
        )
        y_holdout = self._create_empty_series(y_train, X.shape[0])
        y_holdout = infer_feature_types(y_holdout)
        y_holdout.index = X.index
        return self.predict_in_sample(
            X, y_holdout, X_train, y_train, objective=objective
        )

    def _estimator_predict(self, features):
        """Get estimator predictions.

        This helper passes y as an argument if needed by the estimator.
        """
        return self.estimator.predict(features)
