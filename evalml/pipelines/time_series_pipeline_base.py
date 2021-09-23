"""Pipeline base class for time-series problems."""
import pandas as pd
import woodwork as ww

from evalml.pipelines import PipelineBase
from evalml.pipelines.pipeline_meta import PipelineBaseMeta
from evalml.utils import drop_rows_with_nans, infer_feature_types


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

        Args:
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
        """Fill in the index of the gap features and values with the right values."""
        if isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
            return index.shift(gap)
        else:
            return index + gap

    @staticmethod
    def _are_datasets_separated_by_gap(train_index, test_index, gap):
        """Determine if the train and test datasets are separated by gap number of units.

        This will be true when users are predicting on unseen data but not during cross
        validation since the target is known.
        """
        gap_difference = gap + 1
        index_difference = test_index[0] - train_index[-1]
        if isinstance(
            train_index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)
        ):
            gap_difference *= test_index.freq
        return index_difference == gap_difference

    def _validate_holdout_datasets(self, X, X_train):
        """Validate the holdout datasets match out expectations.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            X_train (pd.DataFrame): Training data.

        Raises:
            ValueError: If holdout data does not have forecast_horizon entries or if datasets
                are not separated by gap.
        """
        right_length = len(X) == self.forecast_horizon
        X_separated_by_gap = self._are_datasets_separated_by_gap(
            X_train.index, X.index, self.gap
        )
        if not (right_length and X_separated_by_gap):
            raise ValueError(
                f"Holdout data X must have {self.forecast_horizon}  rows (value of forecast horizon) "
                "and its index needs to "
                f"start {self.gap + 1} values ahead of the training index. "
                f"Data received - Length X: {len(X)}, "
                f"X index start: {X.index[0]}, X_train index end {X.index[-1]}."
            )

    def _add_training_data_to_X_Y(self, X, y, X_train, y_train):
        """Append the training data to the holdout data.

        Need to do this so that we have all the data we need to compute lagged features on the holdout set.
        """
        last_row_of_training = self.forecast_horizon + self.max_delay + self.gap
        gap_features = pd.DataFrame()
        gap_target = pd.Series()
        if (
            self._are_datasets_separated_by_gap(X_train.index, X.index, self.gap)
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
        padded_features = pd.concat(features_to_concat, axis=0)
        padded_target = pd.concat(targets_to_concat, axis=0)

        padded_features.ww.init(schema=X_train.ww.schema)
        padded_target = ww.init_series(
            padded_target, logical_type=y_train.ww.logical_type
        )
        return padded_features, padded_target

    def compute_estimator_features(self, X, y=None, X_train=None, y_train=None):
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
            features_holdout = super().compute_estimator_features(X, y)
        else:
            padded_features, padded_target = self._add_training_data_to_X_Y(
                X, y, X_train, y_train
            )
            features = super().compute_estimator_features(
                padded_features, padded_target
            )
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
        features = self.compute_estimator_features(X, target, X_train, y_train)
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
        self._validate_holdout_datasets(X, X_train)
        y_holdout = self._create_empty_series(y_train)
        y_holdout = infer_feature_types(y_holdout)
        y_holdout.index = X.index
        return self.predict_in_sample(
            X, y_holdout, X_train, y_train, objective=objective
        )

    def _fit(self, X, y):
        self.input_target_name = y.name
        X_t, y_t = self.component_graph.fit_features(X, y)
        X_t, y_shifted = drop_rows_with_nans(X_t, y_t)

        if self.estimator is not None:
            self.estimator.fit(X_t, y_shifted)
        else:
            self.component_graph.get_last_component().fit(X_t, y_shifted)

        self.input_feature_names = self.component_graph.input_feature_names

    def _estimator_predict(self, features, y):
        """Get estimator predictions.

        This helper passes y as an argument if needed by the estimator.
        """
        y_arg = None
        if self.estimator.predict_uses_y:
            y_arg = y
        return self.estimator.predict(features, y=y_arg)
