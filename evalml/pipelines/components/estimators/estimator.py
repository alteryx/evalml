"""A component that fits and predicts given data."""
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Type, Union

import pandas as pd
import scipy.stats as st

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ComponentBase
from evalml.problem_types import ProblemTypes
from evalml.utils import infer_feature_types


def _handle_column_names_for_scikit(X):
    if any(isinstance(col, str) for col in X.columns) and not all(
        isinstance(col, str) for col in X.columns
    ):
        X.columns = X.columns.astype(str)
    return X


class Estimator(ComponentBase):
    """A component that fits and predicts given data.

    To implement a new Estimator, define your own class which is a subclass of Estimator, including
    a name and a list of acceptable ranges for any parameters to be tuned during the automl search (hyperparameters).
    Define an `__init__` method which sets up any necessary state and objects. Make sure your `__init__` only
    uses standard keyword arguments and calls `super().__init__()` with a parameters dict. You may also override the
    `fit`, `transform`, `fit_transform` and other methods in this class if appropriate.

    To see some examples, check out the definitions of any Estimator component subclass.

    Args:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    model_family = ModelFamily.NONE
    """ModelFamily.NONE"""

    modifies_features = True
    modifies_target = False
    training_only = False

    @property
    @classmethod
    @abstractmethod
    def model_family(cls):
        """Returns ModelFamily of this component."""

    @property
    @classmethod
    @abstractmethod
    def supported_problem_types(cls):
        """Problem types this estimator supports."""

    def __init__(
        self,
        parameters: dict = None,
        component_obj: Type[ComponentBase] = None,
        random_seed: Union[int, float] = 0,
        **kwargs,
    ):
        self.input_feature_names = None
        super().__init__(
            parameters=parameters,
            component_obj=component_obj,
            random_seed=random_seed,
            **kwargs,
        )

    def _manage_woodwork(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Function to convert the input and target data to Pandas data structures."""
        if X is not None:
            X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        return X, y

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits estimator to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self
        """
        X, y = self._manage_woodwork(X, y)
        X = _handle_column_names_for_scikit(X)
        self.input_feature_names = list(X.columns)
        self._component_obj.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.

        Raises:
            MethodPropertyNotFoundError: If estimator does not have a predict method or a component_obj that implements predict.
        """
        try:
            X = infer_feature_types(X)
            X = _handle_column_names_for_scikit(X)
            predictions = self._component_obj.predict(X)
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Estimator requires a predict method or a component_obj that implements predict",
            )
        predictions = infer_feature_types(predictions)
        predictions.index = X.index
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            pd.Series: Probability estimates.

        Raises:
            MethodPropertyNotFoundError: If estimator does not have a predict_proba method or a component_obj that implements predict_proba.
        """
        try:
            X = infer_feature_types(X)
            X = _handle_column_names_for_scikit(X)
            pred_proba = self._component_obj.predict_proba(X)
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Estimator requires a predict_proba method or a component_obj that implements predict_proba",
            )
        pred_proba = infer_feature_types(pred_proba)
        pred_proba.index = X.index
        return pred_proba

    def get_prediction_intervals(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        coverage: List[float] = None,
        predictions: pd.Series = None,
    ) -> Dict[str, pd.Series]:
        """Find the prediction intervals using the fitted regressor.

        This function takes the predictions of the fitted estimator and calculates the rolling standard deviation across
        all predictions using a window size of 5. The lower and upper predictions are determined by taking the percent
        point (quantile) function of the lower tail probability at each bound multiplied by the rolling standard deviation.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): Target data. Ignored.
            coverage (list[float]): A list of floats between the values 0 and 1 that the upper and lower bounds of the
                prediction interval should be calculated for.
            predictions (pd.Series): Optional list of predictions to use. If None, will generate predictions using `X`.

        Returns:
            dict: Prediction intervals, keys are in the format {coverage}_lower or {coverage}_upper.

        Raises:
            MethodPropertyNotFoundError: If the estimator does not support Time Series Regression as a problem type.
        """
        if ProblemTypes.TIME_SERIES_REGRESSION not in self.supported_problem_types:
            raise MethodPropertyNotFoundError(
                "Estimator must support Time Series Regression",
            )
        if coverage is None:
            coverage = [0.95]
        X, _ = self._manage_woodwork(X, y)
        if predictions is None:
            predictions = self._component_obj.predict(X)

        prediction_interval_result = {}
        for conf_int in coverage:
            rolling_std = pd.Series(predictions).rolling(5).std().bfill()
            preds_lower = (
                predictions + st.norm.ppf(round((1 - conf_int) / 2, 3)) * rolling_std
            )
            preds_upper = (
                predictions + st.norm.ppf(round((1 + conf_int) / 2, 3)) * rolling_std
            )

            preds_lower = pd.Series(preds_lower.values, index=X.index)
            preds_upper = pd.Series(preds_upper.values, index=X.index)
            prediction_interval_result[f"{conf_int}_lower"] = preds_lower
            prediction_interval_result[f"{conf_int}_upper"] = preds_upper

        return prediction_interval_result

    @property
    def feature_importance(self) -> pd.Series:
        """Returns importance associated with each feature.

        Returns:
            np.ndarray: Importance associated with each feature.

        Raises:
            MethodPropertyNotFoundError: If estimator does not have a feature_importance method or a component_obj that implements feature_importance.
        """
        try:
            return pd.Series(self._component_obj.feature_importances_)
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Estimator requires a feature_importance property or a component_obj that implements feature_importances_",
            )

    def __eq__(self, other):
        """Check for equality."""
        return (
            super().__eq__(other)
            and self.supported_problem_types == other.supported_problem_types
        )
