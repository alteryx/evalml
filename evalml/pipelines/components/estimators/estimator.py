"""A component that fits and predicts given data."""
from abc import abstractmethod

from pandas.core.indexes import range

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ComponentBase
from evalml.utils import infer_feature_types


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

    # We can't use the inspect module to dynamically determine this because of issue 1582
    predict_uses_y = False
    model_family = ModelFamily.NONE
    """ModelFamily.NONE"""

    modifies_features = True
    modifies_target = False

    @property
    @classmethod
    @abstractmethod
    def supported_problem_types(cls):
        """Problem types this estimator supports."""

    def __init__(self, parameters=None, component_obj=None, random_seed=0, **kwargs):
        self.input_feature_names = None
        super().__init__(
            parameters=parameters,
            component_obj=component_obj,
            random_seed=random_seed,
            **kwargs,
        )

    def _manage_woodwork(self, X, y=None):
        """Function to convert the input and target data to Pandas data structures."""
        if X is not None:
            X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        return X, y

    def fit(self, X, y=None):
        """Fits estimator to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self
        """
        X, y = self._manage_woodwork(X, y)
        self.input_feature_names = list(X.columns)
        self._component_obj.fit(X, y)
        return self

    def predict(self, X):
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
            if isinstance(X.columns, range.RangeIndex):
                X.columns = [x for x in X.columns]
            predictions = self._component_obj.predict(X)
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Estimator requires a predict method or a component_obj that implements predict"
            )
        return infer_feature_types(predictions)

    def predict_proba(self, X):
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
            pred_proba = self._component_obj.predict_proba(X)
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Estimator requires a predict_proba method or a component_obj that implements predict_proba"
            )
        return infer_feature_types(pred_proba)

    @property
    def feature_importance(self):
        """Returns importance associated with each feature.

        Returns:
            np.ndarray: Importance associated with each feature.

        Raises:
            MethodPropertyNotFoundError: If estimator does not have a feature_importance method or a component_obj that implements feature_importance.
        """
        try:
            return self._component_obj.feature_importances_
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Estimator requires a feature_importance property or a component_obj that implements feature_importances_"
            )

    def __eq__(self, other):
        """Check for equality."""
        return (
            super().__eq__(other)
            and self.supported_problem_types == other.supported_problem_types
        )
