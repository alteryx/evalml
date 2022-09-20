"""A component that may or may not need fitting that transforms data. These components are used before an estimator."""
from abc import abstractmethod

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components import ComponentBase
from evalml.utils import infer_feature_types


class Transformer(ComponentBase):
    """A component that may or may not need fitting that transforms data. These components are used before an estimator.

    To implement a new Transformer, define your own class which is a subclass of Transformer, including
    a name and a list of acceptable ranges for any parameters to be tuned during the automl search (hyperparameters).
    Define an `__init__` method which sets up any necessary state and objects. Make sure your `__init__` only
    uses standard keyword arguments and calls `super().__init__()` with a parameters dict. You may also override the
    `fit`, `transform`, `fit_transform` and other methods in this class if appropriate.

    To see some examples, check out the definitions of any Transformer component.

    Args:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    modifies_features = True
    modifies_target = False
    training_only = False

    def __init__(self, parameters=None, component_obj=None, random_seed=0, **kwargs):
        super().__init__(
            parameters=parameters,
            component_obj=component_obj,
            random_seed=random_seed,
            **kwargs,
        )

    @abstractmethod
    def transform(self, X, y=None):
        """Transforms data X.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Target data.

        Returns:
            pd.DataFrame: Transformed X

        Raises:
            MethodPropertyNotFoundError: If transformer does not have a transform method or a component_obj that implements transform.
        """

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X.

        Args:
            X (pd.DataFrame): Data to fit and transform.
            y (pd.Series): Target data.

        Returns:
            pd.DataFrame: Transformed X.

        Raises:
            MethodPropertyNotFoundError: If transformer does not have a transform method or a component_obj that implements transform.
        """
        X_ww = infer_feature_types(X)
        if y is not None:
            y_ww = infer_feature_types(y)
        else:
            y_ww = y

        try:
            return self.fit(X_ww, y_ww).transform(X_ww, y_ww)
        except MethodPropertyNotFoundError as e:
            raise e

    def _get_feature_provenance(self):
        return {}
