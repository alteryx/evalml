"""Component that imputes missing target data according to a specified imputation strategy."""
from functools import wraps

import pandas as pd
import woodwork as ww
from sklearn.impute import SimpleImputer as SkImputer

from evalml.exceptions import ComponentNotYetFittedError
from evalml.pipelines.components import ComponentBaseMeta
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class TargetImputerMeta(ComponentBaseMeta):
    """A version of the ComponentBaseMeta class which handles when input features is None."""

    @classmethod
    def check_for_fit(cls, method):
        """`check_for_fit` wraps a method that validates if `self._is_fitted` is `True`.

        Args:
            method (callable): Method to wrap.

        Raises:
            ComponentNotYetFittedError: If component is not fitted.

        Returns:
            The wrapped input method.
        """

        @wraps(method)
        def _check_for_fit(self, X=None, y=None):
            klass = type(self).__name__
            if not self._is_fitted and self.needs_fitting:
                raise ComponentNotYetFittedError(
                    f"This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}."
                )
            else:
                return method(self, X, y)

        return _check_for_fit


class TargetImputer(Transformer, metaclass=TargetImputerMeta):
    """Imputes missing target data according to a specified imputation strategy.

    Args:
        impute_strategy (string): Impute strategy to use. Valid values include "mean", "median", "most_frequent", "constant" for
           numerical data, and "most_frequent", "constant" for object data types. Defaults to "most_frequent".
        fill_value (string): When impute_strategy == "constant", fill_value is used to replace missing data.
           Defaults to None which uses 0 when imputing numerical data and "missing_value" for strings or object data types.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Target Imputer"
    hyperparameter_ranges = {"impute_strategy": ["mean", "median", "most_frequent"]}
    """{
        "impute_strategy": ["mean", "median", "most_frequent"]
    }"""
    modifies_features = False
    modifies_target = True

    def __init__(
        self, impute_strategy="most_frequent", fill_value=None, random_seed=0, **kwargs
    ):
        parameters = {"impute_strategy": impute_strategy, "fill_value": fill_value}
        parameters.update(kwargs)
        imputer = SkImputer(strategy=impute_strategy, fill_value=fill_value, **kwargs)
        super().__init__(
            parameters=parameters, component_obj=imputer, random_seed=random_seed
        )

    def fit(self, X, y):
        """Fits imputer to target data. 'None' values are converted to np.nan before imputation and are treated as the same.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]. Ignored.
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            TypeError: If target is filled with all null values.
        """
        if y is None:
            return self
        y = infer_feature_types(y)
        if all(y.isnull()):
            raise TypeError("Provided target full of nulls.")
        y = y.to_frame()

        # Convert all bool dtypes to category for fitting
        if (y.dtypes == bool).all():
            y = y.astype("category")

        self._component_obj.fit(y)
        return self

    def transform(self, X, y):
        """Transforms input target data by imputing missing values. 'None' and np.nan values are treated as the same.

        Args:
            X (pd.DataFrame): Features. Ignored.
            y (pd.Series): Target data to impute.

        Returns:
            (pd.DataFrame, pd.Series): The original X, transformed y
        """
        if X is not None:
            X = infer_feature_types(X)
        if y is None:
            return X, None
        y_ww = infer_feature_types(y)
        y_df = y_ww.ww.to_frame()

        # Return early since bool dtype doesn't support nans and sklearn errors if all cols are bool
        if (y_df.dtypes == bool).all():
            return X, y_ww

        transformed = self._component_obj.transform(y_df)
        y_t = pd.Series(transformed[:, 0], index=y_ww.index)
        y_t = ww.init_series(
            y_t, logical_type=y_ww.ww.logical_type, semantic_tags=y_ww.ww.semantic_tags
        )
        return X, y_t

    def fit_transform(self, X, y):
        """Fits on and transforms the input target data.

        Args:
            X (pd.DataFrame): Features. Ignored.
            y (pd.Series): Target data to impute.

        Returns:
            (pd.DataFrame, pd.Series): The original X, transformed y
        """
        return self.fit(X, y).transform(X, y)
