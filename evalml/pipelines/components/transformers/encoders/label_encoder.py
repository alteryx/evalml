"""A transformer that encodes target labels using values between 0 and num_classes - 1."""
import pandas as pd
import woodwork as ww
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from ..transformer import Transformer

from evalml.utils import infer_feature_types


class LabelEncoder(Transformer):
    """A transformer that encodes target labels using values between 0 and num_classes - 1.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0. Ignored.
    """

    name = "Label Encoder"
    hyperparameter_ranges = {}
    """{}"""

    modifies_features = False
    modifies_target = True

    def __init__(self, random_seed=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)

        label_encoder_obj = SKLabelEncoder()
        super().__init__(
            parameters=parameters,
            component_obj=label_encoder_obj,
            random_seed=random_seed,
        )

    def fit(self, X, y):
        """Fits the label encoder.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]. Ignored.
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If input `y` is None.
        """
        if y is None:
            raise ValueError("y cannot be None!")
        self._component_obj.fit(y)
        return self

    def transform(self, X, y=None):
        """Transform the target using the fitted label encoder.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]. Ignored.
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            pd.DataFrame, pd.Series: The original features and an encoded version of the target.

        Raises:
            ValueError: If input `y` is None.
        """
        if y is None:
            raise ValueError("y cannot be None!")

        y_ww = infer_feature_types(y)
        y_t = self._component_obj.transform(y_ww)
        y_t = pd.Series(y_t, index=y_ww.index)
        return X, ww.init_series(y_t)

    def fit_transform(self, X, y):
        """Fit and transform data using the label encoder.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            pd.DataFrame, pd.Series: The original features and an encoded version of the target.
        """
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y):
        """Decodes the target data.

        Args:
            y (pd.Series): Target data.

        Returns:
            pd.Series: The decoded version of the target.

        Raises:
            ValueError: If input `y` is None.
        """
        if y is None:
            raise ValueError("y cannot be None!")

        y_it = self._component_obj.inverse_transform(y)
        return ww.init_series(y_it)
