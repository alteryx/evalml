"""A transformer that encodes target labels using values between 0 and num_classes - 1."""
import pandas as pd
import woodwork as ww
from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from ..transformer import Transformer

from evalml.utils import infer_feature_types


class LabelEncoder(Transformer):
    """A transformer that encodes target labels using values between 0 and num_classes - 1.

    Args:
        positive_label (int, str): The label for the class that should be treated as positive (1) for binary classification problems. Ignored for multiclass problems. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0. Ignored.
    """

    name = "Label Encoder"
    hyperparameter_ranges = {}
    """{}"""

    modifies_features = False
    modifies_target = True

    def __init__(self, positive_label=None, random_seed=0, **kwargs):
        parameters = {"positive_label": positive_label}
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
        if self.parameters["positive_label"] is None:
            self._component_obj.fit(y)
        else:
            classes_ = set(pd.Series(y).unique())
            if len(classes_) != 2:
                raise ValueError(
                    "positive_label should only be set for binary classification targets. Otherwise, positive_label should be None."
                )
            try:
                classes_.remove(self.parameters["positive_label"])
            except KeyError:
                raise ValueError(
                    f"positive_label was set to `{self.parameters['positive_label']}` but was not found in the input target data."
                )

            negative_label = classes_.pop()
            self.mapping = {negative_label: 0, self.parameters["positive_label"]: 1}
            self.inverse_mapping = {
                0: negative_label,
                1: self.parameters["positive_label"],
            }
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
            return X, y
        y_ww = infer_feature_types(y)
        if self.parameters["positive_label"] is None:
            y_t = self._component_obj.transform(y_ww)
            y_t = pd.Series(y_t, index=y_ww.index)
            return X, ww.init_series(y_t)
        else:
            y_unique_values = set(pd.Series(y).unique())
            if y_unique_values != self.mapping.keys():
                raise ValueError(
                    f"y contains previously unseen labels: {y_unique_values.difference(self.mapping.keys())}"
                )
            y_t = y.map(self.mapping)
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
        y_ww = infer_feature_types(y)
        if self.parameters["positive_label"] is None:

            y_it = self._component_obj.inverse_transform(y)
            y_it = infer_feature_types(pd.Series(y_it, index=y_ww.index))
        else:
            y_it = infer_feature_types(y_ww.map(self.inverse_mapping))
        return y_it
