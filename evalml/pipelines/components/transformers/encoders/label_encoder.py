"""A transformer that encodes target labels using values between 0 and num_classes - 1."""
import woodwork as ww

from evalml.pipelines.components.transformers.transformer import Transformer
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

        super().__init__(
            parameters=parameters,
            component_obj=None,
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
        y_ww = infer_feature_types(y)
        self.mapping = {val: i for i, val in enumerate(sorted(y_ww.unique()))}
        if self.parameters["positive_label"] is not None:
            if len(self.mapping) != 2:
                raise ValueError(
                    "positive_label should only be set for binary classification targets. Otherwise, positive_label should be None.",
                )
            if self.parameters["positive_label"] not in self.mapping:
                raise ValueError(
                    f"positive_label was set to `{self.parameters['positive_label']}` but was not found in the input target data.",
                )
            self.mapping = {
                val: int(val == self.parameters["positive_label"])
                for val in self.mapping
            }
        self.inverse_mapping = {i: val for val, i in self.mapping.items()}
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
        y_unique_values = set(y_ww.unique())
        if y_unique_values.difference(self.mapping.keys()):
            raise ValueError(
                f"y contains previously unseen labels: {y_unique_values.difference(self.mapping.keys())}",
            )
        y_t = y_ww.map(self.mapping)
        return X, ww.init_series(y_t, logical_type="integer")

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
        y_it = infer_feature_types(y_ww.map(self.inverse_mapping))
        return y_it
