"""A transformer that encodes categorical features into target encodings."""
import pandas as pd

from ..transformer import Transformer

from evalml.pipelines.components.transformers.encoders.onehot_encoder import (
    OneHotEncoderMeta,
)
from evalml.utils import import_or_raise, infer_feature_types


class TargetEncoder(Transformer, metaclass=OneHotEncoderMeta):
    """A transformer that encodes categorical features into target encodings.

    Args:
        cols (list): Columns to encode. If None, all string columns will be encoded, otherwise only the columns provided will be encoded.
            Defaults to None
        smoothing (float): The smoothing factor to apply. The larger this value is, the more influence the expected target value has
            on the resulting target encodings. Must be strictly larger than 0. Defaults to 1.0
        handle_unknown (string): Determines how to handle unknown categories for a feature encountered. Options are 'value', 'error', nd 'return_nan'.
            Defaults to 'value', which replaces with the target mean
        handle_missing (string): Determines how to handle missing values encountered during `fit` or `transform`. Options are 'value', 'error', and 'return_nan'.
            Defaults to 'value', which replaces with the target mean
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Target Encoder"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        cols=None,
        smoothing=1.0,
        handle_unknown="value",
        handle_missing="value",
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "cols": cols,
            "smoothing": smoothing,
            "handle_unknown": handle_unknown,
            "handle_missing": handle_missing,
        }
        parameters.update(kwargs)

        unknown_and_missing_input_options = ["error", "return_nan", "value"]
        if handle_unknown not in unknown_and_missing_input_options:
            raise ValueError(
                "Invalid input '{}' for handle_unknown".format(handle_unknown)
            )
        if handle_missing not in unknown_and_missing_input_options:
            raise ValueError(
                "Invalid input '{}' for handle_missing".format(handle_missing)
            )
        if smoothing <= 0:
            raise ValueError(
                "Smoothing value needs to be strictly larger than 0. {} provided".format(
                    smoothing
                )
            )

        category_encode = import_or_raise(
            "category_encoders",
            error_msg="category_encoders not installed. Please install using `pip install category_encoders`",
        )
        super().__init__(
            parameters=parameters,
            component_obj=category_encode.target_encoder.TargetEncoder(**parameters),
            random_seed=random_seed,
        )

    def fit(self, X, y):
        """Fits the target encoder.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self
        """
        return super().fit(X, y)

    def transform(self, X, y=None):
        """Transform data using the fitted target encoder.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        X_ww = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        X_t = self._component_obj.transform(X, y)

        X_t_df = pd.DataFrame(X_t, columns=X_ww.columns, index=X_ww.index)
        no_cat_schema = X_ww.ww.select(exclude="category", return_schema=True)
        X_t_df.ww.init(schema=no_cat_schema)
        return X_t_df

    def fit_transform(self, X, y):
        """Fit and transform data using the target encoder.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        return self.fit(X, y).transform(X, y)

    def get_feature_names(self):
        """Return feature names for the input features after fitting.

        Returns:
            np.array: The feature names after encoding.
        """
        return self._component_obj.get_feature_names()

    def _get_feature_provenance(self):
        return {col: col for col in self.get_feature_names()}
