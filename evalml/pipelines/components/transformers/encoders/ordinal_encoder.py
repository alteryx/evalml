"""A transformer that encodes ordinal features as an array of ordinal integers representing the relative order of categories."""
import numpy as np
import pandas as pd
import woodwork as ww
from sklearn.preprocessing import OrdinalEncoder as SKOrdinalEncoder
from woodwork.logical_types import Ordinal

from evalml.pipelines.components import ComponentBaseMeta
from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types

"""A transformer that encodes ordinal features."""


class OrdinalEncoderMeta(ComponentBaseMeta):
    """A version of the ComponentBaseMeta class which includes validation on an additional ordinal-encoder-specific method `categories`."""

    METHODS_TO_CHECK = ComponentBaseMeta.METHODS_TO_CHECK + [
        "categories",
        "get_feature_names",
    ]


class OrdinalEncoder(Transformer, metaclass=OrdinalEncoderMeta):
    """A transformer that encodes ordinal features as an array of ordinal integers representing the relative order of categories.

    Args:
        features_to_encode (list[str]): List of columns to encode. All other columns will remain untouched.
            If None, all appropriate columns will be encoded. Defaults to None. The order of columns does not matter.
        categories (dict[str, list[str]]): A dictionary mapping column names to their categories
            in the dataframes passed in at fit and transform. The order of categories specified for a column does not matter.
            Any category found in the data that is not present in categories will be handled as an unknown value.
            To not have unknown values raise an error, set handle_unknown to "use_encoded_value".
            Defaults to None.
        handle_unknown ("error" or "use_encoded_value"): Whether to ignore or error for unknown categories
            for a feature encountered during `fit` or `transform`. When set to "error",
            an error will be raised when an unknown category is found.
            When set to "use_encoded_value", unknown categories will be encoded as the value given
            for the parameter unknown_value. Defaults to "error."
        unknown_value (int or np.nan): The value to use for unknown categories seen during fit or transform.
            Required when the parameter handle_unknown is set to "use_encoded_value."
            The value has to be distinct from the values used to encode any of the categories in fit.
            Defaults to None.
        encoded_missing_value (int or np.nan): The value to use for missing (null) values seen during
            fit or transform. Defaults to np.nan.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Ordinal Encoder"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        features_to_encode=None,
        categories=None,
        handle_unknown="error",
        unknown_value=None,
        encoded_missing_value=None,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "features_to_encode": features_to_encode,
            "categories": categories,
            "handle_unknown": handle_unknown,
            "unknown_value": unknown_value,
            "encoded_missing_value": encoded_missing_value,
        }
        parameters.update(kwargs)

        # Check correct inputs
        unknown_input_options = ["use_encoded_value", "error"]
        if handle_unknown not in unknown_input_options:
            raise ValueError(
                "Invalid input {} for handle_unknown".format(handle_unknown),
            )
        if handle_unknown == "use_encoded_value" and unknown_value is None:
            raise ValueError(
                "To use encoded value for unknown categories, unknown_value must"
                "be specified as either np.nan or as an int that is distinct from"
                "the other encoded categories ",
            )

        self.features_to_encode = features_to_encode
        self._component_obj = None

        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )
        self._initial_state = self.random_seed
        self._provenance = {}

    @staticmethod
    def _get_ordinal_cols(X):
        """Get names of ordinal columns in the input DataFrame."""
        return list(X.ww.select(include=["ordinal"], return_schema=True).columns)

    def fit(self, X, y=None):
        """Fits the ordinal encoder component.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If encoding a column failed.
            TypeError: If non-Ordinal columns are specified in features_to_encode.
        """
        # Ordinal type is not inferred by Woodwork, so if it wasn't set before, it won't be set at init
        X = infer_feature_types(X)
        if self.features_to_encode is None:
            self.features_to_encode = self._get_ordinal_cols(X)
        else:
            # When features_to_encode is user-specified, check that all columns are present
            # and have the Ordinal logical type
            not_present_features = [
                col for col in self.features_to_encode if col not in list(X.columns)
            ]
            if len(not_present_features) > 0:
                raise ValueError(
                    "Could not find and encode {} in input data.".format(
                        ", ".join(not_present_features),
                    ),
                )

            logical_types = X.ww.logical_types
            for col in self.features_to_encode:
                ltype = logical_types[col]
                if not isinstance(ltype, Ordinal):
                    raise TypeError(
                        f"Column {col} specified in features_to_encode is not Ordinal in nature",
                    )

        ww_logical_types = X.ww.logical_types
        categories = {}
        if len(self.features_to_encode) == 0:
            # No ordinal features present - no transformation can take place so return early
            return self
        elif self.parameters["categories"] is not None:
            # Categories specified - make sure they match the ordinal columns
            input_categories = self.parameters["categories"]

            if len(input_categories) != len(self.features_to_encode):
                raise ValueError(
                    "Categories argument must contain as many elements as there are features to encode.",
                )

            if not all(isinstance(cats, list) for cats in input_categories.values()):
                raise ValueError(
                    "Each of the values in the categories argument must be a list.",
                )
            # Categories, as they're passed into SKOrdinalEncoder should be in the same order
            # as the data's Ordinal.order categories even if it's a subset
            for col_name in self.features_to_encode:
                col_categories = input_categories[col_name]
                categories_order = ww_logical_types[col_name].order

                ordered_categories = [
                    cat for cat in categories_order if cat in col_categories
                ]
                categories[col_name] = ordered_categories
        else:
            # Categories unspecified - use ordered categories from a columns' Ordinal logical type
            for col_name in self.features_to_encode:
                ltype = ww_logical_types[col_name]
                # Copy the order list, since we might mutate it later by adding nans
                # and don't want to impact the Woodwork types
                categories[col_name] = ltype.order.copy()

        # Add any null values into the categories lists so that they aren't treated as unknown values
        # This is needed because Ordinal.order won't indicate if nulls are present, and SKOrdinalEncoder
        # requires any null values be present in the categories list if they are to be encoded as
        # missing values
        for col_name in self.features_to_encode:
            if X[col_name].isna().any():
                categories[col_name].append(np.nan)

        # sklearn needs categories to be a list in the order of the columns in features_to_encode
        categories_for_sk_encoder = [
            categories[col_name] for col_name in self.features_to_encode
        ]

        encoded_missing_value = self.parameters["encoded_missing_value"]
        if encoded_missing_value is None:
            encoded_missing_value = np.nan

        self._component_obj = SKOrdinalEncoder(
            categories=categories_for_sk_encoder,
            handle_unknown=self.parameters["handle_unknown"],
            unknown_value=self.parameters["unknown_value"],
            encoded_missing_value=encoded_missing_value,
        )

        self._component_obj.fit(X[self.features_to_encode])
        return self

    def transform(self, X, y=None):
        """Ordinally encode the input data.

        Args:
            X (pd.DataFrame): Features to encode.
            y (pd.Series): Ignored.

        Returns:
            pd.DataFrame: Transformed data, where each ordinal feature has been encoded into
            a numerical column where ordinal integers represent
            the relative order of categories.
        """
        X = infer_feature_types(X)

        if not self.features_to_encode:
            # If there are no features to encode, X needs no transformation
            return X

        X_orig = X.ww.drop(columns=self.features_to_encode)

        # Call sklearn's transform on only the ordinal columns
        X_t = pd.DataFrame(
            self._component_obj.transform(X[self.features_to_encode]),
            index=X.index,
        )
        X_t.columns = self._get_feature_names()
        X_t.ww.init(logical_types={c: "Double" for c in X_t.columns})
        self._feature_names = X_t.columns

        X_t = ww.utils.concat_columns([X_orig, X_t])

        return X_t

    def _get_feature_names(self):
        """Return feature names for the ordinal features after fitting.

        Since ordinal encoding creates one encoded feature per column in features_to_encode, feature
        names are formatted as {column_name}_ordinal_encoding

        Returns:
            np.ndarray: The feature names after encoding, provided in the same order as input_features.
        """
        self._features_to_drop = []
        unique_names = []
        provenance = {}
        for col_name in self.features_to_encode:
            encoded_name = f"{col_name}_ordinal_encoding"
            unique_names.append(encoded_name)
            provenance[col_name] = [encoded_name]
        self._provenance = provenance
        return unique_names

    def categories(self, feature_name):
        """Returns a list of the unique categories to be encoded for the particular feature, in order.

        Args:
            feature_name (str): The name of any feature provided to ordinal encoder during fit.

        Returns:
            np.ndarray: The unique categories, in the same dtype as they were provided during fit.

        Raises:
            ValueError: If feature was not provided to ordinal encoder as a training feature.
        """
        try:
            index = self.features_to_encode.index(feature_name)
        except Exception:
            raise ValueError(
                f'Feature "{feature_name}" was not provided to ordinal encoder as a training feature',
            )
        return self._component_obj.categories_[index]

    def get_feature_names(self):
        """Return feature names for the ordinal features after fitting.

        Feature names are formatted as {column name}_ordinal_encoding.

        Returns:
            np.ndarray: The feature names after encoding, provided in the same order as input_features.
        """
        return self._get_feature_names()

    def _get_feature_provenance(self):
        return self._provenance
