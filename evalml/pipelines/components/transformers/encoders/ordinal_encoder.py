"""A transformer that encodes categorical features in a one-hot numeric array."""
from pdb import set_trace

import numpy as np
import pandas as pd
import woodwork as ww
from sklearn.preprocessing import OrdinalEncoder as SKOrdinalEncoder
from woodwork.logical_types import Ordinal

from evalml.pipelines.components import ComponentBaseMeta
from evalml.pipelines.components.transformers.transformer import Transformer
from evalml.utils import infer_feature_types

"""A transformer that encodes categorical features in a one-hot numeric array."""


class OrdinalEncoderMeta(ComponentBaseMeta):
    METHODS_TO_CHECK = ComponentBaseMeta.METHODS_TO_CHECK + [
        "get_feature_names",
    ]


class OrdinalEncoder(Transformer, metaclass=OrdinalEncoderMeta):
    name = "Ordinal Encoder"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        top_n=10,
        features_to_encode=None,
        categories=None,
        handle_unknown="error",
        unknown_value=None,
        encoded_missing_value=np.nan,  # --> maybe this should be np.nan since that's the utils ddefault
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "top_n": top_n,
            "features_to_encode": features_to_encode,
            "categories": categories,  # --> the cols must have their categories set - so maybe don't need this set?
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
        if top_n is not None and categories is not None:
            raise ValueError("Cannot use categories and top_n arguments simultaneously")

        # --> add a check for encoded_missing_values is int or npnan? What about unknown value?
        self.features_to_encode = features_to_encode
        self._encoder = None

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
        top_n = self.parameters["top_n"]
        # --> we don't ever infer as Ordinal if theyre not set before this wont get used
        X = infer_feature_types(X)
        if self.features_to_encode is None:
            # --> should update to not include ordinals? Maybe that's configurable based on whether ordinal encoder is used?
            self.features_to_encode = self._get_ordinal_cols(X)

        X_t = X
        invalid_features = [
            col for col in self.features_to_encode if col not in list(X.columns)
        ]
        if len(invalid_features) > 0:
            # --> what if features to encode includes non ordinal cols?
            raise ValueError(
                "Could not find and encode {} in input data.".format(
                    ", ".join(invalid_features),
                ),
            )

        # helper util to handle unknown ? Probs not needed bc I think the encoder can do wha twe need
        # --> handle categories logic - includes topn - which means we do need to do value counts when theres more than n values
        if len(self.features_to_encode) == 0:
            categories = "auto"
        elif self.parameters["categories"] is not None:
            categories = self.parameters["categories"]
            if len(categories) != len(self.features_to_encode) or not isinstance(
                categories[0],
                list,
            ):
                raise ValueError(
                    "Categories argument must contain a list of categories for each categorical feature",
                )
            # --> should we compare with the ordinal categories to make sure they're all at least in there?
            # --> if so, add a test
        else:
            categories = []
            ww_logical_types = X.ww.logical_types
            for col in X_t[self.features_to_encode]:
                ltype = ww_logical_types[col]
                assert isinstance(ltype, Ordinal)
                # --> if this is sampled data, the order might not be accurate?
                column_categories = ltype.order

                if top_n is None or len(column_categories) <= top_n:
                    unique_values = column_categories
                else:
                    value_counts = X_t[col].value_counts(dropna=False).to_frame()
                    # --> is it worth comparing to the column's order? maybe not
                    value_counts = value_counts.sample(
                        frac=1,
                        random_state=self._initial_state,
                    )
                    value_counts = value_counts.sort_values(
                        [col],
                        ascending=False,
                        kind="mergesort",
                    )
                    unique_values = value_counts.head(top_n).index.tolist()
                unique_values = np.sort(unique_values)
                categories.append(unique_values)

        self._encoder = SKOrdinalEncoder(
            categories=categories,
            handle_unknown=self.parameters["handle_unknown"],
            unknown_value=self.parameters["unknown_value"],
            encoded_missing_value=self.parameters["encoded_missing_value"],
        )

        self._encoder.fit(X_t[self.features_to_encode])
        # --> logic to set up input parameters?
        return self

    def transform(self, X, y=None):
        X = infer_feature_types(X)

        X_copy = X.ww.copy()
        X = X.ww.drop(columns=self.features_to_encode)

        # Call sklearn's transform on the ordinal columns
        if len(self.features_to_encode) > 0:
            X_cat = pd.DataFrame(
                self._encoder.transform(X_copy[self.features_to_encode]),
                index=X_copy.index,
            )
            X_cat.columns = self._get_feature_names()
            # --> could we do Integer or IntegerNullably? Maybe but Double is simpler
            X_cat.ww.init(logical_types={c: "Double" for c in X_cat.columns})
            self._feature_names = X_cat.columns

            X = ww.utils.concat_columns([X, X_cat])

        return X

    def _get_feature_names(self):
        """Return feature names for the ordinal features after fitting.

        Since ordinal encoding creates one encoded feature per column in features_to_encode, feature
        names are formatted as {column_name}_ordinally_encoded --> choose a better name?? maybe one that includes how many categories were encoded

        Returns:
            np.ndarray: The feature names after encoding, provided in the same order as input_features.
        """
        self._features_to_drop = []
        unique_names = []
        provenance = {}
        for col_name in self.features_to_encode:
            encoded_name = f"{col_name}_ordinally_encoded"
            unique_names.append(encoded_name)
            provenance[col_name] = [encoded_name]
        # --> make sure provenance should point to a list even with only one element
        self._provenance = provenance
        return unique_names

    def categories(self, feature_name):
        # --> need to make sure this works
        """Returns a list of the unique categories to be encoded for the particular feature, in order.

        Args:
            feature_name (str): The name of any feature provided to one-hot encoder during fit.

        Returns:
            np.ndarray: The unique categories, in the same dtype as they were provided during fit.

        Raises:
            ValueError: If feature was not provided to one-hot encoder as a training feature.
        """
        try:
            index = self.features_to_encode.index(feature_name)
        except Exception:
            raise ValueError(
                f'Feature "{feature_name}" was not provided to one-hot encoder as a training feature',
            )
        return self._encoder.categories_[index]

    def get_feature_names(self):
        return self._get_feature_names()

    def _get_feature_provenance(self):
        return self._provenance
