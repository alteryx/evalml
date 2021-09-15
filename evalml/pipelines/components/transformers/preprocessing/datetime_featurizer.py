"""Transformer that can automatically extract features from datetime columns."""
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


def _extract_year(col, encode_as_categories=False):
    return col.dt.year, None


_month_to_int_mapping = {
    "January": 0,
    "February": 1,
    "March": 2,
    "April": 3,
    "May": 4,
    "June": 5,
    "July": 6,
    "August": 7,
    "September": 8,
    "October": 9,
    "November": 10,
    "December": 11,
}


def _extract_month(col, encode_as_categories=False):
    months = col.dt.month_name()
    months_unique = months.unique()
    months_encoded = months.map(lambda m: _month_to_int_mapping[m])
    if encode_as_categories:
        months_encoded = months_encoded.astype("category")
    return months_encoded, {m: _month_to_int_mapping[m] for m in months_unique}


_day_to_int_mapping = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
}


def _extract_day_of_week(col, encode_as_categories=False):
    days = col.dt.day_name()
    days_unique = days.unique()
    days_encoded = days.map(lambda d: _day_to_int_mapping[d])
    if encode_as_categories:
        days_encoded = days_encoded.astype("category")
    return days_encoded, {d: _day_to_int_mapping[d] for d in days_unique}


def _extract_hour(col, encode_as_categories=False):
    return col.dt.hour, None


class DateTimeFeaturizer(Transformer):
    """Transformer that can automatically extract features from datetime columns.

    Args:
        features_to_extract (list): List of features to extract. Valid options include "year", "month", "day_of_week", "hour". Defaults to None.
        encode_as_categories (bool): Whether day-of-week and month features should be encoded as pandas "category" dtype.
            This allows OneHotEncoders to encode these features. Defaults to False.
        date_index (str): Name of the column containing the datetime information used to order the data. Ignored.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "DateTime Featurization Component"
    hyperparameter_ranges = {}
    """{}"""
    _function_mappings = {
        "year": _extract_year,
        "month": _extract_month,
        "day_of_week": _extract_day_of_week,
        "hour": _extract_hour,
    }

    def __init__(
        self,
        features_to_extract=None,
        encode_as_categories=False,
        date_index=None,
        random_seed=0,
        **kwargs,
    ):
        if features_to_extract is None:
            features_to_extract = ["year", "month", "day_of_week", "hour"]
        invalid_features = set(features_to_extract) - set(
            self._function_mappings.keys()
        )
        if len(invalid_features) > 0:
            raise ValueError(
                "{} are not valid options for features_to_extract".format(
                    ", ".join([f"'{feature}'" for feature in invalid_features])
                )
            )

        parameters = {
            "features_to_extract": features_to_extract,
            "encode_as_categories": encode_as_categories,
            "date_index": date_index,
        }
        parameters.update(kwargs)
        self._date_time_col_names = None
        self._categories = {}
        self.encode_as_categories = encode_as_categories
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y=None):
        """Fit the datetime featurizer component.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series, optional): Target data. Ignored.

        Returns:
            self
        """
        X = infer_feature_types(X)
        self._date_time_col_names = list(
            X.ww.select("datetime", return_schema=True).columns
        )
        return self

    def transform(self, X, y=None):
        """Transforms data X by creating new features using existing DateTime columns, and then dropping those DateTime columns.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X = infer_feature_types(X)
        X = X.ww.copy()
        features_to_extract = self.parameters["features_to_extract"]
        if len(features_to_extract) == 0:
            return X
        for col_name in self._date_time_col_names:
            for feature in features_to_extract:
                name = f"{col_name}_{feature}"
                features, categories = self._function_mappings[feature](
                    X[col_name], self.encode_as_categories
                )
                X.ww[name] = features
                if categories:
                    self._categories[name] = categories
        X.ww.drop(columns=self._date_time_col_names, inplace=True)
        return X

    def get_feature_names(self):
        """Gets the categories of each datetime feature.

        Returns:
            dict: Dictionary, where each key-value pair is a column name and a dictionary
                mapping the unique feature values to their integer encoding.
        """
        return self._categories

    def _get_feature_provenance(self):
        provenance = {}
        for col_name in self._date_time_col_names:
            provenance[col_name] = []
            for feature in self.parameters["features_to_extract"]:
                provenance[col_name].append(f"{col_name}_{feature}")
        return provenance
