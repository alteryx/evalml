import pandas as pd

from evalml.pipelines.components.transformers import Transformer


def _extract_year(col):
    return col.dt.year


def _extract_month(col):
    return col.dt.month_name().astype('category')


def _extract_day_of_week(col):
    return col.dt.day_name().astype('category')


def _extract_hour(col):
    return col.dt.hour


class DateTimeFeaturization(Transformer):
    """Transformer that can automatically featurize DateTime columns."""
    name = "DateTime Featurization Component"
    hyperparameter_ranges = {}
    function_mappings = {"year": _extract_year, "month": _extract_month, "day_of_week": _extract_day_of_week, "hour": _extract_hour}

    def __init__(self, features_to_extract=None, random_state=0):
        """Extracts features from DateTime columns

        Arguments:
            features_to_extract (list):
            random_state (int, np.random.RandomState): Seed for the random number generator.

        """
        if features_to_extract is None:
            features_to_extract = ["year", "month", "day_of_week", "hour"]

        valid_features = ["year", "month", "day_of_week", "hour"]
        invalid_features = [feature for feature in features_to_extract if feature not in valid_features]
        if len(invalid_features) > 0:
            raise ValueError("{} are not valid options for features_to_extract".format(", ".join([f"'{feature}'" for feature in invalid_features])))
        parameters = {"features_to_extract": features_to_extract}
        self._date_time_cols = None
        self.featurization_functions = {key: self.function_mappings[key] for key in features_to_extract}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._date_time_cols = X.select_dtypes(include=['datetime64'])
        return self

    def transform(self, X, y=None):
        """Transforms data X by creating new features using existing DateTime columns, and then dropping those DateTime columns

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        if self._date_time_cols is None:
            raise RuntimeError(f"You must fit {self.name} before calling transform!")
        X_t = X
        if not isinstance(X_t, pd.DataFrame):
            X_t = pd.DataFrame(X_t)
        for col_name, col in self._date_time_cols.iteritems():
            for feature, feature_function in self.featurization_functions.items():
                X_t[f"{col_name}_{feature}"] = feature_function(col)
        return X_t.drop(self._date_time_cols, axis=1)
