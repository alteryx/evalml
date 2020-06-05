import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class DateTimeFeaturization(Transformer):
    """"""
    name = "DateTime Featurization Component"
    hyperparameter_ranges = {}

    def __init__(self, features_to_extract=None, random_state=0):
        """Extracts features from DateTime columns

        Arguments:
            features_to_extract (list)
        """
        if features_to_extract is None:
            features_to_extract = ["year", "month", "day_of_week", "hour"]

        valid_features = ["year", "month", "day_of_week", "hour"]
        invalid_features = [feature for feature in features_to_extract if feature not in valid_features]
        if len(invalid_features) > 0:
            raise ValueError("{} are not valid options for features_to_extract".format(", ".join([f"'{feature}'" for feature in invalid_features])))
        parameters = {"features_to_extract": features_to_extract}

        X = pd.DataFrame(pd.date_range('2015-02-24', periods=5, freq='T'))[0]

        self._date_time_cols = None
        self.functions = [self.mappings[key] for key in features_to_extract]
        for f in self.functions:
            print (f(X))
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._date_time_cols = X.select_dtypes(include=['datetime64'])
        return self

    def _extract_year(col):
        col_name = f"{col.name}_Year"
        return col.apply(lambda datetime_val: datetime_val.strftime('%H')).reset_index(drop=True).rename(col_name)

    def _extract_month(col):
        col_name = f"{col.name}_Month"
        return col.apply(lambda datetime_val: datetime_val.strftime("%B")).reset_index(drop=True).rename(col_name)

    def _extract_day_of_week(col):
        col_name = f"{col.name}_DayOfWeek"
        return col.dt.hour.rename(col_name)

    def _extract_hour(col):
        col_name = f"{col.name}_Hour"
        return col.dt.year.rename(col_name)

    mappings = {"year": _extract_year, "month": _extract_month, "day_of_week": _extract_day_of_week, "hour": _extract_hour}

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
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)


def _extract_year(col):
    return col.to_series().apply(lambda datetime_val: datetime_val.strftime('%H')).reset_index(drop=True)


def _extract_month(col):
    return col.to_series().apply(lambda datetime_val: datetime_val.strftime("%B")).reset_index(drop=True)


def _extract_day_of_week(col):
    return col.hour


def _extract_hour(col):
    return col.year
