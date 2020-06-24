import numpy as np
import pandas as pd

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import classproperty


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
    _function_mappings = {"year": _extract_year,
                          "month": _extract_month,
                          "day_of_week": _extract_day_of_week,
                          "hour": _extract_hour}

    def __init__(self, features_to_extract=None, random_state=0, **kwargs):
        """Extracts features from DateTime columns

        Arguments:
            features_to_extract (list): list of features to extract. Valid options include "year", "month", "day_of_week", "hour".
            random_state (int, np.random.RandomState): Seed for the random number generator.

        """
        if features_to_extract is None:
            features_to_extract = ["year", "month", "day_of_week", "hour"]
        invalid_features = set(features_to_extract) - set(self._function_mappings.keys())
        if len(invalid_features) > 0:
            raise ValueError("{} are not valid options for features_to_extract".format(", ".join([f"'{feature}'" for feature in invalid_features])))

        parameters = {"features_to_extract": features_to_extract}
        parameters.update(kwargs)

        self._date_time_col_names = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    @classproperty
    def default_parameters(cls,):
        """Returns the default parameters for this component."""
        # Our convention is that default parameters are what get passed in to the parameters dict
        # when nothing is changed to the init. In this case, since we use None to encode a list of date units,
        # we need to manually specify the defaults.
        return {"features_to_extract": ["year", "month", "day_of_week", "hour"]}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._date_time_col_names = X.select_dtypes(include=[np.datetime64]).columns
        return self

    def transform(self, X, y=None):
        """Transforms data X by creating new features using existing DateTime columns, and then dropping those DateTime columns

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        if self._date_time_col_names is None:
            raise RuntimeError(f"You must fit {self.name} before calling transform!")
        X_t = X
        if not isinstance(X_t, pd.DataFrame):
            X_t = pd.DataFrame(X_t)
        features_to_extract = self.parameters["features_to_extract"]
        if len(features_to_extract) == 0:
            return X_t
        for col_name in self._date_time_col_names:
            for feature in features_to_extract:
                X_t[f"{col_name}_{feature}"] = self._function_mappings[feature](X_t[col_name])
        return X_t.drop(self._date_time_col_names, axis=1)
