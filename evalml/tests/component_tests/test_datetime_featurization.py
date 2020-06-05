import pandas as pd
import pytest

from evalml.pipelines.components import DateTimeFeaturization

# from evalml.pipelines.components.transformers.preprocessing.datetime_featurization import


def test_datetime_featurization_init():
    datetime_transformer = DateTimeFeaturization()
    assert datetime_transformer.parameters["features_to_extract"] == ["year", "month", "day_of_week", "hour"]
    assert datetime_transformer._date_time_cols is None

    datetime_transformer = DateTimeFeaturization(features_to_extract=["year", "month"])
    assert datetime_transformer.parameters["features_to_extract"] == ["year", "month"]
    assert datetime_transformer._date_time_cols is None

    with pytest.raises(ValueError, match="'invalid', 'parameters' are not valid options for features_to_extract"):
        DateTimeFeaturization(features_to_extract=["invalid", "parameters"])


def test_datetime_featurization_transform():
    datetime_transformer = DateTimeFeaturization()
    X = pd.date_range('2015-02-24', periods=5, freq='T')


# test: col doesn't have name originally
