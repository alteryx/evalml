import pytest

from evalml.pipelines.components import DateTimeFeaturization


def test_datetime_featurization_init():
    datetime_transformer = DateTimeFeaturization()
    assert datetime_transformer.parameters["features_to_extract"] == ["year", "month", "day_of_week", "hour"]
    assert datetime_transformer._date_time_cols is None

    datetime_transformer = DateTimeFeaturization(features_to_extract=["year", "month"])
    assert datetime_transformer.parameters["features_to_extract"] == ["year", "month"]
    assert datetime_transformer._date_time_cols is None

    with pytest.raises(ValueError, match="'invalid', 'parameters' are not valid options for features_to_extract"):
        DateTimeFeaturization(features_to_extract=["invalid", "parameters"])
