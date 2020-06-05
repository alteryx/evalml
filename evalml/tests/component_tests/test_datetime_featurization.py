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
    rng = pd.date_range('2020-02-24', periods=20, freq='D')
    X = pd.DataFrame({'Date': rng, 'Val': [0] * len(rng)})
    datetime_transformer.fit(X)

    # print (datetime_transformer.transform(X))
    # assert
# test: col doesn't have name originally
# test: no datetime cols
# test: some datetime cols
# assert original column is gone
# test features_to_extract is empty list
# test custom features_to_extract
# assert type is categorical
