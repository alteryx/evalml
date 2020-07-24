import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import DateTimeFeaturization


def test_datetime_featurization_init():
    datetime_transformer = DateTimeFeaturization()
    assert datetime_transformer.parameters == {"features_to_extract": ["year", "month", "day_of_week", "hour"]}

    datetime_transformer = DateTimeFeaturization(features_to_extract=["year", "month"])
    assert datetime_transformer.parameters == {"features_to_extract": ["year", "month"]}

    with pytest.raises(ValueError, match="not valid options for features_to_extract"):
        DateTimeFeaturization(features_to_extract=["invalid", "parameters"])


def test_datetime_featurization_transform_without_fit():
    datetime_transformer = DateTimeFeaturization()
    with pytest.raises(RuntimeError, match="You must fit DateTime Featurization Component before calling transform!"):
        datetime_transformer.transform(pd.DataFrame())


def test_datetime_featurization_transform():
    datetime_transformer = DateTimeFeaturization(features_to_extract=["year"])
    X = pd.DataFrame({'Numerical 1': range(20),
                      'Date Col 1': pd.date_range('2000-05-19', periods=20, freq='D'),
                      'Date Col 2': pd.date_range('2000-02-03', periods=20, freq='W'),
                      'Numerical 2': [0] * 20})
    X_test = pd.DataFrame({'Numerical 1': range(20),
                           'Date Col 1': pd.date_range('2020-05-19', periods=20, freq='D'),
                           'Date Col 2': pd.date_range('2020-02-03', periods=20, freq='W'),
                           'Numerical 2': [0] * 20})
    datetime_transformer.fit(X)
    transformed = datetime_transformer.transform(X_test)
    assert list(transformed.columns) == ['Numerical 1', 'Numerical 2', 'Date Col 1_year', 'Date Col 2_year']
    assert transformed["Date Col 1_year"].equals(pd.Series([2020] * 20))
    assert transformed["Date Col 2_year"].equals(pd.Series([2020] * 20))


def test_datetime_featurization_fit_transform():
    datetime_transformer = DateTimeFeaturization(features_to_extract=["year"])
    X = pd.DataFrame({'Numerical 1': range(20),
                      'Date Col 1': pd.date_range('2020-05-19', periods=20, freq='D'),
                      'Date Col 2': pd.date_range('2020-02-03', periods=20, freq='W'),
                      'Numerical 2': [0] * 20})
    transformed = datetime_transformer.fit_transform(X)
    assert list(transformed.columns) == ['Numerical 1', 'Numerical 2', 'Date Col 1_year', 'Date Col 2_year']
    assert transformed["Date Col 1_year"].equals(pd.Series([2020] * 20))
    assert transformed["Date Col 2_year"].equals(pd.Series([2020] * 20))


def test_datetime_featurization_no_col_names():
    datetime_transformer = DateTimeFeaturization()
    X = pd.DataFrame(pd.Series(pd.date_range('2020-02-24', periods=10, freq='D')))
    datetime_transformer.fit(X)
    assert list(datetime_transformer.transform(X).columns) == ['0_year', '0_month', '0_day_of_week', '0_hour']


def test_datetime_featurization_no_features_to_extract():
    datetime_transformer = DateTimeFeaturization(features_to_extract=[])
    rng = pd.date_range('2020-02-24', periods=20, freq='D')
    X = pd.DataFrame({"date col": rng, "numerical": [0] * len(rng)})
    datetime_transformer.fit(X)
    assert datetime_transformer.transform(X).equals(X)


def test_datetime_featurization_custom_features_to_extract():
    datetime_transformer = DateTimeFeaturization(features_to_extract=["month", "year"])
    rng = pd.date_range('2020-02-24', periods=20, freq='D')
    X = pd.DataFrame({"date col": rng, "numerical": [0] * len(rng)})
    datetime_transformer.fit(X)
    assert list(datetime_transformer.transform(X).columns) == ["numerical", "date col_month", "date col_year"]


def test_datetime_featurization_no_datetime_cols():
    datetime_transformer = DateTimeFeaturization(features_to_extract=["year", "month"])
    X = pd.DataFrame([[1, 3, 4], [2, 5, 2]])
    datetime_transformer.fit(X)
    assert datetime_transformer.transform(X).equals(X)


def test_datetime_featurization_numpy_array_input():
    datetime_transformer = DateTimeFeaturization()
    X = np.array(['2007-02-03', '2016-06-07', '2020-05-19'], dtype='datetime64')
    datetime_transformer.fit(X)
    assert list(datetime_transformer.transform(X).columns) == ["0_year", "0_month", "0_day_of_week", "0_hour"]
