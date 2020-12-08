import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import DateTimeFeaturizer


def test_datetime_featurizer_init():
    datetime_transformer = DateTimeFeaturizer()
    assert datetime_transformer.parameters == {"features_to_extract": ["year", "month", "day_of_week", "hour"],
                                               "encode_as_categories": False}

    datetime_transformer = DateTimeFeaturizer(features_to_extract=["year", "month"], encode_as_categories=True)
    assert datetime_transformer.parameters == {"features_to_extract": ["year", "month"],
                                               "encode_as_categories": True}

    with pytest.raises(ValueError, match="not valid options for features_to_extract"):
        DateTimeFeaturizer(features_to_extract=["invalid", "parameters"])


def test_datetime_featurizer_encodes_as_ints():
    X = pd.DataFrame({"date": ["2016-04-10 16:10:09", "2017-03-15 13:32:05", "2018-07-10 07:15:10",
                               "2019-08-19 20:20:20", "2020-01-03 06:45:12"]})
    dt = DateTimeFeaturizer()
    X_transformed = dt.fit_transform(X)
    answer = pd.DataFrame({"date_year": [2016, 2017, 2018, 2019, 2020],
                           "date_month": [3, 2, 6, 7, 0],
                           "date_day_of_week": [0, 3, 2, 1, 5],
                           "date_hour": [16, 13, 7, 20, 6]})
    feature_names = {'date_month': {'April': 3, 'March': 2, 'July': 6, 'August': 7, 'January': 0},
                     'date_day_of_week': {'Sunday': 0, 'Wednesday': 3, 'Tuesday': 2, 'Monday': 1, 'Friday': 5}
                     }
    pd.testing.assert_frame_equal(X_transformed, answer)
    assert dt.get_feature_names() == feature_names

    # Test that changing encode_as_categories to True only changes the dtypes but not the values
    dt_with_cats = DateTimeFeaturizer(encode_as_categories=True)
    X_transformed = dt_with_cats.fit_transform(X)
    answer = answer.astype({"date_day_of_week": "category", "date_month": "category"})
    pd.testing.assert_frame_equal(X_transformed, answer)
    assert dt_with_cats.get_feature_names() == feature_names

    # Test that sequential calls to the same DateTimeFeaturizer work as expected by using the first dt we defined
    X = pd.DataFrame({"date": ["2020-04-10", "2017-03-15", "2019-08-19"]})
    X_transformed = dt.fit_transform(X)
    answer = pd.DataFrame({"date_year": [2020, 2017, 2019],
                           "date_month": [3, 2, 7],
                           "date_day_of_week": [5, 3, 1],
                           "date_hour": [0, 0, 0]})
    pd.testing.assert_frame_equal(X_transformed, answer)
    assert dt.get_feature_names() == {'date_month': {'April': 3, 'March': 2, 'August': 7},
                                      'date_day_of_week': {'Friday': 5, 'Wednesday': 3, 'Monday': 1}}

    dt = DateTimeFeaturizer(features_to_extract=["year", "hour"])
    dt.fit_transform(X)
    assert dt.get_feature_names() == {}


def test_datetime_featurizer_transform():
    datetime_transformer = DateTimeFeaturizer(features_to_extract=["year"])
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
    assert datetime_transformer.get_feature_names() == {}


def test_datetime_featurizer_fit_transform():
    datetime_transformer = DateTimeFeaturizer(features_to_extract=["year"])
    X = pd.DataFrame({'Numerical 1': range(20),
                      'Date Col 1': pd.date_range('2020-05-19', periods=20, freq='D'),
                      'Date Col 2': pd.date_range('2020-02-03', periods=20, freq='W'),
                      'Numerical 2': [0] * 20})
    transformed = datetime_transformer.fit_transform(X)
    assert list(transformed.columns) == ['Numerical 1', 'Numerical 2', 'Date Col 1_year', 'Date Col 2_year']
    assert transformed["Date Col 1_year"].equals(pd.Series([2020] * 20))
    assert transformed["Date Col 2_year"].equals(pd.Series([2020] * 20))
    assert datetime_transformer.get_feature_names() == {}


def test_datetime_featurizer_no_col_names():
    datetime_transformer = DateTimeFeaturizer()
    X = pd.DataFrame(pd.Series(pd.date_range('2020-02-24', periods=10, freq='D')))
    datetime_transformer.fit(X)
    assert list(datetime_transformer.transform(X).columns) == ['0_year', '0_month', '0_day_of_week', '0_hour']
    assert datetime_transformer.get_feature_names() == {'0_month': {'February': 1, 'March': 2},
                                                        '0_day_of_week': {'Monday': 1, 'Tuesday': 2,
                                                                          'Wednesday': 3, 'Thursday': 4, 'Friday': 5,
                                                                          'Saturday': 6, 'Sunday': 0}}


def test_datetime_featurizer_no_features_to_extract():
    datetime_transformer = DateTimeFeaturizer(features_to_extract=[])
    rng = pd.date_range('2020-02-24', periods=20, freq='D')
    X = pd.DataFrame({"date col": rng, "numerical": [0] * len(rng)})
    datetime_transformer.fit(X)
    assert datetime_transformer.transform(X).equals(X)
    assert datetime_transformer.get_feature_names() == {}


def test_datetime_featurizer_custom_features_to_extract():
    datetime_transformer = DateTimeFeaturizer(features_to_extract=["month", "year"])
    rng = pd.date_range('2020-02-24', periods=20, freq='D')
    X = pd.DataFrame({"date col": rng, "numerical": [0] * len(rng)})
    datetime_transformer.fit(X)
    assert list(datetime_transformer.transform(X).columns) == ["numerical", "date col_month", "date col_year"]
    assert datetime_transformer.get_feature_names() == {"date col_month": {"February": 1, "March": 2}}


def test_datetime_featurizer_no_datetime_cols():
    datetime_transformer = DateTimeFeaturizer(features_to_extract=["year", "month"])
    X = pd.DataFrame([[1, 3, 4], [2, 5, 2]])
    datetime_transformer.fit(X)
    assert datetime_transformer.transform(X).equals(X)
    assert datetime_transformer.get_feature_names() == {}


def test_datetime_featurizer_numpy_array_input():
    datetime_transformer = DateTimeFeaturizer()
    X = np.array([['2007-02-03'], ['2016-06-07'], ['2020-05-19']], dtype='datetime64')
    datetime_transformer.fit(X)
    assert list(datetime_transformer.transform(X).columns) == ["0_year", "0_month", "0_day_of_week", "0_hour"]
    assert datetime_transformer.get_feature_names() == {'0_month': {'February': 1, 'June': 5, 'May': 4},
                                                        '0_day_of_week': {'Saturday': 6, 'Tuesday': 2}}
