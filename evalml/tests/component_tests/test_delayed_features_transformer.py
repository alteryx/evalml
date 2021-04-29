import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer
)

from evalml.pipelines import DelayedFeatureTransformer


@pytest.fixture
def delayed_features_data():
    X = pd.DataFrame({"feature": range(1, 32)})
    y = pd.Series(range(1, 32))
    return X, y


def test_delayed_features_transformer_init():
    delayed_features = DelayedFeatureTransformer(max_delay=4, delay_features=True, delay_target=False, date_index="Date",
                                                 random_seed=1)
    assert delayed_features.parameters == {"max_delay": 4, "delay_features": True, "delay_target": False,
                                           "gap": 1, "date_index": "Date"}


def encode_y_as_string(y):
    y_answer = y.astype(int) - 1
    y = y.map(lambda val: str(val).zfill(2))
    return y, y_answer


def encode_X_as_string(X):
    X_answer = X.astype(int) - 1
    # So that the encoder encodes the values in ascending order. This makes it easier to
    # specify the answer for each unit test
    X.feature = pd.Categorical(X.feature.map(lambda val: str(val).zfill(2)))
    return X, X_answer


def encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str):
    y_answer = y
    if encode_y_as_str:
        y, y_answer = encode_y_as_string(y)
    X_answer = X
    if encode_X_as_str:
        X, X_answer = encode_X_as_string(X)
    return X, X_answer, y, y_answer


@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_delayed_feature_extractor_maxdelay3_gap1(encode_X_as_str, encode_y_as_str, delayed_features_data):
    X, y = delayed_features_data
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)
    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X_answer.feature.shift(1),
                           "feature_delay_2": X_answer.feature.shift(2),
                           "feature_delay_3": X_answer.feature.shift(3),
                           "target_delay_0": y_answer.astype("Int64"),
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3)})
    if not encode_X_as_str:
        answer["feature"] = X.feature.astype("Int64")
    if not encode_y_as_str:
        answer["target_delay_0"] = y_answer.astype("Int64")
    assert_frame_equal(answer, DelayedFeatureTransformer(max_delay=3, gap=1).fit_transform(X=X, y=y).to_dataframe())

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer.astype("Int64"),
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3)})
    assert_frame_equal(answer_only_y, DelayedFeatureTransformer(max_delay=3, gap=1).fit_transform(X=None, y=y).to_dataframe())


@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_delayed_feature_extractor_maxdelay5_gap1(encode_X_as_str, encode_y_as_str, delayed_features_data):
    X, y = delayed_features_data
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)
    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X_answer.feature.shift(1),
                           "feature_delay_2": X_answer.feature.shift(2),
                           "feature_delay_3": X_answer.feature.shift(3),
                           "feature_delay_4": X_answer.feature.shift(4),
                           "feature_delay_5": X_answer.feature.shift(5),
                           "target_delay_0": y_answer.astype("Int64"),
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3),
                           "target_delay_4": y_answer.shift(4),
                           "target_delay_5": y_answer.shift(5)})
    if not encode_X_as_str:
        answer["feature"] = X.feature.astype("Int64")
    assert_frame_equal(answer, DelayedFeatureTransformer(max_delay=5, gap=1).fit_transform(X, y).to_dataframe())

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer.astype("Int64"),
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3),
                                  "target_delay_4": y_answer.shift(4),
                                  "target_delay_5": y_answer.shift(5)})
    assert_frame_equal(answer_only_y, DelayedFeatureTransformer(max_delay=5, gap=1).fit_transform(X=None, y=y).to_dataframe())


@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_delayed_feature_extractor_maxdelay3_gap7(encode_X_as_str, encode_y_as_str, delayed_features_data):
    X, y = delayed_features_data
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)
    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X_answer.feature.shift(1),
                           "feature_delay_2": X_answer.feature.shift(2),
                           "feature_delay_3": X_answer.feature.shift(3),
                           "target_delay_0": y_answer.astype("Int64"),
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3)})
    if not encode_X_as_str:
        answer["feature"] = X.feature.astype("Int64")
    assert_frame_equal(answer, DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X, y).to_dataframe())

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer.astype("Int64"),
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3)})
    assert_frame_equal(answer_only_y, DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X=None, y=y).to_dataframe())


@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_delayed_feature_extractor_numpy(encode_X_as_str, encode_y_as_str, delayed_features_data):
    X, y = delayed_features_data
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)
    X_np = X.values
    y_np = y.values
    answer = pd.DataFrame({0: X.feature,
                           "0_delay_1": X_answer.feature.shift(1),
                           "0_delay_2": X_answer.feature.shift(2),
                           "0_delay_3": X_answer.feature.shift(3),
                           "target_delay_0": y_answer.astype("Int64"),
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3)})
    if not encode_X_as_str:
        answer[0] = X.feature.astype("Int64")
    assert_frame_equal(answer, DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X_np, y_np).to_dataframe())

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer.astype("Int64"),
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3)})
    assert_frame_equal(answer_only_y, DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X=None, y=y_np).to_dataframe())


@pytest.mark.parametrize("delay_features,delay_target", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_lagged_feature_extractor_delay_features_delay_target(encode_y_as_str, encode_X_as_str, delay_features,
                                                              delay_target,
                                                              delayed_features_data):
    X, y = delayed_features_data
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)
    all_delays = pd.DataFrame({"feature": X.feature,
                               "feature_delay_1": X_answer.feature.shift(1),
                               "feature_delay_2": X_answer.feature.shift(2),
                               "feature_delay_3": X_answer.feature.shift(3),
                               "target_delay_0": y_answer.astype("Int64"),
                               "target_delay_1": y_answer.shift(1),
                               "target_delay_2": y_answer.shift(2),
                               "target_delay_3": y_answer.shift(3)})
    if not encode_X_as_str:
        all_delays["feature"] = X.feature.astype("Int64")
    if not delay_features:
        all_delays = all_delays.drop(columns=[c for c in all_delays.columns if "feature_" in c])
    if not delay_target:
        all_delays = all_delays.drop(columns=[c for c in all_delays.columns if "target" in c])

    transformer = DelayedFeatureTransformer(max_delay=3, gap=1,
                                            delay_features=delay_features, delay_target=delay_target)
    assert_frame_equal(all_delays, transformer.fit_transform(X, y).to_dataframe())


@pytest.mark.parametrize("delay_features,delay_target", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_lagged_feature_extractor_delay_target(encode_y_as_str, encode_X_as_str, delay_features,
                                               delay_target, delayed_features_data):
    X, y = delayed_features_data
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)
    answer = pd.DataFrame()
    if delay_target:
        answer = pd.DataFrame({"target_delay_0": y_answer.astype("Int64"),
                               "target_delay_1": y_answer.shift(1),
                               "target_delay_2": y_answer.shift(2),
                               "target_delay_3": y_answer.shift(3)})

    transformer = DelayedFeatureTransformer(max_delay=3, gap=1,
                                            delay_features=delay_features, delay_target=delay_target)
    assert_frame_equal(answer, transformer.fit_transform(None, y).to_dataframe())


@pytest.mark.parametrize("gap", [0, 1, 7])
def test_target_delay_when_gap_is_0(gap, delayed_features_data):
    X, y = delayed_features_data
    expected = pd.DataFrame({"feature": X.feature.astype("Int64"),
                             "feature_delay_1": X.feature.shift(1),
                             "target_delay_0": y.astype("Int64"),
                             "target_delay_1": y.shift(1)})

    if gap == 0:
        expected = expected.drop(columns=["target_delay_0"])

    transformer = DelayedFeatureTransformer(max_delay=1, gap=gap)
    assert_frame_equal(expected, transformer.fit_transform(X, y).to_dataframe())
    expected = pd.DataFrame({"target_delay_0": y.astype("Int64"),
                             "target_delay_1": y.shift(1)})

    if gap == 0:
        expected = expected.drop(columns=["target_delay_0"])
    assert_frame_equal(expected, transformer.fit_transform(None, y).to_dataframe())


@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
@pytest.mark.parametrize('data_type', ['ww', 'pd'])
def test_delay_feature_transformer_supports_custom_index(encode_X_as_str, encode_y_as_str, data_type, make_data_type,
                                                         delayed_features_data):
    X, y = delayed_features_data
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)
    X.index = pd.RangeIndex(50, 81)
    X_answer.index = pd.RangeIndex(50, 81)
    y.index = pd.RangeIndex(50, 81)
    y_answer.index = pd.RangeIndex(50, 81)
    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X_answer.feature.shift(1),
                           "feature_delay_2": X_answer.feature.shift(2),
                           "feature_delay_3": X_answer.feature.shift(3),
                           "target_delay_0": y_answer.astype("Int64"),
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3)}, index=pd.RangeIndex(50, 81))
    if not encode_X_as_str:
        answer["feature"] = X.feature.astype("Int64")

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    assert_frame_equal(answer, DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X, y).to_dataframe())

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer.astype("Int64"),
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3)}, index=pd.RangeIndex(50, 81))
    assert_frame_equal(answer_only_y, DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X=None, y=y).to_dataframe())


def test_delay_feature_transformer_multiple_categorical_columns(delayed_features_data):
    X, y = delayed_features_data
    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, True, True)
    X['feature_2'] = pd.Categorical(["a"] * 10 + ['aa'] * 10 + ['aaa'] * 10 + ['aaaa'])
    X_answer['feature_2'] = pd.Series([0] * 10 + [1] * 10 + [2] * 10 + [3])
    answer = pd.DataFrame({"feature": X.feature,
                           'feature_2': X.feature_2,
                           "feature_delay_1": X_answer.feature.shift(1),
                           "feature_2_delay_1": X_answer.feature_2.shift(1),
                           "target_delay_0": y_answer.astype("Int64"),
                           "target_delay_1": y_answer.shift(1),
                           })
    assert_frame_equal(answer, DelayedFeatureTransformer(max_delay=1, gap=11).fit_transform(X, y).to_dataframe())


def test_delay_feature_transformer_y_is_none(delayed_features_data):
    X, _ = delayed_features_data
    answer = pd.DataFrame({"feature": X.feature.astype("Int64"),
                           "feature_delay_1": X.feature.shift(1),
                           })
    assert_frame_equal(answer, DelayedFeatureTransformer(max_delay=1, gap=11).fit_transform(X, y=None).to_dataframe())


@pytest.mark.parametrize("X_df", [pd.DataFrame(pd.to_datetime(['20190902', '20200519', '20190607'], format='%Y%m%d')),
                                  pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
                                  pd.DataFrame(pd.Series([1., 2., 3.], dtype="float")),
                                  pd.DataFrame(pd.Series(['a', 'b', 'a'], dtype="category")),
                                  pd.DataFrame(pd.Series(['this will be a natural language column because length', 'yay', 'hay'], dtype="string"))])
@pytest.mark.parametrize('fit_transform', [True, False])
def test_delay_feature_transformer_woodwork_custom_overrides_returned_by_components(X_df, fit_transform):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, Datetime, Boolean]
    for logical_type in override_types:
        try:
            X = ww.DataTable(X_df, logical_types={0: logical_type})
        except TypeError:
            continue
        dft = DelayedFeatureTransformer(max_delay=1, gap=11)
        if fit_transform:
            transformed = dft.fit_transform(X, y)
        else:
            dft.fit(X, y)
            transformed = dft.transform(X, y)
        assert isinstance(transformed, ww.DataTable)
        if logical_type in [Integer, Double, Categorical]:
            assert transformed.logical_types == {0: logical_type,
                                                 '0_delay_1': Double,
                                                 'target_delay_0': Integer,
                                                 'target_delay_1': Double}
        else:
            assert transformed.logical_types == {0: logical_type,
                                                 '0_delay_1': logical_type,
                                                 'target_delay_0': Integer,
                                                 'target_delay_1': Double}
