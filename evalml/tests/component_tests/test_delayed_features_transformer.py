import pandas as pd
import pytest
import woodwork as ww

from evalml.pipelines import DelayedFeatureTransformer


@pytest.fixture
def delayed_features_data():
    X = pd.DataFrame({"feature": range(1, 32)})
    y = pd.Series(range(1, 32))
    return X, y


def test_delayed_features_transformer_init():

    delayed_features = DelayedFeatureTransformer(max_delay=4, delay_features=True, delay_target=False,
                                                 random_state=1)
    assert delayed_features.parameters == {"max_delay": 4, "delay_features": True, "delay_target": False,
                                           "gap": 1}


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
                           "target_delay_0": y_answer,
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=1).fit_transform(X=X, y=y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer,
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=1).fit_transform(X=None, y=y), answer_only_y)


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
                           "target_delay_0": y_answer,
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3),
                           "target_delay_4": y_answer.shift(4),
                           "target_delay_5": y_answer.shift(5)})

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=5, gap=1).fit_transform(X, y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer,
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3),
                                  "target_delay_4": y_answer.shift(4),
                                  "target_delay_5": y_answer.shift(5)})
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=5, gap=1).fit_transform(X=None, y=y), answer_only_y)


@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_delayed_feature_extractor_maxdelay3_gap7(encode_X_as_str, encode_y_as_str, delayed_features_data):

    X, y = delayed_features_data

    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)

    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X_answer.feature.shift(1),
                           "feature_delay_2": X_answer.feature.shift(2),
                           "feature_delay_3": X_answer.feature.shift(3),
                           "target_delay_0": y_answer,
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X, y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer,
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X=None, y=y), answer_only_y)


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
                           "target_delay_0": y_answer,
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X_np, y_np), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer,
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X=None, y=y_np), answer_only_y)


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
                               "target_delay_0": y_answer,
                               "target_delay_1": y_answer.shift(1),
                               "target_delay_2": y_answer.shift(2),
                               "target_delay_3": y_answer.shift(3)})
    if not delay_features:
        all_delays = all_delays.drop(columns=[c for c in all_delays.columns if "feature_" in c])
    if not delay_target:
        all_delays = all_delays.drop(columns=[c for c in all_delays.columns if "target" in c])

    transformer = DelayedFeatureTransformer(max_delay=3, gap=1,
                                            delay_features=delay_features, delay_target=delay_target)
    pd.testing.assert_frame_equal(transformer.fit_transform(X, y), all_delays)


@pytest.mark.parametrize("delay_features,delay_target", [(False, True), (True, False), (False, False)])
@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_lagged_feature_extractor_delay_target(encode_y_as_str, encode_X_as_str, delay_features,
                                               delay_target, delayed_features_data):
    X, y = delayed_features_data

    X, X_answer, y, y_answer = encode_X_y_as_strings(X, y, encode_X_as_str, encode_y_as_str)

    answer = pd.DataFrame()
    if delay_target:
        answer = pd.DataFrame({"target_delay_0": y_answer,
                               "target_delay_1": y_answer.shift(1),
                               "target_delay_2": y_answer.shift(2),
                               "target_delay_3": y_answer.shift(3)})

    transformer = DelayedFeatureTransformer(max_delay=3, gap=1,
                                            delay_features=delay_features, delay_target=delay_target)
    pd.testing.assert_frame_equal(transformer.fit_transform(None, y), answer)


@pytest.mark.parametrize("gap", [0, 1, 7])
def test_target_delay_when_gap_is_0(gap, delayed_features_data):
    X, y = delayed_features_data

    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X.feature.shift(1),
                           "target_delay_0": y,
                           "target_delay_1": y.shift(1)})

    if gap == 0:
        answer = answer.drop(columns=["target_delay_0"])

    transformer = DelayedFeatureTransformer(max_delay=1, gap=gap)
    pd.testing.assert_frame_equal(transformer.fit_transform(X, y), answer)

    answer = pd.DataFrame({"target_delay_0": y,
                           "target_delay_1": y.shift(1)})

    if gap == 0:
        answer = answer.drop(columns=["target_delay_0"])

    pd.testing.assert_frame_equal(transformer.fit_transform(None, y), answer)


@pytest.mark.parametrize('use_woodwork', [True, False])
@pytest.mark.parametrize('encode_X_as_str', [True, False])
@pytest.mark.parametrize('encode_y_as_str', [True, False])
def test_delay_feature_transformer_supports_custom_index(encode_X_as_str, encode_y_as_str, use_woodwork,
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
                           "target_delay_0": y_answer,
                           "target_delay_1": y_answer.shift(1),
                           "target_delay_2": y_answer.shift(2),
                           "target_delay_3": y_answer.shift(3)}, index=pd.RangeIndex(50, 81))

    if use_woodwork:
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X, y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y_answer,
                                  "target_delay_1": y_answer.shift(1),
                                  "target_delay_2": y_answer.shift(2),
                                  "target_delay_3": y_answer.shift(3)}, index=pd.RangeIndex(50, 81))
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X=None, y=y),
                                  answer_only_y)
