import pandas as pd
import pytest

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


def test_delayed_feature_extractor_maxdelay3_gap1(delayed_features_data):
    X, y = delayed_features_data

    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X.feature.shift(1),
                           "feature_delay_2": X.feature.shift(2),
                           "feature_delay_3": X.feature.shift(3),
                           "target_delay_0": y,
                           "target_delay_1": y.shift(1),
                           "target_delay_2": y.shift(2),
                           "target_delay_3": y.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=1).fit_transform(X=X, y=y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y,
                                  "target_delay_1": y.shift(1),
                                  "target_delay_2": y.shift(2),
                                  "target_delay_3": y.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=1).fit_transform(X=None, y=y), answer_only_y)


def test_delayed_feature_extractor_maxdelay5_gap1(delayed_features_data):

    X, y = delayed_features_data

    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X.feature.shift(1),
                           "feature_delay_2": X.feature.shift(2),
                           "feature_delay_3": X.feature.shift(3),
                           "feature_delay_4": X.feature.shift(4),
                           "feature_delay_5": X.feature.shift(5),
                           "target_delay_0": y,
                           "target_delay_1": y.shift(1),
                           "target_delay_2": y.shift(2),
                           "target_delay_3": y.shift(3),
                           "target_delay_4": y.shift(4),
                           "target_delay_5": y.shift(5)})

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=5, gap=1).fit_transform(X, y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y,
                                  "target_delay_1": y.shift(1),
                                  "target_delay_2": y.shift(2),
                                  "target_delay_3": y.shift(3),
                                  "target_delay_4": y.shift(4),
                                  "target_delay_5": y.shift(5)})
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=5, gap=1).fit_transform(X=None, y=y), answer_only_y)


def test_delayed_feature_extractor_maxdelay3_gap7(delayed_features_data):

    X, y = delayed_features_data

    answer = pd.DataFrame({"feature": X.feature,
                           "feature_delay_1": X.feature.shift(1),
                           "feature_delay_2": X.feature.shift(2),
                           "feature_delay_3": X.feature.shift(3),
                           "target_delay_0": y,
                           "target_delay_1": y.shift(1),
                           "target_delay_2": y.shift(2),
                           "target_delay_3": y.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X, y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y,
                                  "target_delay_1": y.shift(1),
                                  "target_delay_2": y.shift(2),
                                  "target_delay_3": y.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X=None, y=y), answer_only_y)


def test_delayed_feature_extractor_numpy(delayed_features_data):
    X, y = delayed_features_data
    X_np = X.values
    y_np = y.values

    answer = pd.DataFrame({0: X.feature,
                           "0_delay_1": X.feature.shift(1),
                           "0_delay_2": X.feature.shift(2),
                           "0_delay_3": X.feature.shift(3),
                           "target_delay_0": y,
                           "target_delay_1": y.shift(1),
                           "target_delay_2": y.shift(2),
                           "target_delay_3": y.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X_np, y_np), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y,
                                  "target_delay_1": y.shift(1),
                                  "target_delay_2": y.shift(2),
                                  "target_delay_3": y.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeatureTransformer(max_delay=3, gap=7).fit_transform(X=None, y=y_np), answer_only_y)


@pytest.mark.parametrize("delay_features,delay_target", [(False, True), (True, False), (False, False)])
def test_lagged_feature_extractor_delay_features_delay_target(delay_features, delay_target, delayed_features_data):
    X, y = delayed_features_data

    all_delays = pd.DataFrame({"feature": X.feature,
                               "feature_delay_1": X.feature.shift(1),
                               "feature_delay_2": X.feature.shift(2),
                               "feature_delay_3": X.feature.shift(3),
                               "target_delay_0": y,
                               "target_delay_1": y.shift(1),
                               "target_delay_2": y.shift(2),
                               "target_delay_3": y.shift(3)})
    if not delay_features:
        all_delays = all_delays.drop(columns=[c for c in all_delays.columns if "feature_" in c])
    if not delay_target:
        all_delays = all_delays.drop(columns=[c for c in all_delays.columns if "target" in c])

    transformer = DelayedFeatureTransformer(max_delay=3, gap=1,
                                            delay_features=delay_features, delay_target=delay_target)
    pd.testing.assert_frame_equal(transformer.fit_transform(X, y), all_delays)


@pytest.mark.parametrize("delay_features,delay_target", [(False, True), (True, False), (False, False)])
def test_lagged_feature_extractor_delay_target(delay_features, delay_target, delayed_features_data):
    X, y = delayed_features_data

    answer = pd.DataFrame()
    if delay_target:
        answer = pd.DataFrame({"target_delay_0": y,
                               "target_delay_1": y.shift(1),
                               "target_delay_2": y.shift(2),
                               "target_delay_3": y.shift(3)})

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
