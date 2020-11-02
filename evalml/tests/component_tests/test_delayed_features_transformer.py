import pandas as pd
import pytest
from evalml.pipelines import DelayedFeaturesTransformer


@pytest.fixture
def delayed_features_data():
    X = pd.DataFrame({"feature": range(1, 32)})
    y = pd.Series(range(1, 32))
    return X, y


def test_delayed_features_transformer_init():

    delayed_features = DelayedFeaturesTransformer(max_delay=4, random_state=1)
    assert delayed_features.parameters == {"max_delay": 4}


def test_lagged_feature_extractor_maxdelay3_gap1(delayed_features_data):
    X, y = delayed_features_data

    # Example 1 from the design document
    answer = pd.DataFrame({"feature_delay_0": X.feature,
                           "feature_delay_1": X.feature.shift(1),
                           "feature_delay_2": X.feature.shift(2),
                           "feature_delay_3": X.feature.shift(3),
                           "target_delay_0": y,
                           "target_delay_1": y.shift(1),
                           "target_delay_2": y.shift(2),
                           "target_delay_3": y.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_delay=3, gap=1).fit_transform(X, y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y,
                                  "target_delay_1": y.shift(1),
                                  "target_delay_2": y.shift(2),
                                  "target_delay_3": y.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_delay=3, gap=1).fit_transform(y), answer_only_y)


def test_lagged_feature_extractor_maxdelay5_gap1(delayed_features_data):

    X, y = delayed_features_data

    # Example 2 from the design document - Note that min_delay is not supported yet.
    answer = pd.DataFrame({"feature_delay_0": X.feature,
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

    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_delay=5, gap=1).fit_transform(X, y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y,
                                  "target_delay_1": y.shift(1),
                                  "target_delay_2": y.shift(2),
                                  "target_delay_3": y.shift(3),
                                  "target_delay_4": y.shift(4),
                                  "target_delay_5": y.shift(5)})
    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_delay=5, gap=1).fit_transform(y), answer_only_y)


def test_lagged_feature_extractor_maxdelay3_gap7(delayed_features_data):

    X, y = delayed_features_data

    # Example 3 from the design document
    answer = pd.DataFrame({"feature_delay_0": X.feature,
                           "feature_delay_1": X.feature.shift(1),
                           "feature_delay_2": X.feature.shift(2),
                           "feature_delay_3": X.feature.shift(3),
                           "target_delay_0": y,
                           "target_delay_1": y.shift(1),
                           "target_delay_2": y.shift(2),
                           "target_delay_3": y.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_delay=3, gap=7).fit_transform(X, y), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y,
                                  "target_delay_1": y.shift(1),
                                  "target_delay_2": y.shift(2),
                                  "target_delay_3": y.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_delay=3, gap=7).fit_transform(y), answer_only_y)


def test_lagged_feature_extractor_numpy(delayed_features_data):
    X, y = delayed_features_data
    X_np = X.values
    y_np = y.values

    # Example 3 from the design document
    answer = pd.DataFrame({"0_delay_0": X.feature,
                           "0_delay_1": X.feature.shift(1),
                           "0_delay_2": X.feature.shift(2),
                           "0_delay_3": X.feature.shift(3),
                           "target_delay_0": y,
                           "target_delay_1": y.shift(1),
                           "target_delay_2": y.shift(2),
                           "target_delay_3": y.shift(3)})

    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_delay=3, gap=7).fit_transform(X_np, y_np), answer)

    answer_only_y = pd.DataFrame({"target_delay_0": y,
                                  "target_delay_1": y.shift(1),
                                  "target_delay_2": y.shift(2),
                                  "target_delay_3": y.shift(3)})
    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_delay=3, gap=7).fit_transform(y_np), answer_only_y)
