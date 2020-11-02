import pandas as pd

from evalml.pipelines import DelayedFeaturesTransformer


def test_lagged_feature_extractor():

    X = pd.DataFrame({"feature": range(1, 32)})
    y = pd.Series(range(1, 32))

    answer = pd.DataFrame({"feature_lag_7": X.feature,
                           "feature_lag_8": X.feature.shift(1),
                           "feature_lag_9": X.feature.shift(2),
                           "target_lag_7": y,
                           "target_lag_8": y.shift(1),
                           "target_lag_9": y.shift(2)})

    pd.testing.assert_frame_equal(DelayedFeaturesTransformer(max_lag=2, gap=7).fit_transform(X, y), answer)
