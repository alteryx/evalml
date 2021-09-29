from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_understanding import explain, get_influential_features
from evalml.pipelines import BinaryClassificationPipeline

xgboost_component_graph = [
    "Imputer",
    "One Hot Encoder",
    "DateTime Featurization Component",
    "XGBoost Classifier",
]

elasticnet_component_graph = [
    "Imputer",
    "One Hot Encoder",
    "DateTime Featurization Component",
    "Elastic Net Classifier",
]


def test_get_influential_features():
    importance_df = pd.DataFrame(
        {
            "feature": [
                "heavy influence",
                "somewhat influence",
                "zero influence",
                "negative influence",
            ],
            "importance": [0.6, 0.1, 0.0, -0.1],
        }
    )
    heavy, somewhat, negative = get_influential_features(importance_df)
    assert heavy == ["heavy influence"]
    assert somewhat == ["somewhat influence"]
    assert negative == ["negative influence"]


def test_get_influential_features_max_features():
    importance_df = pd.DataFrame(
        {
            "feature": ["heavy 1", "heavy 2", "heavy 3", "somewhat 1", "somewhat 2"],
            "importance": [0.35, 0.3, 0.23, 0.15, 0.07],
        }
    )
    heavy, somewhat, negative = get_influential_features(importance_df, max_features=2)
    assert heavy == ["heavy 1", "heavy 2"]
    assert somewhat == []
    assert negative == []

    heavy, somewhat, negative = get_influential_features(importance_df, max_features=4)
    assert heavy == ["heavy 1", "heavy 2", "heavy 3"]
    assert somewhat == ["somewhat 1"]
    assert negative == []


def test_get_influential_features_max_features_ignore_negative():
    importance_df = pd.DataFrame(
        {
            "feature": ["heavy 1", "heavy 2", "heavy 3", "neg 1", "neg 2", "neg 3"],
            "importance": [0.35, 0.3, 0.23, -0.15, -0.17, -0.43],
        }
    )
    heavy, somewhat, negative = get_influential_features(importance_df, max_features=2)
    assert heavy == ["heavy 1", "heavy 2"]
    assert somewhat == []
    assert negative == ["neg 1", "neg 2", "neg 3"]


def test_get_influential_features_lower_min_importance_threshold():
    importance_df = pd.DataFrame(
        {"feature": np.arange(100), "importance": [0.1] * 25 + [0.05] * 75}
    )
    heavy, somewhat, negative = get_influential_features(importance_df)
    assert heavy == []
    assert somewhat == []
    assert negative == []

    heavy, somewhat, negative = get_influential_features(
        importance_df, min_importance_threshold=0.01
    )
    assert heavy == []
    assert somewhat == [0, 1, 2, 3, 4]
    assert negative == []


def test_get_influential_features_higher_min_importance_threshold():
    importance_df = pd.DataFrame(
        {"feature": np.arange(5), "importance": [0.55, 0.201, 0.15, 0.075, 0.024]}
    )
    heavy, somewhat, negative = get_influential_features(importance_df)
    assert heavy == [0, 1]
    assert somewhat == [2, 3]
    assert negative == []

    heavy, somewhat, negative = get_influential_features(
        importance_df, min_importance_threshold=0.15
    )
    assert heavy == [0]
    assert somewhat == [1, 2]
    assert negative == []


def test_get_influential_features_linear_importance():
    importance_df = pd.DataFrame(
        {
            "feature": [
                "heavy influence",
                "negative influence",
                "somewhat influence",
                "zero influence",
            ],
            "importance": [0.6, -0.2, 0.1, 0.0],
        }
    )
    heavy, somewhat, negative = get_influential_features(
        importance_df, linear_importance=True
    )
    assert heavy == ["heavy influence", "negative influence"]
    assert somewhat == ["somewhat influence"]
    assert negative == []


def test_explain_pipeline_not_fitted(fraud_100):
    pipeline = BinaryClassificationPipeline(xgboost_component_graph)
    X, y = fraud_100

    with pytest.raises(
        ValueError,
        match="Pipelines must be fitted in order to run feature explanations",
    ):
        explain(pipeline, X, y)

    with pytest.raises(
        ValueError,
        match="Pipelines must be fitted in order to run feature explanations",
    ):
        explain(pipeline, importance_method="feature")


def test_explain_pipeline_missing_X_y(fraud_100):
    pipeline = BinaryClassificationPipeline(xgboost_component_graph)
    X, y = fraud_100
    pipeline._is_fitted = True

    with pytest.raises(
        ValueError, match="required parameters for explaining pipelines"
    ):
        explain(pipeline)

    with pytest.raises(
        ValueError, match="required parameters for explaining pipelines"
    ):
        explain(pipeline, X)

    with pytest.raises(
        ValueError, match="required parameters for explaining pipelines"
    ):
        explain(pipeline, y=y)


def test_explain_pipeline_invalid_importance(fraud_100):
    pipeline = BinaryClassificationPipeline(xgboost_component_graph)
    X, y = fraud_100
    pipeline._is_fitted = True

    with pytest.raises(ValueError, match="Unknown importance method"):
        explain(pipeline, importance_method="fake")


def test_explain_pipeline_invalid_min_threshold(fraud_100):
    pipeline = BinaryClassificationPipeline(xgboost_component_graph)
    X, y = fraud_100
    pipeline._is_fitted = True

    with pytest.raises(
        ValueError, match="minimum importance threshold must be a percentage"
    ):
        explain(pipeline, X, y, min_importance_threshold=-1)

    with pytest.raises(
        ValueError, match="minimum importance threshold must be a percentage"
    ):
        explain(pipeline, X, y, min_importance_threshold=2)


def test_explain_pipeline_permutation(caplog, fraud_100):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    X, y = fraud_100
    pipeline.fit(X, y)

    explain(pipeline, X, y, importance_method="permutation")

    out = caplog.text
    expected_influence = "Elastic Net Classifier: The performance of predicting fraud as measured by log loss binary is heavily influenced by lng."
    expected_negative = "The features customer_present, lat, amount, card_id, and store_id detracted from model performance. We suggest removing these features."

    assert expected_influence in out
    assert expected_negative in out
    caplog.clear()


def test_explain_pipeline_different_objective(caplog, fraud_100):
    pipeline = BinaryClassificationPipeline(xgboost_component_graph)
    X, y = fraud_100
    pipeline.fit(X, y)

    explain(pipeline, X, y, objective="precision")

    out = caplog.text
    expected_output = "XGBoost Classifier: The performance of predicting fraud as measured by precision is heavily influenced by amount"

    assert expected_output in out
    caplog.clear()


def test_explain_pipeline_feature(caplog, fraud_100):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    X, y = fraud_100
    pipeline.fit(X, y)

    explain(pipeline, importance_method="feature")

    out = caplog.text
    expected_heavy = "Elastic Net Classifier: The output is heavily influenced by datetime_year and store_id"
    expected_somewhat = "somewhat influenced by card_id"

    assert expected_heavy in out
    assert expected_somewhat in out
    assert "detracted from model performance" not in out
    caplog.clear()
