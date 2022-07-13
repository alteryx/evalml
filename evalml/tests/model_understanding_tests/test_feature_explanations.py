from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest

from evalml.model_understanding import get_influential_features, readable_explanation
from evalml.pipelines import BinaryClassificationPipeline


@pytest.fixture
def elasticnet_component_graph():
    return [
        "Imputer",
        "One Hot Encoder",
        "DateTime Featurizer",
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
        },
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
        },
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
        },
    )
    heavy, somewhat, negative = get_influential_features(importance_df, max_features=2)
    assert heavy == ["heavy 1", "heavy 2"]
    assert somewhat == []
    assert negative == ["neg 1", "neg 2", "neg 3"]


def test_get_influential_features_lower_min_importance_threshold():
    importance_df = pd.DataFrame(
        {"feature": np.arange(100), "importance": [0.1] * 25 + [0.05] * 75},
    )
    heavy, somewhat, negative = get_influential_features(importance_df)
    assert heavy == []
    assert somewhat == []
    assert negative == []

    heavy, somewhat, negative = get_influential_features(
        importance_df,
        min_importance_threshold=0.01,
    )
    assert heavy == []
    assert somewhat == [0, 1, 2, 3, 4]
    assert negative == []


def test_get_influential_features_higher_min_importance_threshold():
    importance_df = pd.DataFrame(
        {"feature": np.arange(5), "importance": [0.55, 0.201, 0.15, 0.075, 0.024]},
    )
    heavy, somewhat, negative = get_influential_features(importance_df)
    assert heavy == [0, 1]
    assert somewhat == [2, 3]
    assert negative == []

    heavy, somewhat, negative = get_influential_features(
        importance_df,
        min_importance_threshold=0.15,
    )
    assert heavy == [0]
    assert somewhat == [1, 2]
    assert negative == []


def test_get_influential_features_heavy_threshold():
    importance_df = pd.DataFrame(
        {"feature": np.arange(5), "importance": [0.52, 0.2, 0.15, 0.09, 0.04]},
    )
    heavy, somewhat, _ = get_influential_features(importance_df)
    assert heavy == [0, 1]
    assert somewhat == [2, 3]

    # Lowering the min threshold doesn't lower the heavy threshold
    heavy, somewhat, _ = get_influential_features(
        importance_df,
        min_importance_threshold=0.01,
    )
    assert heavy == [0, 1]
    assert somewhat == [2, 3, 4]

    # Raising the threshold a little won't change the heavy threshold
    heavy, somewhat, _ = get_influential_features(
        importance_df,
        min_importance_threshold=0.1,
    )
    assert heavy == [0, 1]
    assert somewhat == [2]

    # Raising the threshold when there's a conflict will change the heavy threshold
    heavy, somewhat, _ = get_influential_features(
        importance_df,
        min_importance_threshold=0.2,
    )
    assert heavy == [0]
    assert somewhat == [1]


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
        },
    )
    heavy, somewhat, negative = get_influential_features(
        importance_df,
        linear_importance=True,
    )
    assert heavy == ["heavy influence", "negative influence"]
    assert somewhat == ["somewhat influence"]
    assert negative == []


def test_get_influential_features_on_boundaries():
    importance_df = pd.DataFrame(
        {
            "feature": ["heavy 1", "heavy 2", "heavy 3", "somewhat 1", "somewhat 2"],
            "importance": [0.5, 0.2, 0.2, 0.05, 0.05],
        },
    )
    heavy, somewhat, negative = get_influential_features(importance_df)
    assert heavy == ["heavy 1", "heavy 2", "heavy 3"]
    assert somewhat == ["somewhat 1", "somewhat 2"]
    assert negative == []


def test_readable_explanation_not_fitted(elasticnet_component_graph):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)

    with pytest.raises(
        ValueError,
        match="Pipelines must be fitted in order to run feature explanations",
    ):
        readable_explanation(pipeline)

    with pytest.raises(
        ValueError,
        match="Pipelines must be fitted in order to run feature explanations",
    ):
        readable_explanation(pipeline, importance_method="feature")


def test_readable_explanation_missing_X_y(elasticnet_component_graph, fraud_100):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    X, y = fraud_100
    pipeline._is_fitted = True

    with pytest.raises(
        ValueError,
        match="required parameters for explaining pipelines",
    ):
        readable_explanation(pipeline)

    with pytest.raises(
        ValueError,
        match="required parameters for explaining pipelines",
    ):
        readable_explanation(pipeline, X)

    with pytest.raises(
        ValueError,
        match="required parameters for explaining pipelines",
    ):
        readable_explanation(pipeline, y=y)


def test_readable_explanation_invalid_importance(elasticnet_component_graph):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    pipeline._is_fitted = True

    with pytest.raises(ValueError, match="Unknown importance method"):
        readable_explanation(pipeline, importance_method="fake")


def test_readable_explanation_invalid_min_threshold(elasticnet_component_graph):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    pipeline._is_fitted = True

    with pytest.raises(
        ValueError,
        match="minimum importance threshold must be a percentage",
    ):
        readable_explanation(pipeline, min_importance_threshold=-1)

    with pytest.raises(
        ValueError,
        match="minimum importance threshold must be a percentage",
    ):
        readable_explanation(pipeline, min_importance_threshold=2)


@patch(
    "evalml.model_understanding.feature_explanations.calculate_permutation_importance",
)
def test_readable_explanation_permutation(
    mock_permutation_importance,
    caplog,
    elasticnet_component_graph,
    fraud_100,
):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    X, y = fraud_100
    pipeline._is_fitted = True

    mock_permutation_importance.return_value = pd.DataFrame(
        {
            "feature": ["lng", "customer_present", "lat", "card_id"],
            "importance": [0.55, -0.1, -0.2, -0.3],
        },
    )
    readable_explanation(pipeline, X, y, importance_method="permutation")

    out = caplog.text
    expected_influence = "Elastic Net Classifier: The prediction of fraud as measured by log loss binary is heavily influenced by lng."
    expected_negative = "The features customer_present, lat, and card_id detracted from model performance. We suggest removing these features."

    assert expected_influence in out
    assert expected_negative in out
    caplog.clear()


@patch(
    "evalml.model_understanding.feature_explanations.calculate_permutation_importance",
)
def test_readable_explanation_different_objective(
    mock_permutation_importance,
    caplog,
    elasticnet_component_graph,
    fraud_100,
):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    X, y = fraud_100
    pipeline._is_fitted = True

    mock_permutation_importance.return_value = pd.DataFrame(
        {"feature": [], "importance": []},
    )
    readable_explanation(pipeline, X, y, objective="precision")

    out = caplog.text
    expected_output = "The prediction of fraud as measured by precision"

    assert expected_output in out
    caplog.clear()


def test_readable_explanation_feature(caplog, elasticnet_component_graph, fraud_100):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    X, y = fraud_100
    pipeline.fit(X, y)

    readable_explanation(pipeline, importance_method="feature")

    out = caplog.text
    expected_heavy = "Elastic Net Classifier: The output is heavily influenced by datetime_year and store_id"
    expected_somewhat = "somewhat influenced by card_id."

    assert expected_heavy in out
    assert expected_somewhat in out
    assert "detracted from model performance" not in out
    caplog.clear()


@patch(
    "evalml.model_understanding.feature_explanations.calculate_permutation_importance",
)
@patch("evalml.pipelines.PipelineBase.feature_importance", new_callable=PropertyMock)
def test_readable_explanation_sentence_beginning(
    mock_feature_importance,
    mock_permutation_importance,
    caplog,
    elasticnet_component_graph,
    fraud_100,
):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    X, y = fraud_100
    pipeline._is_fitted = True

    mock_permutation_importance.return_value = pd.DataFrame(
        {"feature": [], "importance": []},
    )
    mock_feature_importance.return_value = pd.DataFrame(
        {"feature": [], "importance": []},
    )

    # Objective is not None, target is not None
    readable_explanation(pipeline, X, y, importance_method="permutation")
    expected = f"The prediction of {y.name} as measured by log loss binary is"
    out = caplog.text
    assert expected in out
    caplog.clear()

    # Objective is None, target is not None
    readable_explanation(pipeline, X, y, importance_method="feature")
    expected = f"The prediction of {y.name} is"
    out = caplog.text
    assert expected in out
    caplog.clear()

    # Objective is not None, target is None
    y.name = None
    readable_explanation(pipeline, X, y, importance_method="permutation")
    expected = "The output as measured by log loss binary is"
    out = caplog.text
    assert expected in out
    caplog.clear()

    # Objective is None, target is None
    readable_explanation(pipeline, importance_method="feature")
    expected = "The output is"
    out = caplog.text
    assert expected in out
    caplog.clear()


@patch("evalml.pipelines.PipelineBase.feature_importance", new_callable=PropertyMock)
def test_readable_explanation_somewhat_important_features(
    mock_feature_importance,
    elasticnet_component_graph,
    caplog,
):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    pipeline._is_fitted = True
    mock_feature_importance.return_value = pd.DataFrame(
        {
            "feature": ["heavy 1", "somewhat 1", "somewhat 2", "somewhat 3"],
            "importance": [0.55, 0.17, 0.15, 0.13],
        },
    )

    readable_explanation(pipeline, importance_method="feature")
    expected_somewhat = (
        "is somewhat influenced by somewhat 1, somewhat 2, and somewhat 3."
    )
    out = caplog.text
    assert expected_somewhat in out
    caplog.clear()

    mock_feature_importance.return_value = pd.DataFrame(
        {
            "feature": ["heavy 1", "heavy 2", "heavy 3", "somewhat 1"],
            "importance": [0.32, 0.3, 0.23, 0.15],
        },
    )
    readable_explanation(pipeline, importance_method="feature")
    expected_somewhat = "is somewhat influenced by somewhat 1."
    out = caplog.text
    assert expected_somewhat in out
    caplog.clear()


@patch(
    "evalml.model_understanding.feature_explanations.calculate_permutation_importance",
)
def test_readable_explanation_detrimental_features(
    mock_permutation_importance,
    caplog,
    elasticnet_component_graph,
    fraud_100,
):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    X, y = fraud_100
    pipeline._is_fitted = True
    mock_permutation_importance.return_value = pd.DataFrame(
        {
            "feature": [
                "heavy 1",
                "somewhat 1",
                "somewhat 2",
                "somewhat 3",
                "detrimental 1",
            ],
            "importance": [0.55, 0.17, 0.15, 0.13, -0.1],
        },
    )
    readable_explanation(pipeline, X, y)
    expected_detrimental = "The feature detrimental 1 detracted from model performance. We suggest removing this feature"
    out = caplog.text
    assert expected_detrimental in out
    caplog.clear()

    mock_permutation_importance.return_value = pd.DataFrame(
        {
            "feature": ["heavy 1", "somewhat 1", "detrimental 1", "detrimental 2"],
            "importance": [0.55, 0.13, -0.1, -0.25],
        },
    )
    readable_explanation(pipeline, X, y)
    expected_detrimental = "The features detrimental 1 and detrimental 2 detracted from model performance. We suggest removing these features"
    out = caplog.text
    assert expected_detrimental in out
    caplog.clear()


@patch("evalml.pipelines.PipelineBase.feature_importance", new_callable=PropertyMock)
def test_readable_explanation_neither_heavy_somewhat(
    mock_feature_importance,
    elasticnet_component_graph,
    caplog,
):
    pipeline = BinaryClassificationPipeline(elasticnet_component_graph)
    pipeline._is_fitted = True
    mock_feature_importance.return_value = pd.DataFrame(
        {"feature": np.arange(100), "importance": [0.1] * 25 + [0.05] * 75},
    )

    readable_explanation(pipeline, importance_method="feature")
    out = caplog.text
    expected_neither = "is not strongly influenced by any single feature. Lower the `min_importance_threshold` to see more"
    assert expected_neither in out
