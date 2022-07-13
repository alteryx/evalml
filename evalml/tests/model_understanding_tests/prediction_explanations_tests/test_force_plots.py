from itertools import product
from unittest.mock import patch

import pandas as pd
import pytest
import shap

from evalml.model_understanding.force_plots import force_plot, graph_force_plot
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
)
from evalml.tests.model_understanding_tests.prediction_explanations_tests.test_explainers import (
    pipeline_test_cases,
    transform_y_for_problem_type,
)


def validate_plot_feature_values(results, X):
    """Helper to validate feature values returned from force plots"""
    for row, result in enumerate(results):
        for result_label in result:
            # Check feature values in generated force plot correspond to input rows
            row_force_plot = result[result_label]["plot"]
            assert "features" in row_force_plot.data
            plot_features = row_force_plot.data["features"]
            feature_vals = [plot_features[k]["value"] for k in plot_features]

            # Features in results depend on effect size; filter feature names
            effect_features_ix = plot_features.keys()
            effect_features = [
                row_force_plot.data["featureNames"][i] for i in effect_features_ix
            ]

            assert all(feature_vals == X[effect_features].iloc[row].values)


def test_exceptions():
    with pytest.raises(
        TypeError,
        match="rows_to_explain should be provided as a list of row index integers!",
    ):
        force_plot(None, None, None, None)
    with pytest.raises(
        TypeError,
        match="rows_to_explain should only contain integers!",
    ):
        force_plot(
            pipeline=None,
            rows_to_explain=["this", "party's", "over"],
            training_data=None,
            y=None,
        )


@patch("evalml.model_understanding.force_plots.jupyter_check")
@patch("evalml.model_understanding.force_plots.initjs")
@pytest.mark.parametrize(
    "rows_to_explain, just_data",
    product([[0], [0, 1, 2, 3, 4]], [False, True]),
)
def test_force_plot_binary(
    initjs,
    jupyter_check,
    rows_to_explain,
    just_data,
    breast_cancer_local,
):
    X, y = breast_cancer_local

    pipeline = BinaryClassificationPipeline(
        component_graph=["Simple Imputer", "Random Forest Classifier"],
    )
    pipeline.fit(X, y)

    if just_data:
        results = force_plot(pipeline, rows_to_explain, X, y)
    else:
        # Code chunk to test where initjs is called if jupyter is recognized
        jupyter_check.return_value = False
        with pytest.warns(None) as graph_valid:
            results = graph_force_plot(
                pipeline,
                rows_to_explain=rows_to_explain,
                training_data=X,
                y=y,
                matplotlib=False,
            )
            assert not initjs.called
            warnings = set([str(gv) for gv in graph_valid.list])
            assert all(["DeprecationWarning" in w for w in warnings])

        jupyter_check.return_value = True
        with pytest.warns(None) as graph_valid:
            results = graph_force_plot(
                pipeline,
                rows_to_explain=rows_to_explain,
                training_data=X,
                y=y,
                matplotlib=False,
            )
            assert initjs.called
            warnings = set([str(gv) for gv in graph_valid.list])
            assert all(["DeprecationWarning" in w for w in warnings])

    # Should have a result per row to explain.
    assert len(results) == len(rows_to_explain)

    expected_class_labels = {"malignant"}

    for result in results:
        class_labels = result.keys()
        assert (
            len(class_labels) == 1
        )  # Binary classification tends to only return results for positive class
        assert set(class_labels) == expected_class_labels
        for class_label in expected_class_labels:
            assert {"expected_value", "feature_names", "shap_values"}.issubset(
                set(result[class_label].keys()),
            )

    if not just_data:
        # Should have a force plot for each row result.
        for result in results:
            for class_label in expected_class_labels:
                assert isinstance(
                    result[class_label]["plot"],
                    shap.plots._force.AdditiveForceVisualizer,
                )

        validate_plot_feature_values(results, X)


@pytest.mark.parametrize(
    "rows_to_explain, just_data",
    product([[0], [0, 1, 2, 3, 4]], [False, True]),
)
def test_force_plot_multiclass(rows_to_explain, just_data, wine_local):
    X, y = wine_local

    pipeline = MulticlassClassificationPipeline(
        component_graph=["Simple Imputer", "Random Forest Classifier"],
    )
    pipeline.fit(X, y)

    if just_data:
        results = force_plot(pipeline, rows_to_explain, X, y)
    else:
        results = graph_force_plot(
            pipeline,
            rows_to_explain=rows_to_explain,
            training_data=X,
            y=y,
            matplotlib=False,
        )

    # Should have a result per row to explain.
    assert len(results) == len(rows_to_explain)

    expected_class_labels = {"class_0", "class_1", "class_2"}

    for result in results:
        class_labels = result.keys()
        assert len(class_labels) == len(set(y))
        assert set(class_labels) == expected_class_labels
        for class_label in expected_class_labels:
            assert {"expected_value", "feature_names", "shap_values"}.issubset(
                set(result[class_label].keys()),
            )

    if not just_data:
        # Should have a force plot for each row result.
        for result in results:
            for class_label in expected_class_labels:
                assert isinstance(
                    result[class_label]["plot"],
                    shap.plots._force.AdditiveForceVisualizer,
                )

        validate_plot_feature_values(results, X)


@pytest.mark.parametrize(
    "rows_to_explain, just_data",
    product([[0], [0, 1, 2, 3, 4]], [False, True]),
)
def test_force_plot_regression(rows_to_explain, just_data, X_y_regression):
    X, y = X_y_regression
    X = pd.DataFrame(X)
    y = pd.Series(y)

    pipeline = RegressionPipeline(
        component_graph=["Simple Imputer", "LightGBM Regressor"],
    )
    pipeline.fit(X, y)

    if just_data:
        results = force_plot(pipeline, rows_to_explain, X, y)
    else:
        results = graph_force_plot(
            pipeline,
            rows_to_explain=rows_to_explain,
            training_data=X,
            y=y,
            matplotlib=False,
        )

    # Should have a result per row to explain.
    assert len(results) == len(rows_to_explain)

    for result in results:
        class_labels = result.keys()
        assert len(class_labels) == 1
        assert "regression" in class_labels
        assert {"expected_value", "feature_names", "shap_values"}.issubset(
            set(result["regression"].keys()),
        )

    if not just_data:
        # Should have a force plot for each row result.
        for result in results:
            assert isinstance(
                result["regression"]["plot"],
                shap.plots._force.AdditiveForceVisualizer,
            )
        validate_plot_feature_values(results, X)


@pytest.mark.parametrize("pipeline_class,estimator", pipeline_test_cases)
def test_categories_aggregated_date_ohe(pipeline_class, estimator, fraud_100):
    X, y = fraud_100
    columns_to_select = ["datetime", "amount", "provider", "currency"]

    pipeline = pipeline_class(
        component_graph=[
            "Select Columns Transformer",
            "DateTime Featurizer",
            "One Hot Encoder",
            estimator,
        ],
        parameters={
            "Select Columns Transformer": {"columns": columns_to_select},
            "DateTime Featurizer": {"encode_as_categories": True},
            estimator: {"n_jobs": 1},
        },
    )
    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)

    force_plots_dict = force_plot(pipeline, [0], X, y)
    for plot in force_plots_dict:
        for cls in plot:
            assert set(plot[cls]["feature_names"]) == set(columns_to_select)

    force_plots = graph_force_plot(pipeline, [0, 1, 2], X, y)
    for plot in force_plots:
        for cls in plot:
            assert set(plot[cls]["feature_names"]) == set(columns_to_select)


@pytest.mark.parametrize("pipeline_class,estimator", pipeline_test_cases)
def test_categories_aggregated_text(pipeline_class, estimator, fraud_100):
    X, y = fraud_100
    columns_to_select = ["datetime", "amount", "provider", "currency"]
    X.ww.init(logical_types={"currency": "categorical"})

    X.ww.set_types(logical_types={"provider": "NaturalLanguage"})
    component_graph = [
        "Select Columns Transformer",
        "One Hot Encoder",
        "Natural Language Featurizer",
        "DateTime Featurizer",
        estimator,
    ]

    pipeline = pipeline_class(
        component_graph,
        parameters={
            "Select Columns Transformer": {
                "columns": ["amount", "provider", "currency", "datetime"],
            },
            estimator: {"n_jobs": 1},
        },
    )

    y = transform_y_for_problem_type(pipeline.problem_type, y)

    pipeline.fit(X, y)

    force_plots_dict = force_plot(pipeline, [0], X, y)
    for plot in force_plots_dict:
        for cls in plot:
            assert set(plot[cls]["feature_names"]) == set(columns_to_select)

    force_plots = graph_force_plot(pipeline, [0, 1, 2], X, y)
    for plot in force_plots:
        for cls in plot:
            assert set(plot[cls]["feature_names"]) == set(columns_to_select)
