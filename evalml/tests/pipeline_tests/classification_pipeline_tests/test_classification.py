from itertools import product

import pandas as pd
import pytest


@pytest.mark.parametrize("problem_type", ["binary", "multi"])
def test_new_unique_targets_in_score(
    X_y_binary,
    logistic_regression_binary_pipeline_class,
    X_y_multi,
    logistic_regression_multiclass_pipeline_class,
    problem_type,
):
    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        objective = "Log Loss Binary"
    elif problem_type == "multi":
        X, y = X_y_multi
        pipeline = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        objective = "Log Loss Multiclass"
    pipeline.fit(X, y)
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        pipeline.score(X, pd.Series([4] * len(y)), [objective])


@pytest.mark.parametrize(
    "problem_type,use_ints", product(["binary", "multi"], [True, False])
)
def test_pipeline_has_classes_property(
    breast_cancer_local,
    wine_local,
    logistic_regression_binary_pipeline_class,
    logistic_regression_multiclass_pipeline_class,
    problem_type,
    use_ints,
):
    if problem_type == "binary":
        X, y = breast_cancer_local
        pipeline = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        if use_ints:
            y = y.map({"malignant": 0, "benign": 1})
            answer = [0, 1]
        else:
            answer = ["benign", "malignant"]
    elif problem_type == "multi":
        X, y = wine_local
        pipeline = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        if use_ints:
            y = y.map({"class_0": 0, "class_1": 1, "class_2": 2})
            answer = [0, 1, 2]
        else:
            answer = ["class_0", "class_1", "class_2"]

    # Check that .classes_ is None before fitting
    assert pipeline.classes_ is None

    pipeline.fit(X, y)
    assert pipeline.classes_ == answer


def test_woodwork_classification_pipeline(
    breast_cancer_local, logistic_regression_binary_pipeline_class
):
    X, y = breast_cancer_local
    mock_pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    mock_pipeline.fit(X, y)
    assert not pd.isnull(mock_pipeline.predict(X)).any()
    assert not pd.isnull(mock_pipeline.predict_proba(X)).any().any()


# @pytest.mark.parametrize(
#     "index",
#     [
#         list(range(-5, 0)),
#         list(range(100, 105)),
#         [f"row_{i}" for i in range(5)],
#         pd.date_range("2020-09-08", periods=5),
#     ],
# )
# @pytest.mark.parametrize("with_estimator_last_component", [True, False])
# def test_pipeline_transform_and_predict_with_custom_index(
#     index,
#     with_estimator_last_component,
#     example_graph,
#     example_graph_with_transformer_last_component,
# ):
#     X = pd.DataFrame(
#         {"categories": [f"cat_{i}" for i in range(5)], "numbers": np.arange(5)},
#         index=index,
#     )
#     y = pd.Series([1, 2, 1, 2, 1], index=index)
#     X.ww.init(logical_types={"categories": "categorical"})

#     graph_to_use = (
#         example_graph
#         if with_estimator_last_component
#         else example_graph_with_transformer_last_component
#     )
#     component_graph = ComponentGraph(graph_to_use)
#     component_graph.instantiate()
#     component_graph.fit(X, y)

#     if with_estimator_last_component:
#         predictions = component_graph.predict(X)
#         assert_index_equal(predictions.index, X.index)
#         assert not predictions.isna().any(axis=None)
#     else:
#         X_t = component_graph.transform(X)
#         assert_index_equal(X_t.index, X.index)
#         assert not X_t.isna().any(axis=None)

#         y_in = pd.Series([0, 1, 0, 1, 0], index=index)
#         y_inv = component_graph.inverse_transform(y_in)
#         assert_index_equal(y_inv.index, y.index)
#         assert not y_inv.isna().any(axis=None)
