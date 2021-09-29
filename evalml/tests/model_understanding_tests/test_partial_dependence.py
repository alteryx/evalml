import re

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.exceptions import (
    NullsInColumnWarning,
    PartialDependenceError,
    PartialDependenceErrorCode,
)
from evalml.model_understanding import (
    graph_partial_dependence,
    partial_dependence,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    ClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
)
from evalml.pipelines.utils import _make_stacked_ensemble_pipeline
from evalml.problem_types import ProblemTypes


@pytest.fixture
def test_pipeline():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = [
            "Simple Imputer",
            "One Hot Encoder",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ]

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters)

        @property
        def feature_importance(self):
            importance = [1.0, 0.2, 0.0002, 0.0, 0.0, -1.0]
            feature_names = range(len(importance))
            f_i = list(zip(feature_names, importance))
            df = pd.DataFrame(f_i, columns=["feature", "importance"])
            return df

    return TestPipeline(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})


def check_partial_dependence_dataframe(pipeline, part_dep, grid_size=5):
    columns = ["feature_values", "partial_dependence"]
    if isinstance(pipeline, ClassificationPipeline):
        columns.append("class_label")
    n_rows_for_class = (
        len(pipeline.classes_)
        if isinstance(pipeline, MulticlassClassificationPipeline)
        else 1
    )
    assert list(part_dep.columns) == columns
    assert len(part_dep["partial_dependence"]) == grid_size * n_rows_for_class
    assert len(part_dep["feature_values"]) == grid_size * n_rows_for_class
    if isinstance(pipeline, ClassificationPipeline):
        per_class_counts = part_dep["class_label"].value_counts()
        assert all(value == grid_size for value in per_class_counts.values)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_partial_dependence_problem_types(
    data_type,
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    logistic_regression_binary_pipeline_class,
    logistic_regression_multiclass_pipeline_class,
    linear_regression_pipeline_class,
    make_data_type,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        pipeline = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        pipeline = linear_regression_pipeline_class(
            parameters={"Linear Regressor": {"n_jobs": 1}}
        )

    X = make_data_type(data_type, X)
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features=0, grid_resolution=5)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().any(axis=None)


def test_partial_dependence_string_feature_name(
    breast_cancer_local,
    logistic_regression_binary_pipeline_class,
):
    X, y = breast_cancer_local
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)
    part_dep = partial_dependence(
        pipeline, X, features="mean radius", grid_resolution=5
    )
    assert list(part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]
    assert len(part_dep["partial_dependence"]) == 5
    assert len(part_dep["feature_values"]) == 5
    assert not part_dep.isnull().any(axis=None)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_partial_dependence_with_non_numeric_columns(
    data_type,
    linear_regression_pipeline_class,
    logistic_regression_binary_pipeline_class,
):
    X = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 0] * 4,
            "also numeric": [2, 3, 4, 1] * 4,
            "string": ["a", "b", "a", "c"] * 4,
            "also string": ["c", "b", "a", "c"] * 4,
        }
    )
    if data_type == "ww":
        X.ww.init()
    y = [0, 0.2, 1.4, 1] * 4
    pipeline = linear_regression_pipeline_class(
        parameters={"Linear Regressor": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features="numeric")
    assert list(part_dep.columns) == ["feature_values", "partial_dependence"]
    assert len(part_dep["partial_dependence"]) == 4
    assert len(part_dep["feature_values"]) == 4
    assert not part_dep.isnull().any(axis=None)

    part_dep = partial_dependence(pipeline, X, features="string")
    assert list(part_dep.columns) == ["feature_values", "partial_dependence"]
    assert len(part_dep["partial_dependence"]) == 3
    assert len(part_dep["feature_values"]) == 3
    assert not part_dep.isnull().any(axis=None)


def test_partial_dependence_baseline():
    X = pd.DataFrame([[1, 0], [0, 1]])
    y = pd.Series([0, 1])
    pipeline = BinaryClassificationPipeline(
        component_graph=["Baseline Classifier"], parameters={}
    )
    pipeline.fit(X, y)
    with pytest.raises(
        PartialDependenceError,
        match="Partial dependence plots are not supported for Baseline pipelines",
    ) as e:
        partial_dependence(pipeline, X, features=0, grid_resolution=5)
    assert e.value.code == PartialDependenceErrorCode.PIPELINE_IS_BASELINE


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_partial_dependence_catboost(
    problem_type, X_y_binary, X_y_multi, has_minimal_dependencies
):
    if not has_minimal_dependencies:

        if problem_type == ProblemTypes.BINARY:
            X, y = X_y_binary
            y_small = ["a", "b", "a"] * 5
            pipeline_class = BinaryClassificationPipeline
        else:
            X, y = X_y_multi
            y_small = ["a", "b", "c"] * 5
            pipeline_class = MulticlassClassificationPipeline

        pipeline = pipeline_class(
            component_graph=["CatBoost Classifier"],
            parameters={"CatBoost Classifier": {"thread_count": 1}},
        )
        pipeline.fit(X, y)
        part_dep = partial_dependence(pipeline, X, features=0, grid_resolution=5)
        check_partial_dependence_dataframe(pipeline, part_dep)
        assert not part_dep.isnull().all().all()

        # test that CatBoost can natively handle non-numerical columns as feature passed to partial_dependence
        X = pd.DataFrame(
            {
                "numeric": [1, 2, 3] * 5,
                "also numeric": [2, 3, 4] * 5,
                "string": ["a", "b", "c"] * 5,
                "also string": ["c", "b", "a"] * 5,
            }
        )
        pipeline = pipeline_class(
            component_graph=["CatBoost Classifier"],
            parameters={"CatBoost Classifier": {"thread_count": 1}},
        )
        pipeline.fit(X, y_small)
        part_dep = partial_dependence(pipeline, X, features="string")
        check_partial_dependence_dataframe(pipeline, part_dep, grid_size=3)
        assert not part_dep.isnull().all().all()


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_partial_dependence_xgboost_feature_names(
    problem_type, has_minimal_dependencies, X_y_binary, X_y_multi, X_y_regression
):
    if has_minimal_dependencies:
        pytest.skip("Skipping because XGBoost not installed for minimal dependencies")
    if problem_type == ProblemTypes.REGRESSION:
        pipeline = RegressionPipeline(
            component_graph=["Simple Imputer", "XGBoost Regressor"],
            parameters={"XGBoost Classifier": {"nthread": 1}},
        )
        X, y = X_y_regression
    elif problem_type == ProblemTypes.BINARY:
        pipeline = BinaryClassificationPipeline(
            component_graph=["Simple Imputer", "XGBoost Classifier"],
            parameters={"XGBoost Classifier": {"nthread": 1}},
        )
        X, y = X_y_binary
    elif problem_type == ProblemTypes.MULTICLASS:
        pipeline = MulticlassClassificationPipeline(
            component_graph=["Simple Imputer", "XGBoost Classifier"],
            parameters={"XGBoost Classifier": {"nthread": 1}},
        )
        X, y = X_y_multi

    X = pd.DataFrame(X)
    X = X.rename(columns={0: "<[0]"})
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features="<[0]", grid_resolution=5)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().all().all()

    part_dep = partial_dependence(pipeline, X, features=1, grid_resolution=5)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().all().all()


def test_partial_dependence_multiclass(
    wine_local, logistic_regression_multiclass_pipeline_class
):
    X, y = wine_local
    pipeline = logistic_regression_multiclass_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)

    num_classes = y.nunique()
    grid_resolution = 5

    one_way_part_dep = partial_dependence(
        pipeline=pipeline, X=X, features="magnesium", grid_resolution=grid_resolution
    )
    assert "class_label" in one_way_part_dep.columns
    assert one_way_part_dep["class_label"].nunique() == num_classes
    assert len(one_way_part_dep.index) == num_classes * grid_resolution
    assert list(one_way_part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]

    two_way_part_dep = partial_dependence(
        pipeline=pipeline,
        X=X,
        features=("magnesium", "alcohol"),
        grid_resolution=grid_resolution,
    )

    assert "class_label" in two_way_part_dep.columns
    assert two_way_part_dep["class_label"].nunique() == num_classes
    assert len(two_way_part_dep.index) == num_classes * grid_resolution
    assert len(two_way_part_dep.columns) == grid_resolution + 1

    two_way_part_dep, two_way_ice_data = partial_dependence(
        pipeline=pipeline,
        X=X,
        features=("magnesium", "alcohol"),
        grid_resolution=grid_resolution,
        kind="both",
    )

    assert "class_label" in two_way_part_dep.columns
    assert two_way_part_dep["class_label"].nunique() == num_classes
    assert len(two_way_part_dep.index) == num_classes * grid_resolution
    assert len(two_way_part_dep.columns) == grid_resolution + 1

    assert len(two_way_ice_data) == len(X)
    for ind_data in two_way_ice_data:
        assert "class_label" in ind_data.columns
        assert two_way_part_dep["class_label"].nunique() == num_classes
        assert len(two_way_part_dep.index) == num_classes * grid_resolution
        assert len(two_way_part_dep.columns) == grid_resolution + 1


def test_partial_dependence_multiclass_numeric_labels(
    logistic_regression_multiclass_pipeline_class, X_y_multi
):
    X, y = X_y_multi
    X = pd.DataFrame(X)
    y = pd.Series(y, dtype="int64")
    pipeline = logistic_regression_multiclass_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)

    num_classes = y.nunique()
    grid_resolution = 5

    one_way_part_dep = partial_dependence(
        pipeline=pipeline, X=X, features=1, grid_resolution=grid_resolution
    )
    assert "class_label" in one_way_part_dep.columns
    assert one_way_part_dep["class_label"].nunique() == num_classes
    assert len(one_way_part_dep.index) == num_classes * grid_resolution
    assert list(one_way_part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]

    two_way_part_dep = partial_dependence(
        pipeline=pipeline,
        X=X,
        features=(1, 2),
        grid_resolution=grid_resolution,
    )

    assert "class_label" in two_way_part_dep.columns
    assert two_way_part_dep["class_label"].nunique() == num_classes
    assert len(two_way_part_dep.index) == num_classes * grid_resolution
    assert len(two_way_part_dep.columns) == grid_resolution + 1


def test_partial_dependence_not_fitted(
    X_y_binary, logistic_regression_binary_pipeline_class
):
    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    with pytest.raises(
        PartialDependenceError,
        match="Pipeline to calculate partial dependence for must be fitted",
    ) as e:
        partial_dependence(pipeline, X, features=0, grid_resolution=5)
    assert e.value.code == PartialDependenceErrorCode.UNFITTED_PIPELINE


def test_partial_dependence_warning(logistic_regression_binary_pipeline_class):
    X = pd.DataFrame({"a": [1, 2, None, 2, 2], "b": [1, 1, 2, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)
    with pytest.warns(
        NullsInColumnWarning,
        match="There are null values in the features, which will cause NaN values in the partial dependence output",
    ):
        partial_dependence(pipeline, X, features=0, grid_resolution=5)
    with pytest.warns(
        NullsInColumnWarning,
        match="There are null values in the features, which will cause NaN values in the partial dependence output",
    ):
        partial_dependence(pipeline, X, features=("a", "b"), grid_resolution=5)
    with pytest.warns(
        NullsInColumnWarning,
        match="There are null values in the features, which will cause NaN values in the partial dependence output",
    ):
        partial_dependence(pipeline, X, features="a", grid_resolution=5)


def test_partial_dependence_errors(logistic_regression_binary_pipeline_class):
    X = pd.DataFrame({"a": [2, None, 2, 2], "b": [1, 2, 2, 1], "c": [0, 0, 0, 0]})
    y = pd.Series([0, 1, 0, 1])
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)

    with pytest.raises(
        PartialDependenceError,
        match="Too many features given to graph_partial_dependence.  Only one or two-way partial dependence is supported.",
    ) as e:
        partial_dependence(pipeline, X, features=("a", "b", "c"), grid_resolution=5)
    assert e.value.code == PartialDependenceErrorCode.TOO_MANY_FEATURES

    with pytest.raises(
        PartialDependenceError,
        match="Features provided must be a tuple entirely of integers or strings, not a mixture of both.",
    ) as e:
        partial_dependence(pipeline, X, features=(0, "b"))
    assert e.value.code == PartialDependenceErrorCode.FEATURES_ARGUMENT_INCORRECT_TYPES


def test_partial_dependence_more_categories_than_grid_resolution(
    fraud_local,
    logistic_regression_binary_pipeline_class,
):
    def round_dict_keys(dictionary, places=6):
        """Function to round all keys of a dictionary that has floats as keys."""
        dictionary_rounded = {}
        for key in dictionary:
            dictionary_rounded[round(key, places)] = dictionary[key]
        return dictionary_rounded

    X, y = fraud_local
    X = X[:100]
    y = y[:100]
    X = X.drop(columns=["datetime", "expiration_date", "country", "region", "provider"])
    pipeline = logistic_regression_binary_pipeline_class({})
    pipeline.fit(X, y)
    num_cat_features = len(set(X["currency"]))
    assert num_cat_features == 73

    part_dep_ans = {
        0.05824028901694482: 63,
        0.1349235160940143: 1,
        0.6353030372157324: 1,
        0.031171284274810262: 1,
        0.009093086236362272: 1,
        0.33547688991040336: 1,
        0.01746660818843149: 1,
        0.018205973481273202: 1,
        0.2876661156872482: 1,
        0.015320197702897345: 1,
        0.2821023107719306: 1,
    }

    part_dep_ans_rounded = round_dict_keys(part_dep_ans)

    # Check the case where grid_resolution < number of categorical features
    part_dep = partial_dependence(
        pipeline, X, "currency", grid_resolution=round(num_cat_features / 2)
    )
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)

    # Check the case where grid_resolution == number of categorical features
    part_dep = partial_dependence(
        pipeline, X, "currency", grid_resolution=round(num_cat_features)
    )
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)

    # Check the case where grid_resolution > number of categorical features
    part_dep = partial_dependence(
        pipeline, X, "currency", grid_resolution=round(num_cat_features * 2)
    )
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)


def test_partial_dependence_ice_plot(logistic_regression_binary_pipeline_class):
    X = pd.DataFrame({"a": [1, 2, None, 2, 2], "b": [1, 1, 2, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)

    avg_pred, ind_preds = partial_dependence(pipeline, X, features="a", kind="both")
    assert isinstance(avg_pred, pd.DataFrame)
    assert isinstance(ind_preds, pd.DataFrame)

    assert avg_pred.shape == (3, 3)
    assert ind_preds.shape == (3, 7)

    ind_preds = partial_dependence(pipeline, X, features="b", kind="individual")
    assert isinstance(ind_preds, pd.DataFrame)

    assert ind_preds.shape == (2, 7)


def test_two_way_partial_dependence_ice_plot(logistic_regression_binary_pipeline_class):
    X = pd.DataFrame({"a": [1, 2, None, 2, 2], "b": [1, 1, 2, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)

    avg_pred, ind_preds = partial_dependence(
        pipeline, X, features=["a", "b"], grid_resolution=5, kind="both"
    )
    assert isinstance(avg_pred, pd.DataFrame)
    assert isinstance(ind_preds, list)
    assert isinstance(ind_preds[0], pd.DataFrame)

    assert avg_pred.shape == (3, 3)
    assert len(ind_preds) == 5
    for ind_df in ind_preds:
        assert ind_df.shape == (3, 3)

    ind_preds = partial_dependence(
        pipeline, X, features=["a", "b"], grid_resolution=5, kind="individual"
    )
    assert isinstance(ind_preds, list)
    assert isinstance(ind_preds[0], pd.DataFrame)

    assert len(ind_preds) == 5
    for ind_df in ind_preds:
        assert ind_df.shape == (3, 3)


@pytest.mark.parametrize("use_sklearn", [True, False])
@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.REGRESSION])
def test_partial_dependence_ensemble_pipeline(
    problem_type, use_sklearn, X_y_binary, X_y_regression
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        input_pipelines = [
            BinaryClassificationPipeline(["Random Forest Classifier"]),
            BinaryClassificationPipeline(["Elastic Net Classifier"]),
        ]
    else:
        X, y = X_y_regression
        input_pipelines = [
            RegressionPipeline(["Random Forest Regressor"]),
            RegressionPipeline(["Elastic Net Regressor"]),
        ]
    pipeline = _make_stacked_ensemble_pipeline(
        input_pipelines=input_pipelines,
        problem_type=problem_type,
        use_sklearn=use_sklearn,
    )
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features=0, grid_resolution=5)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().all().all()


def test_graph_partial_dependence(breast_cancer_local, test_pipeline):
    X, y = breast_cancer_local

    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_partial_dependence(clf, X, features="mean radius", grid_resolution=5)
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict["layout"]["title"]["text"] == "Partial Dependence of 'mean radius'"
    assert len(fig_dict["data"]) == 1
    assert fig_dict["data"][0]["name"] == "Partial Dependence"

    part_dep_data = partial_dependence(
        clf, X, features="mean radius", grid_resolution=5
    )
    assert np.array_equal(fig_dict["data"][0]["x"], part_dep_data["feature_values"])
    assert np.array_equal(
        fig_dict["data"][0]["y"], part_dep_data["partial_dependence"].values
    )


def test_graph_two_way_partial_dependence(breast_cancer_local, test_pipeline):
    X, y = breast_cancer_local

    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_partial_dependence(
        clf, X, features=("mean radius", "mean area"), grid_resolution=5
    )
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == "Partial Dependence of 'mean radius' vs. 'mean area'"
    )
    assert len(fig_dict["data"]) == 1
    assert fig_dict["data"][0]["name"] == "Partial Dependence"

    part_dep_data = partial_dependence(
        clf, X, features=("mean radius", "mean area"), grid_resolution=5
    )
    part_dep_data.drop(columns=["class_label"], inplace=True)
    assert np.array_equal(fig_dict["data"][0]["x"], part_dep_data.columns)
    assert np.array_equal(fig_dict["data"][0]["y"], part_dep_data.index)
    assert np.array_equal(fig_dict["data"][0]["z"], part_dep_data.values)


def test_graph_partial_dependence_multiclass(
    wine_local,
    logistic_regression_multiclass_pipeline_class,
):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )
    X, y = wine_local
    pipeline = logistic_regression_multiclass_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)

    # Test one-way without class labels
    fig_one_way_no_class_labels = graph_partial_dependence(
        pipeline, X, features="magnesium", grid_resolution=5
    )
    assert isinstance(fig_one_way_no_class_labels, go.Figure)
    fig_dict = fig_one_way_no_class_labels.to_dict()
    assert len(fig_dict["data"]) == len(pipeline.classes_)
    for data, label in zip(fig_dict["data"], pipeline.classes_):
        assert len(data["x"]) == 5
        assert len(data["y"]) == 5
        assert data["name"] == "Partial Dependence: " + label

    # Check that all the subplots axes have the same range
    for suplot_1_axis, suplot_2_axis in [
        ("axis2", "axis3"),
        ("axis2", "axis4"),
        ("axis3", "axis4"),
    ]:
        for axis_type in ["x", "y"]:
            assert (
                fig_dict["layout"][axis_type + suplot_1_axis]["range"]
                == fig_dict["layout"][axis_type + suplot_2_axis]["range"]
            )

    # Test one-way with class labels
    fig_one_way_class_labels = graph_partial_dependence(
        pipeline, X, features="magnesium", class_label="class_1", grid_resolution=5
    )
    assert isinstance(fig_one_way_class_labels, go.Figure)
    fig_dict = fig_one_way_class_labels.to_dict()
    assert len(fig_dict["data"]) == 1
    assert len(fig_dict["data"][0]["x"]) == 5
    assert len(fig_dict["data"][0]["y"]) == 5
    assert fig_dict["data"][0]["name"] == "Partial Dependence: class_1"

    msg = "Class wine is not one of the classes the pipeline was fit on: class_0, class_1, class_2"
    with pytest.raises(PartialDependenceError, match=msg) as e:
        graph_partial_dependence(pipeline, X, features="alcohol", class_label="wine")
    assert e.value.code == PartialDependenceErrorCode.INVALID_CLASS_LABEL

    # Test two-way without class labels
    fig_two_way_no_class_labels = graph_partial_dependence(
        pipeline, X, features=("magnesium", "alcohol"), grid_resolution=5
    )
    assert isinstance(fig_two_way_no_class_labels, go.Figure)
    fig_dict = fig_two_way_no_class_labels.to_dict()
    assert (
        len(fig_dict["data"]) == 3
    ), "Figure does not have partial dependence data for each class."
    assert all([len(fig_dict["data"][i]["x"]) == 5 for i in range(3)])
    assert all([len(fig_dict["data"][i]["y"]) == 5 for i in range(3)])
    assert [fig_dict["data"][i]["name"] for i in range(3)] == [
        "class_0",
        "class_1",
        "class_2",
    ]

    # Check that all the subplots axes have the same range
    for suplot_1_axis, suplot_2_axis in [
        ("axis", "axis2"),
        ("axis", "axis3"),
        ("axis2", "axis3"),
    ]:
        for axis_type in ["x", "y"]:
            assert (
                fig_dict["layout"][axis_type + suplot_1_axis]["range"]
                == fig_dict["layout"][axis_type + suplot_2_axis]["range"]
            )

    # Test two-way with class labels
    fig_two_way_class_labels = graph_partial_dependence(
        pipeline,
        X,
        features=("magnesium", "alcohol"),
        class_label="class_1",
        grid_resolution=5,
    )
    assert isinstance(fig_two_way_class_labels, go.Figure)
    fig_dict = fig_two_way_class_labels.to_dict()
    assert len(fig_dict["data"]) == 1
    assert len(fig_dict["data"][0]["x"]) == 5
    assert len(fig_dict["data"][0]["y"]) == 5
    assert fig_dict["data"][0]["name"] == "class_1"

    msg = "Class wine is not one of the classes the pipeline was fit on: class_0, class_1, class_2"
    with pytest.raises(PartialDependenceError, match=msg) as e:
        graph_partial_dependence(pipeline, X, features="alcohol", class_label="wine")
    assert e.value.code == PartialDependenceErrorCode.INVALID_CLASS_LABEL


def test_partial_dependence_percentile_errors(
    logistic_regression_binary_pipeline_class,
):
    # random_col will be 5% 0, 95% 1
    X = pd.DataFrame(
        {
            "A": [i % 3 for i in range(1000)],
            "B": [(j + 3) % 5 for j in range(1000)],
            "random_col": [0 if i < 50 else 1 for i in range(1000)],
            "random_col_2": [0 if i < 40 else 1 for i in range(1000)],
        }
    )
    y = pd.Series([i % 2 for i in range(1000)])
    pipeline = logistic_regression_binary_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
    )
    pipeline.fit(X, y)
    with pytest.raises(
        PartialDependenceError,
        match="Features \\('random_col'\\) are mostly one value, \\(1\\), and cannot be",
    ) as e:
        partial_dependence(pipeline, X, features="random_col", grid_resolution=5)
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE

    with pytest.raises(
        PartialDependenceError,
        match="Features \\('random_col'\\) are mostly one value, \\(1\\), and cannot be",
    ) as e:
        partial_dependence(
            pipeline,
            X,
            features="random_col",
            percentiles=(0.01, 0.955),
            grid_resolution=5,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE

    with pytest.raises(
        PartialDependenceError,
        match="Features \\('random_col'\\) are mostly one value, \\(1\\), and cannot be",
    ) as e:
        partial_dependence(
            pipeline, X, features=2, percentiles=(0.01, 0.955), grid_resolution=5
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE

    with pytest.raises(
        PartialDependenceError,
        match="Features \\('random_col'\\) are mostly one value, \\(1\\), and cannot be",
    ) as e:
        partial_dependence(
            pipeline,
            X,
            features=("A", "random_col"),
            percentiles=(0.01, 0.955),
            grid_resolution=5,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE

    with pytest.raises(
        PartialDependenceError,
        match="Features \\('random_col', 'random_col_2'\\) are mostly one value, \\(1, 1\\), and cannot be",
    ):
        partial_dependence(
            pipeline,
            X,
            features=("random_col", "random_col_2"),
            percentiles=(0.01, 0.955),
            grid_resolution=5,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE

    part_dep = partial_dependence(
        pipeline, X, features="random_col", percentiles=(0.01, 0.96), grid_resolution=5
    )
    assert list(part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]
    assert len(part_dep["partial_dependence"]) == 2
    assert len(part_dep["feature_values"]) == 2
    assert not part_dep.isnull().any(axis=None)


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_graph_partial_dependence_regression_and_binary_categorical(
    problem_type,
    linear_regression_pipeline_class,
    X_y_regression,
    X_y_binary,
    logistic_regression_binary_pipeline_class,
):
    pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )

    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline_class(
            {"Logistic Regression Classifier": {"n_jobs": 1}}
        )
    else:
        X, y = X_y_regression
        pipeline = linear_regression_pipeline_class({"Linear Regressor": {"n_jobs": 1}})

    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y = pd.Series(y)
    X["categorical_column"] = pd.Series([i % 3 for i in range(X.shape[0])]).astype(
        "str"
    )
    X["categorical_column_2"] = pd.Series([i % 6 for i in range(X.shape[0])]).astype(
        "str"
    )

    pipeline.fit(X, y)

    fig = graph_partial_dependence(
        pipeline, X, features="categorical_column", grid_resolution=5
    )
    plot_data = fig.to_dict()["data"][0]
    assert plot_data["type"] == "bar"
    assert list(plot_data["x"]) == ["0", "1", "2"]

    fig = graph_partial_dependence(
        pipeline, X, features=("0", "categorical_column"), grid_resolution=5
    )
    fig_dict = fig.to_dict()
    plot_data = fig_dict["data"][0]
    assert plot_data["type"] == "contour"
    assert fig_dict["layout"]["yaxis"]["ticktext"] == ["0", "1", "2"]
    assert (
        fig_dict["layout"]["title"]["text"]
        == "Partial Dependence of 'categorical_column' vs. '0'"
    )

    fig = graph_partial_dependence(
        pipeline,
        X,
        features=("categorical_column_2", "categorical_column"),
        grid_resolution=5,
    )
    fig_dict = fig.to_dict()
    plot_data = fig_dict["data"][0]
    assert plot_data["type"] == "contour"
    assert fig_dict["layout"]["xaxis"]["ticktext"] == ["0", "1", "2"]
    assert fig_dict["layout"]["yaxis"]["ticktext"] == ["0", "1", "2", "3", "4", "5"]
    assert (
        fig_dict["layout"]["title"]["text"]
        == "Partial Dependence of 'categorical_column_2' vs. 'categorical_column'"
    )


@pytest.mark.parametrize("class_label", [None, "class_1"])
def test_partial_dependence_multiclass_categorical(
    wine_local, class_label, logistic_regression_multiclass_pipeline_class
):
    pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )

    X, y = wine_local
    X.ww["categorical_column"] = ww.init_series(
        pd.Series([i % 3 for i in range(X.shape[0])]).astype(str),
        logical_type="Categorical",
    )
    X.ww["categorical_column_2"] = ww.init_series(
        pd.Series([i % 6 for i in range(X.shape[0])]).astype(str),
        logical_type="Categorical",
    )

    pipeline = logistic_regression_multiclass_pipeline_class(
        {"Logistic Regression Classifier": {"n_jobs": 1}}
    )

    pipeline.fit(X, y)

    fig = graph_partial_dependence(
        pipeline,
        X,
        features="categorical_column",
        class_label=class_label,
        grid_resolution=5,
    )

    for i, plot_data in enumerate(fig.to_dict()["data"]):
        assert plot_data["type"] == "bar"
        assert list(plot_data["x"]) == ["0", "1", "2"]
        if class_label is None:
            assert plot_data["name"] == f"class_{i}"
        else:
            assert plot_data["name"] == class_label

    fig = graph_partial_dependence(
        pipeline,
        X,
        features=("alcohol", "categorical_column"),
        class_label=class_label,
        grid_resolution=5,
    )

    for i, plot_data in enumerate(fig.to_dict()["data"]):
        assert plot_data["type"] == "contour"
        assert fig.to_dict()["layout"]["yaxis"]["ticktext"] == ["0", "1", "2"]
        if class_label is None:
            assert plot_data["name"] == f"class_{i}"
        else:
            assert plot_data["name"] == class_label

    fig = graph_partial_dependence(
        pipeline,
        X,
        features=("categorical_column_2", "categorical_column"),
        class_label=class_label,
        grid_resolution=5,
    )

    for i, plot_data in enumerate(fig.to_dict()["data"]):
        assert plot_data["type"] == "contour"
        assert fig.to_dict()["layout"]["xaxis"]["ticktext"] == ["0", "1", "2"]
        assert fig.to_dict()["layout"]["yaxis"]["ticktext"] == [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
        ]
        if class_label is None:
            assert plot_data["name"] == f"class_{i}"
        else:
            assert plot_data["name"] == class_label


def test_partial_dependence_all_nan_value_error(
    logistic_regression_binary_pipeline_class,
):
    pl = logistic_regression_binary_pipeline_class({})

    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    y = pd.Series([0, 1, 0])
    pl.fit(X, y)

    pred_df = pd.DataFrame({"a": [None] * 5, "b": [1, 2, 3, 4, 4], "c": [None] * 5})
    message = "The following features have all NaN values and so the partial dependence cannot be computed: {}"
    with pytest.raises(PartialDependenceError, match=message.format("'a'")) as e:
        partial_dependence(pl, pred_df, features="a", grid_resolution=10)
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS

    with pytest.raises(PartialDependenceError, match=message.format("'a'")) as e:
        partial_dependence(pl, pred_df, features=0, grid_resolution=10)
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS

    with pytest.raises(PartialDependenceError, match=message.format("'a'")) as e:
        partial_dependence(pl, pred_df, features=("a", "b"), grid_resolution=10)
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS

    with pytest.raises(PartialDependenceError, match=message.format("'a', 'c'")) as e:
        partial_dependence(pl, pred_df, features=("a", "c"), grid_resolution=10)
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS

    pred_df = pred_df.rename(columns={"a": 0})
    with pytest.raises(PartialDependenceError, match=message.format("'0'")) as e:
        partial_dependence(pl, pred_df, features=0, grid_resolution=10)
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS


@pytest.mark.parametrize("grid", [20, 100])
@pytest.mark.parametrize("size", [10, 100])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_partial_dependence_datetime(
    problem_type, size, grid, X_y_regression, X_y_binary, X_y_multi
):
    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = BinaryClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurization Component",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ]
        )
    elif problem_type == "multiclass":
        X, y = X_y_multi
        pipeline = MulticlassClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurization Component",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ]
        )
    else:
        X, y = X_y_regression
        pipeline = RegressionPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurization Component",
                "Standard Scaler",
                "Linear Regressor",
            ]
        )

    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y = pd.Series(y)
    random_dates = pd.Series(pd.date_range("20200101", periods=size))
    if size == 10:
        random_dates = random_dates.sample(replace=True, random_state=0, n=100)
    random_dates.index = X.index
    X["dt_column"] = random_dates
    pipeline.fit(X, y)
    part_dep = partial_dependence(
        pipeline, X, features="dt_column", grid_resolution=grid
    )
    expected_size = min(size, grid)
    num_classes = y.nunique()
    if problem_type == "multiclass":
        assert (
            len(part_dep["partial_dependence"]) == num_classes * expected_size
        )  # 10 rows * 3 classes
        assert len(part_dep["feature_values"]) == num_classes * expected_size
    else:
        assert len(part_dep["partial_dependence"]) == expected_size
        assert len(part_dep["feature_values"]) == expected_size
    assert not part_dep.isnull().any(axis=None)
    # keeps the test from running too long. The part below still runs for 3 other tests
    if grid == 100 or size == 100:
        return
    part_dep = partial_dependence(pipeline, X, features=20, grid_resolution=grid)
    if problem_type == "multiclass":
        assert (
            len(part_dep["partial_dependence"]) == num_classes * expected_size
        )  # 10 rows * 3 classes
        assert len(part_dep["feature_values"]) == num_classes * expected_size
    else:
        assert len(part_dep["partial_dependence"]) == expected_size
        assert len(part_dep["feature_values"]) == expected_size
    assert not part_dep.isnull().any(axis=None)

    with pytest.raises(
        PartialDependenceError,
        match="Two-way partial dependence is not supported for datetime columns.",
    ) as e:
        partial_dependence(pipeline, X, features=("0", "dt_column"))
    assert e.value.code == PartialDependenceErrorCode.TWO_WAY_REQUESTED_FOR_DATES

    with pytest.raises(
        PartialDependenceError,
        match="Two-way partial dependence is not supported for datetime columns.",
    ) as e:
        partial_dependence(pipeline, X, features=(0, 20))
    assert e.value.code == PartialDependenceErrorCode.TWO_WAY_REQUESTED_FOR_DATES


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_graph_partial_dependence_regression_and_binary_datetime(
    problem_type, X_y_regression, X_y_binary
):
    pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )

    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = BinaryClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurization Component",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ]
        )
    else:
        X, y = X_y_regression
        pipeline = RegressionPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurization Component",
                "Standard Scaler",
                "Linear Regressor",
            ]
        )

    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y = pd.Series(y)
    X["dt_column"] = pd.to_datetime(
        pd.Series(pd.date_range("20200101", periods=5))
        .sample(n=X.shape[0], replace=True, random_state=0)
        .reset_index(drop=True),
        errors="coerce",
    )

    pipeline.fit(X, y)

    fig = graph_partial_dependence(pipeline, X, features="dt_column", grid_resolution=5)
    plot_data = fig.to_dict()["data"][0]
    assert plot_data["type"] == "scatter"
    assert list(plot_data["x"]) == list(pd.date_range("20200101", periods=5))


def test_graph_partial_dependence_regression_date_order(X_y_binary):
    pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )

    X, y = X_y_binary
    pipeline = BinaryClassificationPipeline(
        component_graph=[
            "Imputer",
            "One Hot Encoder",
            "DateTime Featurization Component",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ]
    )
    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y = pd.Series(y)
    dt_series = (
        pd.Series(pd.date_range("20200101", periods=5))
        .sample(replace=True, n=X.shape[0])
        .sample(frac=1)
        .reset_index(drop=True)
    )
    X["dt_column"] = pd.to_datetime(dt_series, errors="coerce")

    pipeline.fit(X, y)

    fig = graph_partial_dependence(pipeline, X, features="dt_column", grid_resolution=5)
    plot_data = fig.to_dict()["data"][0]
    assert plot_data["type"] == "scatter"
    assert list(plot_data["x"]) == list(pd.date_range("20200101", periods=5))


def test_partial_dependence_respect_grid_resolution(fraud_100):
    X, y = fraud_100
    pl = BinaryClassificationPipeline(
        component_graph=[
            "DateTime Featurization Component",
            "One Hot Encoder",
            "Random Forest Classifier",
        ]
    )
    pl.fit(X, y)
    dep = partial_dependence(pl, X, features="amount", grid_resolution=5)

    assert dep.shape[0] == 5
    assert dep.shape[0] != max(X.ww.select("categorical").describe().loc["unique"]) + 1

    dep = partial_dependence(pl, X, features="provider", grid_resolution=5)
    assert dep.shape[0] == X["provider"].nunique()
    assert dep.shape[0] != max(X.ww.select("categorical").describe().loc["unique"]) + 1


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_graph_partial_dependence_ice_plot(
    problem_type,
    wine_local,
    breast_cancer_local,
    test_pipeline,
    logistic_regression_multiclass_pipeline_class,
):
    go = pytest.importorskip(
        "plotly.graph_objects",
        reason="Skipping plotting test because plotly not installed",
    )

    if problem_type == ProblemTypes.MULTICLASS:
        test_pipeline = logistic_regression_multiclass_pipeline_class(
            parameters={"Logistic Regression Classifier": {"n_jobs": 1}}
        )
        X, y = wine_local
        feature = "ash"
    else:
        X, y = breast_cancer_local
        feature = "mean radius"
    clf = test_pipeline
    clf.fit(X, y)

    fig = graph_partial_dependence(
        clf, X, features=feature, grid_resolution=5, kind="both"
    )
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == f"Partial Dependence of '{feature}' <br><sub>Including Individual Conditional Expectation Plot</sub>"
    )
    n_classes = len(y.unique()) if problem_type == ProblemTypes.MULTICLASS else 1
    assert len(fig_dict["data"]) == len(X) * n_classes + n_classes
    expected_label = (
        "Individual Conditional Expectation"
        if problem_type == ProblemTypes.BINARY
        else "Individual Conditional Expectation: class_0"
    )
    assert fig_dict["data"][0]["name"] == expected_label
    assert (
        fig_dict["data"][-1]["name"] == "Partial Dependence"
        if problem_type == ProblemTypes.BINARY
        else "Partial Dependence: class_2"
    )

    avg_dep_data, ind_dep_data = partial_dependence(
        clf, X, features=feature, grid_resolution=5, kind="both"
    )
    assert np.array_equal(
        fig_dict["data"][-1]["x"],
        avg_dep_data["feature_values"][: len(fig_dict["data"][-1]["x"])].values,
    )

    if problem_type == ProblemTypes.BINARY:
        assert np.array_equal(
            fig_dict["data"][-1]["y"], avg_dep_data["partial_dependence"].values
        )
    else:
        class_2_data = avg_dep_data[avg_dep_data["class_label"] == "class_2"][
            "partial_dependence"
        ].values
        assert np.array_equal(fig_dict["data"][-1]["y"], class_2_data)

    for i in range(len(X)):
        assert np.array_equal(fig_dict["data"][i]["x"], ind_dep_data["feature_values"])
        if problem_type == ProblemTypes.MULTICLASS:
            window_length = len(ind_dep_data[f"Sample {i}"].values)
            data = np.concatenate(
                [
                    fig_dict["data"][i]["y"][:window_length],
                    fig_dict["data"][i + len(X) + 1]["y"][:window_length],
                    fig_dict["data"][i + 2 * len(X) + 2]["y"][:window_length],
                ]
            )
            assert np.array_equal(data, ind_dep_data[f"Sample {i}"].values)
        else:
            assert np.array_equal(
                fig_dict["data"][i]["y"], ind_dep_data[f"Sample {i}"].values
            )

    fig = graph_partial_dependence(
        clf, X, features=feature, grid_resolution=5, kind="individual"
    )
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == f"Individual Conditional Expectation of '{feature}'"
    )
    assert len(fig_dict["data"]) == len(X) * (
        len(y.unique()) if problem_type == ProblemTypes.MULTICLASS else 1
    )
    expected_label = (
        "Individual Conditional Expectation"
        if problem_type == ProblemTypes.BINARY
        else "Individual Conditional Expectation: class_0"
    )
    assert fig_dict["data"][0]["name"] == expected_label

    ind_dep_data = partial_dependence(
        clf, X, features=feature, grid_resolution=5, kind="individual"
    )

    for i in range(len(X)):
        assert np.array_equal(fig_dict["data"][i]["x"], ind_dep_data["feature_values"])
        if problem_type == ProblemTypes.MULTICLASS:
            window_length = len(ind_dep_data[f"Sample {i}"].values)
            data = np.concatenate(
                [
                    fig_dict["data"][i]["y"][:window_length],
                    fig_dict["data"][i + len(X)]["y"][:window_length],
                    fig_dict["data"][i + 2 * len(X)]["y"][:window_length],
                ]
            )
            assert np.array_equal(data, ind_dep_data[f"Sample {i}"].values)
        else:
            assert np.array_equal(
                fig_dict["data"][i]["y"], ind_dep_data[f"Sample {i}"].values
            )


def test_graph_partial_dependence_ice_plot_two_way_error(
    breast_cancer_local, test_pipeline
):
    X, y = breast_cancer_local
    clf = test_pipeline
    clf.fit(X, y)
    with pytest.raises(
        PartialDependenceError,
        match="Individual conditional expectation plot can only be created with a one-way partial dependence plot",
    ) as e:
        graph_partial_dependence(
            clf,
            X,
            features=["mean radius", "mean area"],
            grid_resolution=5,
            kind="both",
        )
    assert (
        e.value.code == PartialDependenceErrorCode.ICE_PLOT_REQUESTED_FOR_TWO_WAY_PLOT
    )

    with pytest.raises(
        PartialDependenceError,
        match="Individual conditional expectation plot can only be created with a one-way partial dependence plot",
    ) as e:
        graph_partial_dependence(
            clf,
            X,
            features=["mean radius", "mean area"],
            grid_resolution=5,
            kind="individual",
        )
    assert (
        e.value.code == PartialDependenceErrorCode.ICE_PLOT_REQUESTED_FOR_TWO_WAY_PLOT
    )


def test_partial_dependence_scale_error():
    """Test to catch the case when the scale of the features is so small
    that the 5th and 95th percentiles are too close to each other.  This is
    an sklearn exception."""

    pl = RegressionPipeline(["Random Forest Regressor"])
    X = pd.DataFrame({"a": list(range(30)), "b": list(range(-10, 20))})
    y = 10 * X["a"] + X["b"]

    pl.fit(X, y)

    X_pd = X.copy()
    X_pd["a"] = X["a"] * 1.0e-10

    # Catch the intended sklearn error and change the message.
    with pytest.raises(
        PartialDependenceError, match="scale of these features is too small"
    ) as e:
        partial_dependence(pl, X_pd, "a", grid_resolution=5)
    assert e.value.code == PartialDependenceErrorCode.COMPUTED_PERCENTILES_TOO_CLOSE

    # Ensure that sklearn partial_dependence exceptions are still caught as expected.
    with pytest.raises(
        PartialDependenceError,
        match="'grid_resolution' must be strictly greater than 1.",
    ) as e:
        partial_dependence(pl, X_pd, "a", grid_resolution=0)
    assert e.value.code == PartialDependenceErrorCode.ALL_OTHER_ERRORS


@pytest.mark.parametrize("indices,error", [(0, True), (1, False)])
def test_partial_dependence_unknown(indices, error, X_y_binary):
    # test to see if we can get partial dependence fine with a dataset that has unknown features
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X.ww.init(logical_types={0: "unknown"})
    pl = BinaryClassificationPipeline(["Random Forest Classifier"])
    pl.fit(X, y)
    if error:
        with pytest.raises(
            PartialDependenceError,
            match=r"Columns \[0\] are of type 'Unknown', which cannot be used for partial dependence",
        ) as e:
            partial_dependence(pl, X, indices, grid_resolution=2)
        assert e.value.code == PartialDependenceErrorCode.INVALID_FEATURE_TYPE
    else:
        s = partial_dependence(pl, X, indices, grid_resolution=2)
        assert not s.isnull().any().any()


@pytest.mark.parametrize(
    "X_datasets",
    [
        pd.DataFrame(
            {
                "date_column": pd.date_range("20200101", periods=100),
                "numbers": [i % 3 for i in range(100)],
                "date2": pd.date_range("20191001", periods=100),
            }
        ),
        pd.DataFrame(
            {
                "date_column": pd.date_range("20200101", periods=10).append(
                    pd.date_range("20191201", periods=50).append(
                        pd.date_range("20180201", periods=40)
                    )
                )
            }
        ),
        pd.DataFrame(
            {
                "date_column": pd.date_range(
                    start="20200101", freq="10h30min50s", periods=100
                )
            }
        ),
    ],
)
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_partial_dependence_datetime_extra(
    problem_type, X_datasets, X_y_regression, X_y_binary, X_y_multi
):
    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = BinaryClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurization Component",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ]
        )
    elif problem_type == "multiclass":
        X, y = X_y_multi
        pipeline = MulticlassClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurization Component",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ]
        )
    else:
        X, y = X_y_regression
        pipeline = RegressionPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurization Component",
                "Standard Scaler",
                "Linear Regressor",
            ]
        )

    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    X = pd.concat([X, X_datasets], axis=1, join="inner")
    y = pd.Series(y)
    pipeline.fit(X, y)
    part_dep = partial_dependence(
        pipeline, X, features="date_column", grid_resolution=10
    )
    num_classes = y.nunique()
    if problem_type == "multiclass":
        assert (
            len(part_dep["partial_dependence"]) == num_classes * 10
        )  # 10 rows * 3 classes
        assert len(part_dep["feature_values"]) == num_classes * 10
    else:
        assert len(part_dep["partial_dependence"]) == 10
        assert len(part_dep["feature_values"]) == 10
    assert not part_dep.isnull().any(axis=None)

    part_dep = partial_dependence(pipeline, X, features=20, grid_resolution=10)
    if problem_type == "multiclass":
        assert len(part_dep["partial_dependence"]) == num_classes * 10
        assert len(part_dep["feature_values"]) == num_classes * 10
    else:
        assert len(part_dep["partial_dependence"]) == 10
        assert len(part_dep["feature_values"]) == 10
    assert not part_dep.isnull().any(axis=None)


@pytest.mark.parametrize(
    "cols,expected_cols",
    [
        (0, ["changing_col"]),
        ([0, 1], ["changing_col", "URL_col"]),
        ([0, 2], ["changing_col"]),
        (2, []),
    ],
)
@pytest.mark.parametrize("types", ["URL", "EmailAddress", "NaturalLanguage"])
def test_partial_dependence_not_allowed_types(types, cols, expected_cols):
    X = pd.DataFrame(
        {
            "changing_col": [i for i in range(1000)],
            "URL_col": [i % 5 for i in range(1000)],
            "col": [i % 10 for i in range(1000)],
        }
    )
    y = pd.Series([i % 2 for i in range(1000)])
    X.ww.init(logical_types={"changing_col": types, "URL_col": "URL"})
    pl = BinaryClassificationPipeline(["Random Forest Classifier"])
    pl.fit(X, y)
    if len(expected_cols):
        expected_types = (
            sorted(set([types, "URL"])) if len(expected_cols) == 2 else [types]
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Columns {expected_cols} are of types {expected_types}, which cannot be used for partial dependence"
            ),
        ):
            partial_dependence(pl, X, cols, grid_resolution=2)
        return
    s = partial_dependence(pl, X, cols, grid_resolution=2)
    assert not s.isnull().any().any()


def test_partial_dependence_categorical_nan(fraud_100):
    X, y = fraud_100
    X.ww["provider"][:10] = None
    pl = BinaryClassificationPipeline(
        component_graph=[
            "Imputer",
            "DateTime Featurization Component",
            "One Hot Encoder",
            "Random Forest Classifier",
        ]
    )
    pl.fit(X, y)

    GRID_RESOLUTION = 5
    dep = partial_dependence(
        pl, X, features="provider", grid_resolution=GRID_RESOLUTION
    )

    assert dep.shape[0] == X["provider"].dropna().nunique()
    assert not dep["feature_values"].isna().any()
    assert not dep["partial_dependence"].isna().any()

    dep2way = partial_dependence(
        pl, X, features=("amount", "provider"), grid_resolution=GRID_RESOLUTION
    )
    assert not dep2way.isna().any().any()
    # Plus 1 in the columns because there is `class_label`
    assert dep2way.shape == (GRID_RESOLUTION, X["provider"].dropna().nunique() + 1)
