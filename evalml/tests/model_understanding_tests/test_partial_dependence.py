import re
from unittest.mock import patch

import featuretools as ft
import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.exceptions import (
    NullsInColumnWarning,
    PartialDependenceError,
    PartialDependenceErrorCode,
)
from evalml.model_understanding import graph_partial_dependence, partial_dependence
from evalml.pipelines import (
    BinaryClassificationPipeline,
    ClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
)
from evalml.pipelines.components.transformers import (
    DFSTransformer,
    DropColumns,
    SelectColumns,
)
from evalml.pipelines.utils import _make_stacked_ensemble_pipeline
from evalml.problem_types import ProblemTypes


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
    logistic_regression_binary_pipeline,
    logistic_regression_multiclass_pipeline,
    linear_regression_pipeline,
    make_data_type,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        pipeline = logistic_regression_multiclass_pipeline

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        pipeline = linear_regression_pipeline

    X = make_data_type(data_type, X)
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features=0, grid_resolution=5)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().any(axis=None)

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features=0,
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


def test_partial_dependence_string_feature_name(
    breast_cancer_local,
    logistic_regression_binary_pipeline,
):
    X, y = breast_cancer_local
    logistic_regression_binary_pipeline.fit(X, y)
    part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="mean radius",
        grid_resolution=5,
    )
    assert list(part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]
    assert len(part_dep["partial_dependence"]) == 5
    assert len(part_dep["feature_values"]) == 5
    assert not part_dep.isnull().any(axis=None)

    fast_part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="mean radius",
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )

    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_partial_dependence_with_non_numeric_columns(
    data_type,
    linear_regression_pipeline,
):
    X = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 0] * 4,
            "also numeric": [2, 3, 4, 1] * 4,
            "string": ["a", "b", "a", "c"] * 4,
            "also string": ["c", "b", "a", "c"] * 4,
        },
    )
    if data_type == "ww":
        X.ww.init()
    y = [0, 0.2, 1.4, 1] * 4

    linear_regression_pipeline.fit(X, y)
    part_dep = partial_dependence(linear_regression_pipeline, X, features="numeric")
    assert list(part_dep.columns) == ["feature_values", "partial_dependence"]
    assert len(part_dep["partial_dependence"]) == 4
    assert len(part_dep["feature_values"]) == 4
    assert not part_dep.isnull().any(axis=None)

    fast_part_dep = partial_dependence(
        linear_regression_pipeline,
        X,
        features="numeric",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    part_dep = partial_dependence(linear_regression_pipeline, X, features="string")
    assert list(part_dep.columns) == ["feature_values", "partial_dependence"]
    assert len(part_dep["partial_dependence"]) == 3
    assert len(part_dep["feature_values"]) == 3
    assert not part_dep.isnull().any(axis=None)

    fast_part_dep = partial_dependence(
        linear_regression_pipeline,
        X,
        features="string",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


@patch(
    "evalml.pipelines.BinaryClassificationPipeline.predict_proba",
    side_effect=lambda X: np.array([[0.2, 0.8]] * X.shape[0]),
)
@patch(
    "evalml.pipelines.components.estimators.Estimator.predict_proba",
    side_effect=lambda X: np.array([[0.2, 0.8]] * X.shape[0]),
)
def test_partial_dependence_with_ww_category_columns(
    mock_predict_proba,
    mock_estimator_predict_proba,
    fraud_100,
    logistic_regression_binary_pipeline,
):
    X, y = fraud_100
    X.ww.set_types(
        logical_types={
            "store_id": "PostalCode",
            "country": "CountryCode",
            "region": "SubRegionCode",
        },
    )
    logistic_regression_binary_pipeline.fit(X, y)

    part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="store_id",
    )
    assert list(part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]
    assert len(part_dep["partial_dependence"]) == 11
    assert len(part_dep["feature_values"]) == 11
    assert not part_dep.isnull().any(axis=None)

    fast_part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="store_id",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="country",
    )
    assert list(part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]
    assert len(part_dep["partial_dependence"]) == 8
    assert len(part_dep["feature_values"]) == 8
    assert not part_dep.isnull().any(axis=None)

    part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="region",
    )
    assert list(part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]
    assert len(part_dep["partial_dependence"]) == 11
    assert len(part_dep["feature_values"]) == 11
    assert not part_dep.isnull().any(axis=None)

    fast_part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="region",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


@patch(
    "evalml.pipelines.BinaryClassificationPipeline.predict_proba",
    side_effect=lambda X: np.array([[0.2, 0.8]] * X.shape[0]),
)
@patch(
    "evalml.pipelines.components.estimators.Estimator.predict_proba",
    side_effect=lambda X: np.array([[0.2, 0.8]] * X.shape[0]),
)
def test_two_way_partial_dependence_with_ww_category_columns(
    mock_predict_proba,
    mock_estimator_predict_proba,
    fraud_100,
    logistic_regression_binary_pipeline,
):
    X, y = fraud_100
    X.ww.set_types(
        logical_types={
            "store_id": "PostalCode",
            "country": "CountryCode",
            "region": "SubRegionCode",
        },
    )

    logistic_regression_binary_pipeline.fit(X, y)

    part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features=("store_id", "country"),
    )
    assert "class_label" in part_dep.columns
    assert part_dep["class_label"].nunique() == 1
    assert len(part_dep.index) == len(set(X["store_id"]))
    assert len(part_dep.columns) == len(set(X["country"])) + 1

    fast_part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features=("store_id", "country"),
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    grid_resolution = 5
    part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features=("store_id", "amount"),
        grid_resolution=grid_resolution,
    )
    assert "class_label" in part_dep.columns
    assert part_dep["class_label"].nunique() == 1
    assert len(part_dep.index) == len(set(X["store_id"]))
    assert len(part_dep.columns) == grid_resolution + 1

    fast_part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features=("store_id", "amount"),
        grid_resolution=grid_resolution,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    grid_resolution = 5
    part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features=("amount", "store_id"),
        grid_resolution=grid_resolution,
    )
    assert "class_label" in part_dep.columns
    assert part_dep["class_label"].nunique() == 1
    assert len(part_dep.columns) == len(set(X["store_id"])) + 1
    assert len(part_dep.index) == grid_resolution

    fast_part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features=("amount", "store_id"),
        grid_resolution=grid_resolution,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


def test_partial_dependence_baseline():
    X = pd.DataFrame([[1, 0], [0, 1]])
    y = pd.Series([0, 1])
    pipeline = BinaryClassificationPipeline(
        component_graph=["Baseline Classifier"],
        parameters={},
    )
    pipeline.fit(X, y)
    with pytest.raises(
        PartialDependenceError,
        match="Partial dependence plots are not supported for Baseline pipelines",
    ) as e:
        partial_dependence(pipeline, X, features=0, grid_resolution=5)
    assert e.value.code == PartialDependenceErrorCode.PIPELINE_IS_BASELINE


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_partial_dependence_catboost(problem_type, X_y_binary, X_y_multi):
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

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features=0,
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    # test that CatBoost can natively handle non-numerical columns as feature passed to partial_dependence
    X = pd.DataFrame(
        {
            "numeric": [1, 2, 3] * 5,
            "also numeric": [2, 3, 4] * 5,
            "string": ["a", "b", "c"] * 5,
            "also string": ["c", "b", "a"] * 5,
        },
    )
    pipeline = pipeline_class(
        component_graph=["CatBoost Classifier"],
        parameters={"CatBoost Classifier": {"thread_count": 1}},
    )
    pipeline.fit(X, y_small)
    part_dep = partial_dependence(pipeline, X, features="string")
    check_partial_dependence_dataframe(pipeline, part_dep, grid_size=3)
    assert not part_dep.isnull().all().all()

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features="string",
        fast_mode=True,
        X_train=X,
        y_train=y_small,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION],
)
def test_partial_dependence_xgboost_feature_names(
    problem_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
):
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

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features="<[0]",
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    part_dep = partial_dependence(pipeline, X, features=1, grid_resolution=5)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().all().all()

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features=1,
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


def test_partial_dependence_multiclass(
    wine_local,
    logistic_regression_multiclass_pipeline,
):
    X, y = wine_local
    logistic_regression_multiclass_pipeline.fit(X, y)

    num_classes = y.nunique()
    grid_resolution = 5

    one_way_part_dep = partial_dependence(
        pipeline=logistic_regression_multiclass_pipeline,
        X=X,
        features="magnesium",
        grid_resolution=grid_resolution,
    )
    assert "class_label" in one_way_part_dep.columns
    assert one_way_part_dep["class_label"].nunique() == num_classes
    assert len(one_way_part_dep.index) == num_classes * grid_resolution
    assert list(one_way_part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]

    fast_part_dep = partial_dependence(
        logistic_regression_multiclass_pipeline,
        X,
        features="magnesium",
        grid_resolution=grid_resolution,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(one_way_part_dep, fast_part_dep)

    two_way_part_dep = partial_dependence(
        pipeline=logistic_regression_multiclass_pipeline,
        X=X,
        features=("magnesium", "alcohol"),
        grid_resolution=grid_resolution,
    )

    assert "class_label" in two_way_part_dep.columns
    assert two_way_part_dep["class_label"].nunique() == num_classes
    assert len(two_way_part_dep.index) == num_classes * grid_resolution
    assert len(two_way_part_dep.columns) == grid_resolution + 1

    fast_part_dep = partial_dependence(
        logistic_regression_multiclass_pipeline,
        X,
        features=("magnesium", "alcohol"),
        grid_resolution=grid_resolution,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(two_way_part_dep, fast_part_dep)

    two_way_part_dep, two_way_ice_data = partial_dependence(
        pipeline=logistic_regression_multiclass_pipeline,
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

    fast_part_dep, new_ice_data = partial_dependence(
        logistic_regression_multiclass_pipeline,
        X,
        features=("magnesium", "alcohol"),
        grid_resolution=grid_resolution,
        kind="both",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(two_way_part_dep, fast_part_dep)
    for i, ice_df in enumerate(two_way_ice_data):
        new_ice_df = new_ice_data[i]
        pd.testing.assert_frame_equal(ice_df, new_ice_df)


def test_partial_dependence_multiclass_numeric_labels(
    logistic_regression_multiclass_pipeline,
    X_y_multi,
):
    X, y = X_y_multi
    X = pd.DataFrame(X)
    y = pd.Series(y, dtype="int64")
    logistic_regression_multiclass_pipeline.fit(X, y)

    num_classes = y.nunique()
    grid_resolution = 5

    one_way_part_dep = partial_dependence(
        pipeline=logistic_regression_multiclass_pipeline,
        X=X,
        features=1,
        grid_resolution=grid_resolution,
    )
    assert "class_label" in one_way_part_dep.columns
    assert one_way_part_dep["class_label"].nunique() == num_classes
    assert len(one_way_part_dep.index) == num_classes * grid_resolution
    assert list(one_way_part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]
    fast_part_dep = partial_dependence(
        logistic_regression_multiclass_pipeline,
        X,
        features=1,
        grid_resolution=grid_resolution,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(one_way_part_dep, fast_part_dep)

    two_way_part_dep = partial_dependence(
        pipeline=logistic_regression_multiclass_pipeline,
        X=X,
        features=(1, 2),
        grid_resolution=grid_resolution,
    )

    assert "class_label" in two_way_part_dep.columns
    assert two_way_part_dep["class_label"].nunique() == num_classes
    assert len(two_way_part_dep.index) == num_classes * grid_resolution
    assert len(two_way_part_dep.columns) == grid_resolution + 1

    fast_part_dep = partial_dependence(
        logistic_regression_multiclass_pipeline,
        X,
        features=(1, 2),
        grid_resolution=grid_resolution,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(two_way_part_dep, fast_part_dep)


def test_partial_dependence_not_fitted(X_y_binary, logistic_regression_binary_pipeline):
    X, _ = X_y_binary
    with pytest.raises(
        PartialDependenceError,
        match="Pipeline to calculate partial dependence for must be fitted",
    ) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
            X,
            features=0,
            grid_resolution=5,
        )
    assert e.value.code == PartialDependenceErrorCode.UNFITTED_PIPELINE


def test_partial_dependence_warning(logistic_regression_binary_pipeline):
    X = pd.DataFrame({"a": [1, 2, None, 2, 2], "b": [1, 1, 2, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipeline = logistic_regression_binary_pipeline
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


def test_partial_dependence_errors(logistic_regression_binary_pipeline):
    X = pd.DataFrame({"a": [2, None, 2, 2], "b": [1, 2, 2, 1], "c": [0, 0, 0, 0]})
    y = pd.Series([0, 1, 0, 1])
    pipeline = logistic_regression_binary_pipeline
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
    logistic_regression_binary_pipeline,
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
    pipeline = logistic_regression_binary_pipeline
    pipeline.fit(X, y)
    num_cat_features = len(set(X["currency"]))
    assert num_cat_features == 73

    part_dep_ans = {
        0.08407392065748104: 63,
        0.10309078422967363: 1,
        0.07237644261749679: 1,
        0.1291414419733119: 1,
        0.11725464403860779: 1,
        0.06948453894869809: 1,
        0.08055687803593513: 1,
        0.14807925421481405: 1,
        0.07341294777400768: 1,
        0.15000846543936355: 1,
        0.07507641569899197: 1,
    }

    part_dep_ans_rounded = round_dict_keys(part_dep_ans)

    # Check the case where grid_resolution < number of categorical features
    part_dep = partial_dependence(
        pipeline,
        X,
        "currency",
        grid_resolution=round(num_cat_features / 2),
    )
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        "currency",
        grid_resolution=round(num_cat_features / 2),
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    # Check the case where grid_resolution == number of categorical features
    part_dep = partial_dependence(
        pipeline,
        X,
        "currency",
        grid_resolution=round(num_cat_features),
    )
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        "currency",
        grid_resolution=round(num_cat_features),
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    # Check the case where grid_resolution > number of categorical features
    part_dep = partial_dependence(
        pipeline,
        X,
        "currency",
        grid_resolution=round(num_cat_features * 2),
    )
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        "currency",
        grid_resolution=round(num_cat_features * 2),
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


def test_partial_dependence_ice_plot(logistic_regression_binary_pipeline):
    X = pd.DataFrame({"a": [1, 2, None, 2, 2], "b": [1, 1, 2, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipeline = logistic_regression_binary_pipeline
    pipeline.fit(X, y)

    avg_pred, ind_preds = partial_dependence(pipeline, X, features="a", kind="both")
    assert isinstance(avg_pred, pd.DataFrame)
    assert isinstance(ind_preds, pd.DataFrame)

    assert avg_pred.shape == (2, 3)
    assert ind_preds.shape == (2, 7)

    fast_avg_pred, fast_ind_preds = partial_dependence(
        pipeline,
        X,
        features="a",
        kind="both",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(avg_pred, fast_avg_pred)
    pd.testing.assert_frame_equal(ind_preds, fast_ind_preds)

    ind_preds = partial_dependence(pipeline, X, features="b", kind="individual")
    assert isinstance(ind_preds, pd.DataFrame)

    assert ind_preds.shape == (2, 7)

    fast_ind_preds = partial_dependence(
        pipeline,
        X,
        features="b",
        kind="individual",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(ind_preds, fast_ind_preds)


def test_two_way_partial_dependence_ice_plot(logistic_regression_binary_pipeline):
    X = pd.DataFrame({"a": [1, 2, None, 2, 2], "b": [1, 1, 2, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipeline = logistic_regression_binary_pipeline
    pipeline.fit(X, y)

    avg_pred, ind_preds = partial_dependence(
        pipeline,
        X,
        features=["a", "b"],
        grid_resolution=5,
        kind="both",
    )
    assert isinstance(avg_pred, pd.DataFrame)
    assert isinstance(ind_preds, list)
    assert isinstance(ind_preds[0], pd.DataFrame)

    assert avg_pred.shape == (2, 3)
    assert len(ind_preds) == 5
    for ind_df in ind_preds:
        assert ind_df.shape == (2, 3)

    fast_avg_pred, fast_ind_preds = partial_dependence(
        pipeline,
        X,
        features=["a", "b"],
        grid_resolution=5,
        kind="both",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(avg_pred, fast_avg_pred)
    for i, ice_df in enumerate(ind_preds):
        new_ice_df = fast_ind_preds[i]
        pd.testing.assert_frame_equal(ice_df, new_ice_df)

    ind_preds = partial_dependence(
        pipeline,
        X,
        features=["a", "b"],
        grid_resolution=5,
        kind="individual",
    )
    assert isinstance(ind_preds, list)
    assert isinstance(ind_preds[0], pd.DataFrame)

    assert len(ind_preds) == 5
    for ind_df in ind_preds:
        assert ind_df.shape == (2, 3)

    fast_ind_preds = partial_dependence(
        pipeline,
        X,
        features=["a", "b"],
        grid_resolution=5,
        kind="individual",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    for i, ice_df in enumerate(ind_preds):
        new_ice_df = fast_ind_preds[i]
        pd.testing.assert_frame_equal(ice_df, new_ice_df)


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.REGRESSION])
def test_partial_dependence_ensemble_pipeline(problem_type, X_y_binary, X_y_regression):
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
    )
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features=0, grid_resolution=5)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().all().all()


def test_graph_partial_dependence(
    breast_cancer_local,
    logistic_regression_binary_pipeline,
    go,
):
    X, y = breast_cancer_local

    logistic_regression_binary_pipeline.fit(X, y)
    fig = graph_partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="mean radius",
        grid_resolution=5,
    )
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict["layout"]["title"]["text"] == "Partial Dependence of 'mean radius'"
    assert len(fig_dict["data"]) == 1
    assert fig_dict["data"][0]["name"] == "Partial Dependence"

    part_dep_data = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="mean radius",
        grid_resolution=5,
    )
    assert np.array_equal(fig_dict["data"][0]["x"], part_dep_data["feature_values"])
    assert np.array_equal(
        fig_dict["data"][0]["y"],
        part_dep_data["partial_dependence"].values,
    )

    fast_part_dep_data = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="mean radius",
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep_data, fast_part_dep_data)


@patch(
    "evalml.pipelines.BinaryClassificationPipeline.predict_proba",
    side_effect=lambda X: np.array([[0.2, 0.8]] * X.shape[0]),
)
def test_graph_partial_dependence_ww_categories(
    mock_predict_proba,
    fraud_100,
    logistic_regression_binary_pipeline,
    go,
):

    X, y = fraud_100
    X.ww.set_types(
        logical_types={
            "store_id": "PostalCode",
            "country": "CountryCode",
            "region": "SubRegionCode",
        },
    )
    pipeline = logistic_regression_binary_pipeline
    pipeline.fit(X, y)

    for feat in ["store_id", "country", "region"]:
        fig = graph_partial_dependence(pipeline, X, features=feat)
        assert isinstance(fig, go.Figure)
        fig_dict = fig.to_dict()

        assert fig_dict["data"][0]["type"] == "bar"

        assert fig_dict["layout"]["title"]["text"] == f"Partial Dependence of '{feat}'"
        assert len(fig_dict["data"]) == 1
        assert fig_dict["data"][0]["name"] == "Partial Dependence"

        part_dep_data = partial_dependence(pipeline, X, features=feat)
        assert np.array_equal(fig_dict["data"][0]["x"], part_dep_data["feature_values"])
        assert np.array_equal(
            fig_dict["data"][0]["y"],
            part_dep_data["partial_dependence"].values,
        )


def test_graph_two_way_partial_dependence(
    breast_cancer_local,
    logistic_regression_binary_pipeline,
    go,
):
    X, y = breast_cancer_local
    logistic_regression_binary_pipeline.fit(X, y)
    fig = graph_partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features=("mean radius", "mean area"),
        grid_resolution=5,
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
        logistic_regression_binary_pipeline,
        X,
        features=("mean radius", "mean area"),
        grid_resolution=5,
    )
    part_dep_data.drop(columns=["class_label"], inplace=True)
    assert np.array_equal(fig_dict["data"][0]["x"], part_dep_data.columns)
    assert np.array_equal(fig_dict["data"][0]["y"], part_dep_data.index)
    assert np.array_equal(fig_dict["data"][0]["z"], part_dep_data.values)

    fast_part_dep_data = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features=("mean radius", "mean area"),
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    fast_part_dep_data.drop(columns=["class_label"], inplace=True)
    pd.testing.assert_frame_equal(part_dep_data, fast_part_dep_data)


@patch(
    "evalml.pipelines.BinaryClassificationPipeline.predict_proba",
    side_effect=lambda X: np.array([[0.2, 0.8]] * X.shape[0]),
)
@patch(
    "evalml.pipelines.components.estimators.Estimator.predict_proba",
    side_effect=lambda X: np.array([[0.2, 0.8]] * X.shape[0]),
)
def test_graph_two_way_partial_dependence_ww_categories(
    mock_predict_proba,
    mock_estimator_predict_proba,
    fraud_100,
    logistic_regression_binary_pipeline,
    go,
):
    X, y = fraud_100
    X.ww.set_types(
        logical_types={
            "store_id": "PostalCode",
            "country": "CountryCode",
            "region": "SubRegionCode",
        },
    )
    pipeline = logistic_regression_binary_pipeline
    pipeline.fit(X, y)

    # Two categorical columns
    fig = graph_partial_dependence(pipeline, X, features=("country", "region"))
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == "Partial Dependence of 'country' vs. 'region'"
    )
    assert len(fig_dict["data"]) == 1
    assert fig_dict["data"][0]["name"] == "Partial Dependence"

    part_dep_data = partial_dependence(pipeline, X, features=("country", "region"))
    part_dep_data.drop(columns=["class_label"], inplace=True)
    assert np.array_equal(fig_dict["data"][0]["z"], part_dep_data.values)

    fast_part_dep_data = partial_dependence(
        pipeline,
        X,
        features=("country", "region"),
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    fast_part_dep_data.drop(columns=["class_label"], inplace=True)
    pd.testing.assert_frame_equal(part_dep_data, fast_part_dep_data)

    # One categorical column, entered first
    fig = graph_partial_dependence(pipeline, X, features=("country", "lat"))
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == "Partial Dependence of 'country' vs. 'lat'"
    )
    assert len(fig_dict["data"]) == 1
    assert fig_dict["data"][0]["name"] == "Partial Dependence"

    part_dep_data = partial_dependence(pipeline, X, features=("country", "lat"))
    part_dep_data.drop(columns=["class_label"], inplace=True)
    assert np.array_equal(fig_dict["data"][0]["x"], part_dep_data.columns)
    assert np.array_equal(fig_dict["data"][0]["z"], part_dep_data.values)

    fast_part_dep_data = partial_dependence(
        pipeline,
        X,
        features=("country", "lat"),
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    fast_part_dep_data.drop(columns=["class_label"], inplace=True)
    pd.testing.assert_frame_equal(part_dep_data, fast_part_dep_data)

    # One categorical column, entered second
    fig = graph_partial_dependence(pipeline, X, features=("lat", "country"))
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert (
        fig_dict["layout"]["title"]["text"]
        == "Partial Dependence of 'country' vs. 'lat'"
    )
    assert len(fig_dict["data"]) == 1
    assert fig_dict["data"][0]["name"] == "Partial Dependence"

    part_dep_data = partial_dependence(pipeline, X, features=("country", "lat"))
    part_dep_data.drop(columns=["class_label"], inplace=True)
    assert np.array_equal(fig_dict["data"][0]["x"], part_dep_data.columns)
    assert np.array_equal(fig_dict["data"][0]["z"], part_dep_data.values)

    fast_part_dep_data = partial_dependence(
        pipeline,
        X,
        features=("country", "lat"),
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    fast_part_dep_data.drop(columns=["class_label"], inplace=True)
    pd.testing.assert_frame_equal(part_dep_data, fast_part_dep_data)


def test_graph_partial_dependence_multiclass(
    wine_local,
    logistic_regression_multiclass_pipeline,
    go,
):

    X, y = wine_local
    logistic_regression_multiclass_pipeline.fit(X, y)

    # Test one-way without class labels
    fig_one_way_no_class_labels = graph_partial_dependence(
        logistic_regression_multiclass_pipeline,
        X,
        features="magnesium",
        grid_resolution=5,
    )
    assert isinstance(fig_one_way_no_class_labels, go.Figure)
    fig_dict = fig_one_way_no_class_labels.to_dict()
    assert len(fig_dict["data"]) == len(
        logistic_regression_multiclass_pipeline.classes_,
    )
    for data, label in zip(
        fig_dict["data"],
        logistic_regression_multiclass_pipeline.classes_,
    ):
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
        logistic_regression_multiclass_pipeline,
        X,
        features="magnesium",
        class_label="class_1",
        grid_resolution=5,
    )
    assert isinstance(fig_one_way_class_labels, go.Figure)
    fig_dict = fig_one_way_class_labels.to_dict()
    assert len(fig_dict["data"]) == 1
    assert len(fig_dict["data"][0]["x"]) == 5
    assert len(fig_dict["data"][0]["y"]) == 5
    assert fig_dict["data"][0]["name"] == "Partial Dependence: class_1"

    msg = "Class wine is not one of the classes the pipeline was fit on: class_0, class_1, class_2"
    with pytest.raises(PartialDependenceError, match=msg) as e:
        graph_partial_dependence(
            logistic_regression_multiclass_pipeline,
            X,
            features="alcohol",
            class_label="wine",
        )
    assert e.value.code == PartialDependenceErrorCode.INVALID_CLASS_LABEL

    # Test two-way without class labels
    fig_two_way_no_class_labels = graph_partial_dependence(
        logistic_regression_multiclass_pipeline,
        X,
        features=("magnesium", "alcohol"),
        grid_resolution=5,
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
        logistic_regression_multiclass_pipeline,
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
        graph_partial_dependence(
            logistic_regression_multiclass_pipeline,
            X,
            features="alcohol",
            class_label="wine",
        )
    assert e.value.code == PartialDependenceErrorCode.INVALID_CLASS_LABEL


def test_partial_dependence_percentile_errors(
    logistic_regression_binary_pipeline,
):
    # random_col will be 5% 0, 95% 1
    X = pd.DataFrame(
        {
            "A": [i % 3 for i in range(1000)],
            "B": [(j + 3) % 5 for j in range(1000)],
            "random_col": [0 if i < 50 else 1 for i in range(1000)],
            "random_col_2": [0 if i < 40 else 1 for i in range(1000)],
        },
    )
    y = pd.Series([i % 2 for i in range(1000)])
    logistic_regression_binary_pipeline.fit(X, y)
    with pytest.raises(
        PartialDependenceError,
        match="Features \\('random_col'\\) are mostly one value, \\(1\\), and cannot be",
    ) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
            X,
            features="random_col",
            grid_resolution=5,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE

    with pytest.raises(
        PartialDependenceError,
        match="Features \\('random_col'\\) are mostly one value, \\(1\\), and cannot be",
    ) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
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
            logistic_regression_binary_pipeline,
            X,
            features=2,
            percentiles=(0.01, 0.955),
            grid_resolution=5,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE

    with pytest.raises(
        PartialDependenceError,
        match="Features \\('random_col'\\) are mostly one value, \\(1\\), and cannot be",
    ) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
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
            logistic_regression_binary_pipeline,
            X,
            features=("random_col", "random_col_2"),
            percentiles=(0.01, 0.955),
            grid_resolution=5,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_MOSTLY_ONE_VALUE

    part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="random_col",
        percentiles=(0.01, 0.96),
        grid_resolution=5,
    )
    assert list(part_dep.columns) == [
        "feature_values",
        "partial_dependence",
        "class_label",
    ]
    assert len(part_dep["partial_dependence"]) == 2
    assert len(part_dep["feature_values"]) == 2
    assert not part_dep.isnull().any(axis=None)

    fast_part_dep = partial_dependence(
        logistic_regression_binary_pipeline,
        X,
        features="random_col",
        percentiles=(0.01, 0.96),
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


@pytest.mark.parametrize("problem_type", ["binary", "regression"])
def test_graph_partial_dependence_regression_and_binary_categorical(
    problem_type,
    linear_regression_pipeline,
    X_y_regression,
    X_y_binary,
    logistic_regression_binary_pipeline,
):

    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline
    else:
        X, y = X_y_regression
        pipeline = linear_regression_pipeline

    X = pd.DataFrame(X)
    X.columns = [str(i) for i in range(X.shape[1])]
    X["categorical_column"] = pd.Series([i % 3 for i in range(X.shape[0])]).astype(
        "str",
    )
    X["categorical_column_2"] = pd.Series([i % 6 for i in range(X.shape[0])]).astype(
        "str",
    )

    pipeline.fit(X, y)

    fig = graph_partial_dependence(
        pipeline,
        X,
        features="categorical_column",
        grid_resolution=5,
    )
    plot_data = fig.to_dict()["data"][0]
    assert plot_data["type"] == "bar"
    assert list(plot_data["x"]) == ["0", "1", "2"]

    fig = graph_partial_dependence(
        pipeline,
        X,
        features=("0", "categorical_column"),
        grid_resolution=5,
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
    wine_local,
    class_label,
    logistic_regression_multiclass_pipeline,
):

    X, y = wine_local
    X.ww["categorical_column"] = ww.init_series(
        pd.Series([i % 3 for i in range(X.shape[0])]).astype(str),
        logical_type="Categorical",
    )
    X.ww["categorical_column_2"] = ww.init_series(
        pd.Series([i % 6 for i in range(X.shape[0])]).astype(str),
        logical_type="Categorical",
    )

    logistic_regression_multiclass_pipeline.fit(X, y)

    fig = graph_partial_dependence(
        logistic_regression_multiclass_pipeline,
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
        logistic_regression_multiclass_pipeline,
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
        logistic_regression_multiclass_pipeline,
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
    logistic_regression_binary_pipeline,
):

    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    y = pd.Series([0, 1, 0])
    logistic_regression_binary_pipeline.fit(X, y)

    pred_df = pd.DataFrame({"a": [None] * 5, "b": [1, 2, 3, 4, 4], "c": [None] * 5})
    pred_df.ww.init(logical_types={"a": "Double", "c": "Double", "b": "Integer"})
    message = "The following features have all NaN values and so the partial dependence cannot be computed: {}"
    with pytest.raises(PartialDependenceError, match=message.format("'a'")) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
            pred_df,
            features="a",
            grid_resolution=10,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS

    with pytest.raises(PartialDependenceError, match=message.format("'a'")) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
            pred_df,
            features=0,
            grid_resolution=10,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS

    with pytest.raises(PartialDependenceError, match=message.format("'a'")) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
            pred_df,
            features=("a", "b"),
            grid_resolution=10,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS

    with pytest.raises(PartialDependenceError, match=message.format("'a', 'c'")) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
            pred_df,
            features=("a", "c"),
            grid_resolution=10,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS

    pred_df = pred_df.ww.rename(columns={"a": 0})
    with pytest.raises(PartialDependenceError, match=message.format("'0'")) as e:
        partial_dependence(
            logistic_regression_binary_pipeline,
            pred_df,
            features=0,
            grid_resolution=10,
        )
    assert e.value.code == PartialDependenceErrorCode.FEATURE_IS_ALL_NANS


@pytest.mark.parametrize("grid", [20, 100])
@pytest.mark.parametrize("size", [10, 100])
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_partial_dependence_datetime(
    problem_type,
    size,
    grid,
    X_y_regression,
    X_y_binary,
    X_y_multi,
):
    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = BinaryClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurizer",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ],
        )
    elif problem_type == "multiclass":
        X, y = X_y_multi
        pipeline = MulticlassClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurizer",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ],
        )
    else:
        X, y = X_y_regression
        pipeline = RegressionPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurizer",
                "Standard Scaler",
                "Linear Regressor",
            ],
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
        pipeline,
        X,
        features="dt_column",
        grid_resolution=grid,
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

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features="dt_column",
        grid_resolution=grid,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

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

    part_dep = partial_dependence(
        pipeline,
        X,
        features=20,
        grid_resolution=grid,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

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
    problem_type,
    X_y_regression,
    X_y_binary,
):

    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = BinaryClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurizer",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ],
        )
    else:
        X, y = X_y_regression
        pipeline = RegressionPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurizer",
                "Standard Scaler",
                "Linear Regressor",
            ],
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

    X, y = X_y_binary
    pipeline = BinaryClassificationPipeline(
        component_graph=[
            "Imputer",
            "One Hot Encoder",
            "DateTime Featurizer",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ],
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
            "DateTime Featurizer",
            "One Hot Encoder",
            "Random Forest Classifier",
        ],
    )
    pl.fit(X, y)
    dep = partial_dependence(pl, X, features="amount", grid_resolution=5)

    assert dep.shape[0] == 5
    assert dep.shape[0] != max(X.ww.select("categorical").describe().loc["unique"]) + 1

    fast_dep = partial_dependence(
        pl,
        X,
        features="amount",
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(dep, fast_dep)

    dep = partial_dependence(pl, X, features="provider", grid_resolution=5)
    assert dep.shape[0] == X["provider"].nunique()
    assert dep.shape[0] != max(X.ww.select("categorical").describe().loc["unique"]) + 1

    fast_dep = partial_dependence(
        pl,
        X,
        features="provider",
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(dep, fast_dep)


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_graph_partial_dependence_ice_plot(
    problem_type,
    wine_local,
    breast_cancer_local,
    logistic_regression_binary_pipeline,
    logistic_regression_multiclass_pipeline,
):
    from plotly import graph_objects as go

    if problem_type == ProblemTypes.MULTICLASS:
        clf = logistic_regression_multiclass_pipeline
        X, y = wine_local
        feature = "ash"
    else:
        X, y = breast_cancer_local
        feature = "mean radius"
        clf = logistic_regression_binary_pipeline
    clf.fit(X, y)

    fig = graph_partial_dependence(
        clf,
        X,
        features=feature,
        grid_resolution=5,
        kind="both",
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
        clf,
        X,
        features=feature,
        grid_resolution=5,
        kind="both",
    )
    assert np.array_equal(
        fig_dict["data"][-1]["x"],
        avg_dep_data["feature_values"][: len(fig_dict["data"][-1]["x"])].values,
    )

    if problem_type == ProblemTypes.BINARY:
        assert np.array_equal(
            fig_dict["data"][-1]["y"],
            avg_dep_data["partial_dependence"].values,
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
                ],
            )
            assert np.array_equal(data, ind_dep_data[f"Sample {i}"].values)
        else:
            assert np.array_equal(
                fig_dict["data"][i]["y"],
                ind_dep_data[f"Sample {i}"].values,
            )
    fast_avg_dep_data, fast_ind_dep_data = partial_dependence(
        clf,
        X,
        features=feature,
        grid_resolution=5,
        kind="both",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(avg_dep_data, fast_avg_dep_data)
    pd.testing.assert_frame_equal(ind_dep_data, fast_ind_dep_data)

    fig = graph_partial_dependence(
        clf,
        X,
        features=feature,
        grid_resolution=5,
        kind="individual",
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
        clf,
        X,
        features=feature,
        grid_resolution=5,
        kind="individual",
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
                ],
            )
            assert np.array_equal(data, ind_dep_data[f"Sample {i}"].values)
        else:
            assert np.array_equal(
                fig_dict["data"][i]["y"],
                ind_dep_data[f"Sample {i}"].values,
            )

    fast_ind_dep_data = partial_dependence(
        clf,
        X,
        features=feature,
        grid_resolution=5,
        kind="individual",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(ind_dep_data, fast_ind_dep_data)


def test_graph_partial_dependence_ice_plot_two_way_error(
    breast_cancer_local,
    logistic_regression_binary_pipeline,
):
    X, y = breast_cancer_local
    logistic_regression_binary_pipeline.fit(X, y)
    with pytest.raises(
        PartialDependenceError,
        match="Individual conditional expectation plot can only be created with a one-way partial dependence plot",
    ) as e:
        graph_partial_dependence(
            logistic_regression_binary_pipeline,
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
            logistic_regression_binary_pipeline,
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
        PartialDependenceError,
        match="scale of these features is too small",
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
    X.ww.set_types({0: "unknown"})
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
            },
        ),
        pd.DataFrame(
            {
                "date_column": pd.date_range("20200101", periods=10).append(
                    pd.date_range("20191201", periods=50).append(
                        pd.date_range("20180201", periods=40),
                    ),
                ),
            },
        ),
        pd.DataFrame(
            {
                "date_column": pd.date_range(
                    start="20200101",
                    freq="10h30min50s",
                    periods=100,
                ),
            },
        ),
    ],
)
@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_partial_dependence_datetime_extra(
    problem_type,
    X_datasets,
    X_y_regression,
    X_y_binary,
    X_y_multi,
):
    if problem_type == "binary":
        X, y = X_y_binary
        pipeline = BinaryClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurizer",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ],
        )
    elif problem_type == "multiclass":
        X, y = X_y_multi
        pipeline = MulticlassClassificationPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurizer",
                "Standard Scaler",
                "Logistic Regression Classifier",
            ],
        )
    else:
        X, y = X_y_regression
        pipeline = RegressionPipeline(
            component_graph=[
                "Imputer",
                "One Hot Encoder",
                "DateTime Featurizer",
                "Standard Scaler",
                "Linear Regressor",
            ],
        )

    X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    X = pd.concat([X, X_datasets], axis=1, join="inner")
    y = pd.Series(y)
    pipeline.fit(X, y)
    part_dep = partial_dependence(
        pipeline,
        X,
        features="date_column",
        grid_resolution=10,
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

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features="date_column",
        grid_resolution=10,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)

    part_dep = partial_dependence(pipeline, X, features=20, grid_resolution=10)
    if problem_type == "multiclass":
        assert len(part_dep["partial_dependence"]) == num_classes * 10
        assert len(part_dep["feature_values"]) == num_classes * 10
    else:
        assert len(part_dep["partial_dependence"]) == 10
        assert len(part_dep["feature_values"]) == 10
    assert not part_dep.isnull().any(axis=None)

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features=20,
        grid_resolution=10,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


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
        },
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
                f"Columns {expected_cols} are of types {expected_types}, which cannot be used for partial dependence",
            ),
        ):
            partial_dependence(pl, X, cols, grid_resolution=2)
        return
    s = partial_dependence(pl, X, cols, grid_resolution=2)
    assert not s.isnull().any().any()

    fast_s = partial_dependence(
        pl,
        X,
        cols,
        grid_resolution=2,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )

    pd.testing.assert_frame_equal(s, fast_s)


def test_partial_dependence_categorical_nan(fraud_100):
    X, y = fraud_100
    X.ww["provider"][:10] = None
    pl = BinaryClassificationPipeline(
        component_graph=[
            "Imputer",
            "DateTime Featurizer",
            "One Hot Encoder",
            "Random Forest Classifier",
        ],
    )
    pl.fit(X, y)

    GRID_RESOLUTION = 5
    dep = partial_dependence(
        pl,
        X,
        features="provider",
        grid_resolution=GRID_RESOLUTION,
    )

    assert dep.shape[0] == X["provider"].dropna().nunique()
    assert not dep["feature_values"].isna().any()
    assert not dep["partial_dependence"].isna().any()

    fast_dep = partial_dependence(
        pl,
        X,
        features="provider",
        grid_resolution=GRID_RESOLUTION,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(dep, fast_dep)

    dep2way = partial_dependence(
        pl,
        X,
        features=("amount", "provider"),
        grid_resolution=GRID_RESOLUTION,
    )
    assert not dep2way.isna().any().any()
    # Plus 1 in the columns because there is `class_label`
    assert dep2way.shape == (GRID_RESOLUTION, X["provider"].dropna().nunique() + 1)

    fast_dep2way = partial_dependence(
        pl,
        X,
        features=("amount", "provider"),
        grid_resolution=GRID_RESOLUTION,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(dep2way, fast_dep2way)


@patch(
    "evalml.pipelines.BinaryClassificationPipeline.predict_proba",
    side_effect=lambda X: np.array([[0.2, 0.8]] * X.shape[0]),
)
def test_partial_dependence_preserves_woodwork_schema(mock_predict_proba, fraud_100):

    X, y = fraud_100
    X_test = X.ww.copy()

    X = X.ww[["card_id", "store_id", "amount", "provider"]]
    X.ww.set_types({"provider": "NaturalLanguage"})

    pl = BinaryClassificationPipeline(
        component_graph={
            "Label Encoder": ["Label Encoder", "X", "y"],
            "Natural Language Featurizer": [
                "Natural Language Featurizer",
                "X",
                "Label Encoder.y",
            ],
            "Imputer": [
                "Imputer",
                "Natural Language Featurizer.x",
                "Label Encoder.y",
            ],
            "Random Forest Classifier": [
                "Random Forest Classifier",
                "Imputer.x",
                "Label Encoder.y",
            ],
        },
    )
    pl.fit(X, y)

    X_test = X_test.ww[["card_id", "store_id", "amount", "provider"]]
    X_test["provider"][-1] = None
    X_test.ww.set_types({"provider": "NaturalLanguage"})

    _ = partial_dependence(pl, X_test, "card_id", grid_resolution=5)
    assert all(
        call_args[0][0].ww.schema == X_test.ww.schema
        for call_args in mock_predict_proba.call_args_list
    )

    _ = partial_dependence(
        pl,
        X_test,
        "card_id",
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )

    assert all(
        call_args[0][0].ww.schema == X_test.ww.schema
        for call_args in mock_predict_proba.call_args_list
    )


def test_partial_dependence_does_not_return_all_nan_grid():
    # In this case, the 95th percentile of "a" if we included all values
    # would be NaN, so the resulting grid by np.linspace would be all NaN
    # This tests verifies that the grid is not all NaN
    X = pd.DataFrame({"a": [1, 2, None, 3, 3.2, 4.5, 2.3, 1.2], "b": [4, 5, 6, 7] * 2})
    y = pd.Series([1, 0, 0, 1] * 2)
    X_holdout = pd.DataFrame(
        {"a": [1, 2, None, 3, 3.2, 4.5, 2.3, 1.2], "b": [4, 5, 6, 7] * 2},
    )

    pipeline = BinaryClassificationPipeline(
        component_graph={
            "Label Encoder": ["Label Encoder", "X", "y"],
            "Imputer": ["Imputer", "X", "Label Encoder.y"],
            "Random Forest Classifier": [
                "Random Forest Classifier",
                "Imputer.x",
                "Label Encoder.y",
            ],
        },
    )
    pipeline.fit(X, y)

    dep = partial_dependence(pipeline, X_holdout, "a", grid_resolution=4)
    assert not dep.feature_values.isna().any()
    assert not dep.partial_dependence.isna().any()

    fast_dep = partial_dependence(
        pipeline,
        X_holdout,
        "a",
        grid_resolution=4,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(dep, fast_dep)


@patch("evalml.model_understanding.partial_dependence_functions.jupyter_check")
@patch("evalml.model_understanding.partial_dependence_functions.import_or_raise")
def test_partial_dependence_jupyter_graph_check(
    import_check,
    jupyter_check,
    X_y_binary,
    X_y_regression,
    logistic_regression_binary_pipeline,
):
    X, y = X_y_binary
    X = X.ww.iloc[:20, :5]
    y = y.ww.iloc[:20]
    logistic_regression_binary_pipeline.fit(X, y)

    jupyter_check.return_value = True
    with pytest.warns(None) as graph_valid:
        graph_partial_dependence(
            logistic_regression_binary_pipeline,
            X,
            features=0,
            grid_resolution=20,
        )
        assert len(graph_valid) == 0
        import_check.assert_called_with("ipywidgets", warning=True)


def test_partial_dependence_fast_mode_after_dropped_feature(X_y_categorical_regression):
    """The feature selector component in this test will drop the 'time' feature, so confirm
    that fast mode handles the dropping of this feature correctly--namely that the feature
    has no impact on predictions"""
    X, y = X_y_categorical_regression

    pipeline = RegressionPipeline(
        [
            "Imputer",
            "One Hot Encoder",
            "RF Regressor Select From Model",
            "Random Forest Regressor",
        ],
    )
    pipeline.fit(X, y)

    original_predictions = pipeline.predict(X)
    average_original_predictions = np.mean(original_predictions, axis=0)

    part_dep = partial_dependence(pipeline, X, features="time")
    assert all(
        np.isclose(average_original_predictions, dependence)
        for dependence in list(part_dep["partial_dependence"])
    )

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features="time",
        fast_mode=True,
        X_train=X,
        y_train=y,
    )

    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.BINARY, ProblemTypes.REGRESSION],
)
def test_partial_dependence_fast_mode_after_dropped_grid_value(
    problem_type,
    X_y_categorical_regression,
    X_y_categorical_classification,
):
    """The feature selector component in this test will drop some of the categories in the 'day'
    column after they are one-hot encoded. This means that we lose some grid values based on their
    feature importance, which is problematic when we fit a pipeline on just the single 'day' feature
    for fast mode. This test confirms that fast mode isn't improperly dropping categories based
    off of the single column."""
    if problem_type == ProblemTypes.REGRESSION:
        feature = "day"
        pipeline = RegressionPipeline(
            [
                "Imputer",
                "One Hot Encoder",
                "RF Regressor Select From Model",
                "Random Forest Regressor",
            ],
        )
        X, y = X_y_categorical_regression
    elif problem_type == ProblemTypes.BINARY:
        feature = "Cabin"
        pipeline = BinaryClassificationPipeline(
            [
                "Imputer",
                "One Hot Encoder",
                "RF Classifier Select From Model",
                "Random Forest Classifier",
            ],
        )
        X, y = X_y_categorical_classification

    pipeline.fit(X, y)

    # Confirm some categories are dropped from the day column
    assert len(pipeline._get_feature_provenance()[feature]) != len(X[feature].unique())

    part_dep = partial_dependence(pipeline, X, features=feature)
    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features=feature,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )

    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


def test_pd_fast_mode_select_cols_transformer_specified_feature_not_selected():
    """Confirms that If a column isn't in the column selector but is the specified feature to
    calculate partial dependence for, that the results are as expected for the optimized implementation.
    """
    y = pd.Series([1.3, 0.7, 1.2, 1.0] * 12)
    X = pd.DataFrame(
        {
            "cats": ["a", "b", "c"] * 16,
            "nums": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 4,
        },
    )
    X.ww.init(logical_types={"cats": "categorical"})
    pipeline = RegressionPipeline(
        [SelectColumns(columns=["nums"]), "One Hot Encoder", "Linear Regressor"],
    )

    pipeline.fit(X, y)

    part_dep = partial_dependence(pipeline, X, features="cats", grid_resolution=5)

    original_predictions = pipeline.predict(X)
    average_original_predictions = np.mean(original_predictions, axis=0)

    assert all(
        np.isclose(average_original_predictions, dependence)
        for dependence in list(part_dep["partial_dependence"])
    )

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features="cats",
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


def test_pd_fast_mode_drop_cols_transformer_specified_feature_not_selected():
    """Confirms that if a column is dropped by the column dropper but is the specified feature to
    calculate partial dependence for, that the results are as expected for the optimized implementation.
    """
    y = pd.Series([1.3, 0.7, 1.2, 1.0] * 12)
    X = pd.DataFrame(
        {
            "cats": ["a", "b", "c"] * 16,
            "nums": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 4,
        },
    )
    X.ww.init(logical_types={"cats": "categorical"})
    pipeline = RegressionPipeline(
        [DropColumns(columns=["cats"]), "One Hot Encoder", "Linear Regressor"],
    )

    pipeline.fit(X, y)

    part_dep = partial_dependence(pipeline, X, features="cats", grid_resolution=5)

    original_predictions = pipeline.predict(X)
    average_original_predictions = np.mean(original_predictions, axis=0)

    assert all(
        np.isclose(average_original_predictions, dependence)
        for dependence in list(part_dep["partial_dependence"])
    )

    fast_part_dep = partial_dependence(
        pipeline,
        X,
        features="cats",
        grid_resolution=5,
        fast_mode=True,
        X_train=X,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


def test_pd_dfs_transformer_fast_mode_works_only_when_features_present(X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X.columns = X.columns.astype(str)

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X,
        index="index",
        make_index=True,
    )
    X_fm, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    dfs_transformer = DFSTransformer(features=features)
    pipeline = BinaryClassificationPipeline(
        [dfs_transformer, "Standard Scaler", "Random Forest Classifier"],
    )

    # When fit on X, the features expected by the dfs transformer aren't present, so they're created via
    # calculate feature matrix, and pd won't work with dfs transformer, for two reasons
    # 1. It doesn't create a feature provenance, so only the base feature gets updated with new pd values - not any engineered features
    # 2. Any multi input features wouldn't get created, because the other inputs wouldn't be present
    pipeline.fit(X, y)
    error = "Cannot use fast mode with DFS Transformer when features are unspecified or not all present in X."
    with pytest.raises(
        PartialDependenceError,
        match=error,
    ):
        partial_dependence(
            pipeline,
            X,
            features=1,
            grid_resolution=5,
            fast_mode=True,
            X_train=X,
            y_train=y,
        )

    # If we pass the feature matrix into the same pipeline, though, DFS transformer will be no op, so pd should match
    pipeline = pipeline.clone()
    pipeline.fit(X_fm, y)
    part_dep = partial_dependence(pipeline, X_fm, features=1, grid_resolution=5)
    fast_part_dep = partial_dependence(
        pipeline,
        X_fm,
        features=1,
        grid_resolution=5,
        fast_mode=True,
        X_train=X_fm,
        y_train=y,
    )
    pd.testing.assert_frame_equal(part_dep, fast_part_dep)


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.REGRESSION])
def test_partial_dependence_fast_mode_ensemble_pipeline_blocked(
    problem_type,
    X_y_binary,
    X_y_regression,
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        input_pipelines = [
            BinaryClassificationPipeline(
                ["Standard Scaler", "Random Forest Classifier"],
            ),
            BinaryClassificationPipeline(["Standard Scaler", "Elastic Net Classifier"]),
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
    )
    pipeline.fit(X, y)

    error = "cannot run partial dependence fast mode"
    with pytest.raises(
        PartialDependenceError,
        match=error,
    ):
        partial_dependence(
            pipeline,
            X,
            features=0,
            grid_resolution=5,
            fast_mode=True,
            X_train=X,
            y_train=y,
        )


def test_partial_dependence_fast_mode_oversampler_blocked(X_y_binary):
    X, y = X_y_binary
    pipeline = BinaryClassificationPipeline(
        component_graph={
            "Oversampler": ["Oversampler", "X", "y"],
            "Standard Scaler": ["Standard Scaler", "Oversampler.x", "Oversampler.y"],
            "Logistic Regression Classifier": [
                "Logistic Regression Classifier",
                "Standard Scaler.x",
                "Oversampler.y",
            ],
        },
    )

    pipeline.fit(X, y)
    error = "cannot run partial dependence fast mode"
    with pytest.raises(
        PartialDependenceError,
        match=error,
    ):
        partial_dependence(
            pipeline,
            X,
            features=0,
            grid_resolution=5,
            fast_mode=True,
            X_train=X,
            y_train=y,
        )


def test_partial_dependence_fast_mode_errors_if_train(
    X_y_regression,
    linear_regression_pipeline,
):
    X, y = X_y_regression

    linear_regression_pipeline.fit(X, y)
    error = "Training data is required for partial dependence fast mode."
    with pytest.raises(
        PartialDependenceError,
        match=error,
    ):
        partial_dependence(
            linear_regression_pipeline,
            X,
            features=0,
            fast_mode=True,
            X_train=X,
        )

    with pytest.raises(
        PartialDependenceError,
        match=error,
    ):
        partial_dependence(
            linear_regression_pipeline,
            X,
            features=0,
            fast_mode=True,
            y_train=y,
        )


@pytest.mark.parametrize("fast_mode", [True, False])
def test_partial_dependence_on_engineered_feature_with_dfs_transformer(
    fast_mode,
    X_y_binary,
):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X.columns = X.columns.astype(str)

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X,
        index="index",
        make_index=True,
    )
    X_fm, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    dfs_transformer = DFSTransformer(features=features)
    pipeline = BinaryClassificationPipeline(
        [dfs_transformer, "Standard Scaler", "Random Forest Classifier"],
    )

    # Engineered features have the their origins specified as either "base" or "engineered"
    # it has to remain set for partial dependence to be able to predict on the updated data
    engineered_feature = "ABSOLUTE(1)"
    assert X_fm.ww.columns[engineered_feature].origin == "engineered"

    pipeline.fit(X_fm, y)
    part_dep = partial_dependence(
        pipeline,
        X_fm,
        features=engineered_feature,
        grid_resolution=2,
        fast_mode=fast_mode,
        X_train=X_fm,
        y_train=y,
    )

    assert part_dep.feature_values.notnull().all()
    assert part_dep.partial_dependence.notnull().all()


@pytest.mark.parametrize("fast_mode", [True, False])
def test_partial_dependence_dfs_transformer_handling_with_multi_output_primitive(
    fast_mode,
    df_with_url_and_email,
):
    X = df_with_url_and_email
    y = pd.Series(range(len(X)))
    X.ww.name = "X"
    X.ww.set_index("numeric")
    X.ww.set_types(logical_types={"categorical": "NaturalLanguage"})

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X,
        index="index",
        make_index=True,
    )
    X_fm, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["LSA"],
    )

    dfs_transformer = DFSTransformer(features=features)
    pipeline = RegressionPipeline(
        [dfs_transformer, "Standard Scaler", "Random Forest Regressor"],
    )
    # Confirm that a multi-output feature is present
    assert any(f.number_output_features > 1 for f in features)

    pipeline.fit(X_fm, y)
    part_dep = partial_dependence(
        pipeline,
        X_fm,
        features=0,
        grid_resolution=2,
        fast_mode=fast_mode,
        X_train=X_fm,
        y_train=y,
    )

    assert part_dep.feature_values.notnull().all()
    assert part_dep.partial_dependence.notnull().all()


@pytest.mark.parametrize("fast_mode", [True, False])
def test_partial_dependence_dfs_transformer_target_in_features(fast_mode, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X.columns = X.columns.astype(str)

    # Insert y into X so that it's part of the EntitySet
    # and then ignore in DFS later on so it's not in X_fm
    X["target"] = y

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X,
        index="index",
        make_index=True,
    )
    seed_features = [ft.Feature(es["X"].ww["target"])]
    X_fm, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
        ignore_columns={"X": ["target"]},
        seed_features=seed_features,
    )
    assert any(f.get_name() == "target" for f in features)

    dfs_transformer = DFSTransformer(features=features)
    pipeline = BinaryClassificationPipeline(
        [dfs_transformer, "Standard Scaler", "Random Forest Classifier"],
    )

    pipeline.fit(X_fm, y)
    part_dep = partial_dependence(
        pipeline,
        X_fm,
        features=0,
        grid_resolution=2,
        fast_mode=fast_mode,
        X_train=X_fm,
        y_train=y,
    )

    assert part_dep.feature_values.notnull().all()
    assert part_dep.partial_dependence.notnull().all()
