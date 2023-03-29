import inspect

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import MissingComponentError
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
)
from evalml.pipelines.components import ComponentBase, RandomForestClassifier
from evalml.pipelines.components.utils import (
    _all_estimators,
    all_components,
    estimator_unable_to_handle_nans,
    handle_component_class,
    handle_float_categories_for_catboost,
    make_balancing_dictionary,
    scikit_learn_wrapped_estimator,
)
from evalml.problem_types import ProblemTypes

binary = pd.Series([0] * 800 + [1] * 200)
multiclass = pd.Series([0] * 800 + [1] * 150 + [2] * 50)


all_requirements_set = set(
    [
        "Baseline Classifier",
        "Baseline Regressor",
        "DFS Transformer",
        "DateTime Featurizer",
        "Decision Tree Classifier",
        "Decision Tree Regressor",
        "Time Series Featurizer",
        "Drop Columns Transformer",
        "Drop Null Columns Transformer",
        "Drop Rows Transformer",
        "Drop NaN Rows Transformer",
        "Elastic Net Classifier",
        "Elastic Net Regressor",
        "Email Featurizer",
        "Extra Trees Classifier",
        "Extra Trees Regressor",
        "Imputer",
        "KNN Classifier",
        "KNN Imputer",
        "LSA Transformer",
        "Label Encoder",
        "Linear Discriminant Analysis Transformer",
        "Linear Regressor",
        "Log Transformer",
        "Logistic Regression Classifier",
        "One Hot Encoder",
        "PCA Transformer",
        "Per Column Imputer",
        "RF Classifier Select From Model",
        "RF Regressor Select From Model",
        "RFE Selector with RF Classifier",
        "RFE Selector with RF Regressor",
        "Random Forest Classifier",
        "Random Forest Regressor",
        "Replace Nullable Types Transformer",
        "SVM Classifier",
        "SVM Regressor",
        "Select Columns By Type Transformer",
        "Select Columns Transformer",
        "Simple Imputer",
        "Stacked Ensemble Classifier",
        "Stacked Ensemble Regressor",
        "Standard Scaler",
        "Target Imputer",
        "Natural Language Featurizer",
        "Time Series Baseline Estimator",
        "Time Series Imputer",
        "Time Series Regularizer",
        "URL Featurizer",
        "Undersampler",
        "ARIMA Regressor",
        "Exponential Smoothing Regressor",
        "CatBoost Classifier",
        "CatBoost Regressor",
        "LightGBM Classifier",
        "LightGBM Regressor",
        "Oversampler",
        "Ordinal Encoder",
        "Polynomial Decomposer",
        "STL Decomposer",
        "Prophet Regressor",
        "Target Encoder",
        "Vowpal Wabbit Binary Classifier",
        "Vowpal Wabbit Multiclass Classifier",
        "Vowpal Wabbit Regressor",
        "XGBoost Classifier",
        "XGBoost Regressor",
    ],
)
not_supported_in_conda = set(
    [
        "Prophet Regressor",
    ],
)

# Keeping here in case we need to add to it when a new component is added
not_supported_in_linux_py39 = set()
not_supported_in_windows = set()
not_supported_in_windows_py39 = set()


def test_all_components(
    is_using_conda,
):
    if is_using_conda:
        # No prophet
        expected_components = all_requirements_set.difference(
            not_supported_in_conda,
        )
    else:
        expected_components = all_requirements_set
    all_component_names = [component.name for component in all_components()]
    assert set(all_component_names) == expected_components


def test_handle_component_class_names():
    for cls in all_components():
        cls_ret = handle_component_class(cls)
        assert inspect.isclass(cls_ret)
        assert issubclass(cls_ret, ComponentBase)
        name_ret = handle_component_class(cls.name)
        assert inspect.isclass(name_ret)
        assert issubclass(name_ret, ComponentBase)

    invalid_name = "This Component Does Not Exist"
    with pytest.raises(
        MissingComponentError,
        match='Component "This Component Does Not Exist" was not found',
    ):
        handle_component_class(invalid_name)

    class NonComponent:
        pass

    with pytest.raises(ValueError):
        handle_component_class(NonComponent())


def test_scikit_learn_wrapper_invalid_problem_type():
    evalml_pipeline = MulticlassClassificationPipeline(
        [RandomForestClassifier],
    )
    evalml_pipeline.problem_type = None
    with pytest.raises(
        ValueError,
        match="Could not wrap EvalML object in scikit-learn wrapper.",
    ):
        scikit_learn_wrapped_estimator(evalml_pipeline)


def test_scikit_learn_wrapper(X_y_binary, X_y_multi, X_y_regression):
    for estimator in [
        estimator
        for estimator in _all_estimators()
        if estimator.model_family != ModelFamily.ENSEMBLE
    ]:
        for problem_type in estimator.supported_problem_types:
            if problem_type == ProblemTypes.BINARY:
                X, y = X_y_binary
                num_classes = 2
                pipeline_class = BinaryClassificationPipeline
            elif problem_type == ProblemTypes.MULTICLASS:
                X, y = X_y_multi
                num_classes = 3
                pipeline_class = MulticlassClassificationPipeline
            elif problem_type == ProblemTypes.REGRESSION:
                X, y = X_y_regression
                pipeline_class = RegressionPipeline

            elif problem_type in [
                ProblemTypes.TIME_SERIES_REGRESSION,
                ProblemTypes.TIME_SERIES_MULTICLASS,
                ProblemTypes.TIME_SERIES_BINARY,
            ]:
                continue

            evalml_pipeline = pipeline_class([estimator])
            scikit_estimator = scikit_learn_wrapped_estimator(evalml_pipeline)
            scikit_estimator.fit(X, y)
            y_pred = scikit_estimator.predict(X)
            assert len(y_pred) == len(y)
            assert not np.isnan(y_pred).all()
            if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
                y_pred_proba = scikit_estimator.predict_proba(X)
                assert y_pred_proba.shape == (len(y), num_classes)
                assert not np.isnan(y_pred_proba).all().all()


def test_make_balancing_dictionary_errors():
    with pytest.raises(ValueError, match="Sampling ratio must be in range"):
        make_balancing_dictionary(pd.Series([1]), 0)

    with pytest.raises(ValueError, match="Sampling ratio must be in range"):
        make_balancing_dictionary(pd.Series([1]), 1.1)

    with pytest.raises(ValueError, match="Sampling ratio must be in range"):
        make_balancing_dictionary(pd.Series([1]), -1)

    with pytest.raises(ValueError, match="Target data must not be empty"):
        make_balancing_dictionary(pd.Series([]), 0.5)


@pytest.mark.parametrize(
    "y,sampling_ratio,result",
    [
        (binary, 1, {0: 800, 1: 800}),
        (binary, 0.5, {0: 800, 1: 400}),
        (binary, 0.25, {0: 800, 1: 200}),
        (binary, 0.1, {0: 800, 1: 200}),
        (multiclass, 1, {0: 800, 1: 800, 2: 800}),
        (multiclass, 0.5, {0: 800, 1: 400, 2: 400}),
        (multiclass, 0.25, {0: 800, 1: 200, 2: 200}),
        (multiclass, 0.1, {0: 800, 1: 150, 2: 80}),
        (multiclass, 0.01, {0: 800, 1: 150, 2: 50}),
    ],
)
def test_make_balancing_dictionary(y, sampling_ratio, result):
    dic = make_balancing_dictionary(y, sampling_ratio)
    assert dic == result


def test_estimator_unable_to_handle_nans():
    test_estimator = RandomForestClassifier()
    assert estimator_unable_to_handle_nans(test_estimator) is True

    with pytest.raises(
        ValueError,
        match="`estimator_class` must have a `model_family` attribute.",
    ):
        estimator_unable_to_handle_nans("error")


def test_handle_float_categories_for_catboost(categorical_floats_df):
    X = categorical_floats_df
    X_t = handle_float_categories_for_catboost(X)

    # Since only the categories' changed, the woodwork schema should be equal before and after
    # But the dtype Series' shouldn't
    assert X.ww.schema == X_t.ww.schema
    assert not X.dtypes.equals(X_t.dtypes)

    expected_dtype_before_and_after = {
        "double_int_cats": ("float64", "int64"),
        # These shouldn't change
        "string_cats": None,
        "int_cats": None,
        "int_col": None,
        "double_col": None,
    }

    for col in X.columns:
        if before_and_after := expected_dtype_before_and_after.get(col):
            before_dtype, after_dtype = before_and_after
            assert X.dtypes[col].categories.dtype == before_dtype
            assert X_t.dtypes[col].categories.dtype == after_dtype
            # Confirm that the numeric values are still equal - we didn't truncate anything
            for i in range(len(X)):
                assert X[col].iloc[i] == float(X_t[col].iloc[i])
        else:
            pd.testing.assert_series_equal(X[col], X_t[col])


def test_handle_float_categories_for_catboost_actual_floats():
    X = pd.DataFrame({"really_double_cats": pd.Series([1.2, 2.3, 3.9, 4.1, 5.5] * 20)})
    X.ww.init(logical_types={"really_double_cats": "Categorical"})

    error = "CatBoost does not support floats as categories."
    with pytest.raises(ValueError, match=error):
        handle_float_categories_for_catboost(X)


def test_handle_float_categories_for_catboost_noop(
    categorical_floats_df,
):
    X = categorical_floats_df.ww[["string_cats", "int_col", "int_cats"]]

    X_t = handle_float_categories_for_catboost(X)
    pd.testing.assert_frame_equal(X, X_t)
    assert X.ww.schema == X_t.ww.schema
