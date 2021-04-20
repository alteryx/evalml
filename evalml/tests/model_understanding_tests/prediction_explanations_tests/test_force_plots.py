from itertools import product

import pandas as pd
import pytest
import shap

from evalml.demos import load_breast_cancer, load_wine
from evalml.model_family.model_family import ModelFamily
from evalml.model_understanding.graphs import force_plot
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline
)
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import is_classification, is_regression
from evalml.problem_types.problem_types import ProblemTypes
from evalml.tests.model_understanding_tests.prediction_explanations_tests.test_algorithms import (
    all_problems,
    interpretable_estimators
)


def test_exceptions():
    with pytest.raises(TypeError, match="rows_to_explain should be provided as a list of row index integers!"):
        force_plot(None, None, None)
    with pytest.raises(TypeError, match="rows_to_explain should only contain integers!"):
        force_plot(pipeline=None, rows_to_explain=["this", "party's", "over"], training_data=None)


@pytest.mark.parametrize("estimator,problem_type,n_points_to_explain",
                         product(interpretable_estimators, all_problems, [[0], [0, 1, 2, 3, 4]]))
def test_plot(estimator, problem_type, n_points_to_explain, X_y_binary, X_y_multi, X_y_regression,
              helper_functions, has_minimal_dependencies):
    if has_minimal_dependencies:
        pytest.skip("Skipping because plotly not installed for minimal dependencies")
    if problem_type not in estimator.supported_problem_types:
        pytest.skip("Skipping because estimator and pipeline are not compatible.")

    if problem_type == ProblemTypes.MULTICLASS and estimator.model_family == ModelFamily.CATBOOST:
        pytest.skip("Skipping Catboost for multiclass problems.")

    if problem_type == ProblemTypes.BINARY:
        training_data, y = X_y_binary
    elif problem_type == ProblemTypes.MULTICLASS:
        training_data, y = X_y_multi
    else:
        training_data, y = X_y_regression

    training_data = pd.DataFrame(training_data)

    pipeline_class = make_pipeline(training_data, y, estimator, problem_type)
    pipeline = helper_functions.safe_init_pipeline_with_njobs_1(pipeline_class)

    pipeline.fit(training_data, y)

    results = force_plot(pipeline, n_points_to_explain, training_data)

    if len(n_points_to_explain) == 1:
        expected_plot_class = shap.plots._force.AdditiveForceVisualizer
    else:
        expected_plot_class = shap.plots._force.AdditiveForceArrayVisualizer

    if is_classification(pipeline.problem_type):
        # Should have one result per class.
        assert len(results) == len(set(y))

        # Should have integer labeled classes.
        classes = [r["class"] for r in results]
        assert classes == [x for x in range(len(set(y)))]

    elif is_regression(pipeline.problem_type):
        # Should have one result per class.
        assert len(results) == 1

        # Should have integer labeled classes.
        classes = [r["class"] for r in results]
        assert classes == ["regression"]

    # Should have a force plot in each dictionary.
    force_plots = [r["force_plot"] for r in results]
    assert all([isinstance(fp, expected_plot_class) for fp in force_plots])


@pytest.mark.parametrize("rows_to_explain", [[0], [0, 1, 2, 3, 4]])
def test_force_plot_binary(rows_to_explain, has_minimal_dependencies):
    if has_minimal_dependencies:
        pytest.skip("Skipping because plotly not installed for minimal dependencies")
    if len(rows_to_explain) == 1:
        expected_plot_class = shap.plots._force.AdditiveForceVisualizer
    else:
        expected_plot_class = shap.plots._force.AdditiveForceArrayVisualizer
    X, y = load_breast_cancer()

    class RFBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'Random Forest Classifier']

    pipeline = RFBinaryClassificationPipeline({})
    pipeline.fit(X, y)

    results = force_plot(pipeline, rows_to_explain=rows_to_explain, training_data=X.df, matplotlib=False)

    # Should have one result per class.
    assert len(results) == 2

    # Should have integer labeled classes.
    classes = [r["class"] for r in results]
    assert classes == [0, 1]

    # Should have a force plot in each dictionary.
    force_plots = [r["force_plot"] for r in results]
    assert all([isinstance(fp, expected_plot_class) for fp in force_plots])


@pytest.mark.parametrize("rows_to_explain", [[0], [0, 1, 2, 3, 4]])
def test_force_plot_multiclass(rows_to_explain, has_minimal_dependencies):
    if has_minimal_dependencies:
        pytest.skip("Skipping because plotly not installed for minimal dependencies")
    if len(rows_to_explain) == 1:
        expected_plot_class = shap.plots._force.AdditiveForceVisualizer
    else:
        expected_plot_class = shap.plots._force.AdditiveForceArrayVisualizer

    X, y = load_wine()

    class RFMultiClassificationPipeline(MulticlassClassificationPipeline):
        component_graph = ['Simple Imputer', 'Random Forest Classifier']

    pipeline = RFMultiClassificationPipeline({})
    pipeline.fit(X, y)

    results = force_plot(pipeline, rows_to_explain=rows_to_explain, training_data=X.df, matplotlib=False)

    # Should have one result per class.
    assert len(results) == 3

    # Should have integer labeled classes.
    classes = [r["class"] for r in results]
    assert classes == [0, 1, 2]

    # Should have a force plot in each dictionary.
    force_plots = [r["force_plot"] for r in results]
    assert all([isinstance(fp, expected_plot_class) for fp in force_plots])


@pytest.mark.parametrize("rows_to_explain", [[0], [0, 1, 2, 3, 4]])
def test_force_plot_regression(rows_to_explain, X_y_regression, has_minimal_dependencies):
    if has_minimal_dependencies:
        pytest.skip("Skipping because plotly not installed for minimal dependencies")
    if len(rows_to_explain) == 1:
        expected_plot_class = shap.plots._force.AdditiveForceVisualizer
    else:
        expected_plot_class = shap.plots._force.AdditiveForceArrayVisualizer

    X, y = X_y_regression

    class TestRegressionPipeline(RegressionPipeline):
        component_graph = ['Simple Imputer', 'LightGBM Regressor']

    pipeline = TestRegressionPipeline({})
    pipeline.fit(X, y)

    results = force_plot(pipeline, rows_to_explain=[0], training_data=pd.DataFrame(X), matplotlib=False)

    assert len(results) == 1  # Should have a single force plot.
    assert results[0]["class"] == "regression"
    assert isinstance(results[0]["force_plot"], expected_plot_class)
