import os

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn import datasets
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.objectives.utils import (
    get_core_objectives,
    get_non_core_objectives
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline
)
from evalml.pipelines.components import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Estimator,
    LogisticRegressionClassifier,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor
)
from evalml.pipelines.components.ensemble.stacked_ensemble_base import (
    _nonstackable_model_families
)
from evalml.pipelines.components.utils import _all_estimators
from evalml.problem_types import ProblemTypes, handle_problem_types


def create_mock_pipeline(estimator, problem_type):
    if problem_type == ProblemTypes.BINARY:
        class MockBinaryPipelineWithOnlyEstimator(BinaryClassificationPipeline):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]
        return MockBinaryPipelineWithOnlyEstimator
    elif problem_type == ProblemTypes.MULTICLASS:
        class MockMulticlassPipelineWithOnlyEstimator(MulticlassClassificationPipeline):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]
        return MockMulticlassPipelineWithOnlyEstimator
    elif problem_type == ProblemTypes.REGRESSION:
        class MockRegressionPipelineWithOnlyEstimator(RegressionPipeline):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]
        return MockRegressionPipelineWithOnlyEstimator
    elif problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        class MockTSRegressionPipelineWithOnlyEstimator(TimeSeriesRegressionPipeline):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]
        return MockTSRegressionPipelineWithOnlyEstimator
    elif problem_type == ProblemTypes.TIME_SERIES_BINARY:
        class MockTSRegressionPipelineWithOnlyEstimator(TimeSeriesBinaryClassificationPipeline):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]
        return MockTSRegressionPipelineWithOnlyEstimator
    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:
        class MockTSRegressionPipelineWithOnlyEstimator(TimeSeriesMulticlassClassificationPipeline):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]
        return MockTSRegressionPipelineWithOnlyEstimator


@pytest.fixture
def all_pipeline_classes():
    all_possible_pipeline_classes = []
    for estimator in [estimator for estimator in _all_estimators() if estimator != StackedEnsembleClassifier and estimator != StackedEnsembleRegressor]:
        for problem_type in estimator.supported_problem_types:
            all_possible_pipeline_classes.append(create_mock_pipeline(estimator, problem_type))
    return all_possible_pipeline_classes


@pytest.fixture
def all_binary_pipeline_classes(all_pipeline_classes):
    return [pipeline_class for pipeline_class in all_pipeline_classes if issubclass(pipeline_class, BinaryClassificationPipeline)]


@pytest.fixture
def all_multiclass_pipeline_classes(all_pipeline_classes):
    return [pipeline_class for pipeline_class in all_pipeline_classes if issubclass(pipeline_class, MulticlassClassificationPipeline)]


def pytest_addoption(parser):
    parser.addoption("--has-minimal-dependencies", action="store_true", default=False,
                     help="If true, tests will assume only the dependencies in"
                     "core-requirements.txt have been installed.")


@pytest.fixture
def has_minimal_dependencies(pytestconfig):
    return pytestconfig.getoption("--has-minimal-dependencies")


@pytest.fixture
def assert_allowed_pipelines_equal_helper():
    def assert_allowed_pipelines_equal_helper(actual_allowed_pipelines, expected_allowed_pipelines):
        for actual, expected in zip(actual_allowed_pipelines, expected_allowed_pipelines):
            for pipeline_subclass in [BinaryClassificationPipeline, MulticlassClassificationPipeline, RegressionPipeline]:
                if issubclass(expected, pipeline_subclass):
                    assert issubclass(expected, pipeline_subclass)
                    break
            assert actual.parameters == expected.parameters
            assert actual.name == expected.name
            assert actual.problem_type == expected.problem_type
            assert actual.component_graph == expected.component_graph
    return assert_allowed_pipelines_equal_helper


@pytest.fixture
def X_y_binary():
    X, y = datasets.make_classification(n_samples=100, n_features=20,
                                        n_informative=2, n_redundant=2, random_state=0)

    return X, y


@pytest.fixture
def X_y_regression():
    X, y = datasets.make_regression(n_samples=100, n_features=20,
                                    n_informative=3, random_state=0)
    return X, y


@pytest.fixture
def X_y_multi():
    X, y = datasets.make_classification(n_samples=100, n_features=20, n_classes=3,
                                        n_informative=3, n_redundant=2, random_state=0)
    return X, y


@pytest.fixture
def X_y_categorical_regression():
    data_path = os.path.join(os.path.dirname(__file__), "data/tips.csv")
    flights = pd.read_csv(data_path)

    y = flights['tip']
    X = flights.drop('tip', axis=1)

    # add categorical dtype
    X['smoker'] = X['smoker'].astype('category')
    return X, y


@pytest.fixture
def X_y_categorical_classification():
    data_path = os.path.join(os.path.dirname(__file__), "data/titanic.csv")
    titanic = pd.read_csv(data_path)

    y = titanic['Survived']
    X = titanic.drop('Survived', axis=1)
    return X, y


@pytest.fixture
def ts_data():
    X, y = pd.DataFrame({"features": range(101, 132)}), pd.Series(range(1, 32))
    y.index = pd.date_range("2020-10-01", "2020-10-31")
    X.index = pd.date_range("2020-10-01", "2020-10-31")
    return X, y


@pytest.fixture
def dummy_pipeline_hyperparameters():
    return {'Mock Classifier': {
        'param a': Integer(0, 10),
        'param b': Real(0, 10),
        'param c': ['option a', 'option b', 'option c'],
        'param d': ['option a', 'option b', 100, np.inf]
    }}


@pytest.fixture
def dummy_pipeline_hyperparameters_unicode():
    return {'Mock Classifier': {
        'param a': Integer(0, 10),
        'param b': Real(0, 10),
        'param c': ['option a 💩', 'option b 💩', 'option c 💩'],
        'param d': ['option a', 'option b', 100, np.inf]
    }}


@pytest.fixture
def dummy_pipeline_hyperparameters_small():
    return {'Mock Classifier': {
        'param a': ['most_frequent', 'median', 'mean'],
        'param b': ['a', 'b', 'c']
    }}


@pytest.fixture
def dummy_classifier_estimator_class():
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                                   ProblemTypes.TIME_SERIES_MULTICLASS, ProblemTypes.TIME_SERIES_BINARY]
        hyperparameter_ranges = {'a': Integer(0, 10),
                                 'b': Real(0, 10)}

        def __init__(self, a=1, b=0, random_state=0):
            super().__init__(parameters={"a": a, "b": b}, component_obj=None, random_state=random_state)

        def fit(self, X, y):
            return self

    return MockEstimator


@pytest.fixture
def dummy_binary_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockBinaryClassificationPipeline(BinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator]

    return MockBinaryClassificationPipeline


@pytest.fixture
def dummy_multiclass_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockMulticlassClassificationPipeline(MulticlassClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator]

    return MockMulticlassClassificationPipeline


@pytest.fixture
def dummy_regressor_estimator_class():
    class MockRegressor(Estimator):
        name = "Mock Regressor"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.REGRESSION]
        hyperparameter_ranges = {'a': Integer(0, 10),
                                 'b': Real(0, 10)}

        def __init__(self, a=1, b=0, random_state=0):
            super().__init__(parameters={"a": a, "b": b}, component_obj=None, random_state=random_state)

        def fit(self, X, y):
            return self

    return MockRegressor


@pytest.fixture
def dummy_regression_pipeline_class(dummy_regressor_estimator_class):
    MockRegressor = dummy_regressor_estimator_class

    class MockRegressionPipeline(RegressionPipeline):
        component_graph = [MockRegressor]
    return MockRegressionPipeline


@pytest.fixture
def dummy_time_series_regressor_estimator_class():
    class MockTimeSeriesRegressor(Estimator):
        name = "Mock Time Series Regressor"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
        hyperparameter_ranges = {'a': Integer(0, 10),
                                 'b': Real(0, 10)}

        def __init__(self, a=1, b=0, random_state=0):
            super().__init__(parameters={"a": a, "b": b}, component_obj=None, random_state=random_state)

    return MockTimeSeriesRegressor


@pytest.fixture
def dummy_time_series_regression_pipeline_class(dummy_time_series_regressor_estimator_class):
    MockTimeSeriesRegressor = dummy_time_series_regressor_estimator_class

    class MockTimeSeriesRegressionPipeline(TimeSeriesRegressionPipeline):
        component_graph = [MockTimeSeriesRegressor]
    return MockTimeSeriesRegressionPipeline


@pytest.fixture
def dummy_ts_binary_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockBinaryClassificationPipeline(TimeSeriesBinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator]

    return MockBinaryClassificationPipeline


@pytest.fixture
def logistic_regression_multiclass_pipeline_class():
    class LogisticRegressionMulticlassPipeline(MulticlassClassificationPipeline):
        """Logistic Regression Pipeline for binary classification."""
        component_graph = ['Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']
    return LogisticRegressionMulticlassPipeline


@pytest.fixture
def logistic_regression_binary_pipeline_class():
    class LogisticRegressionBinaryPipeline(BinaryClassificationPipeline):
        component_graph = ['Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']
    return LogisticRegressionBinaryPipeline


@pytest.fixture
def linear_regression_pipeline_class():
    class LinearRegressionPipeline(RegressionPipeline):
        """Linear Regression Pipeline for regression problems."""
        component_graph = ['One Hot Encoder', 'Imputer', 'Standard Scaler', 'Linear Regressor']
    return LinearRegressionPipeline


@pytest.fixture
def decision_tree_classification_pipeline_class(X_y_categorical_classification):
    class DTBinaryClassificationPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Decision Tree Classifier']
    pipeline = DTBinaryClassificationPipeline({})
    X, y = X_y_categorical_classification
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def nonlinear_binary_pipeline_class():
    class NonLinearBinaryPipeline(BinaryClassificationPipeline):
        component_graph = {
            'Imputer': ['Imputer'],
            'OneHot_RandomForest': ['One Hot Encoder', 'Imputer.x'],
            'OneHot_ElasticNet': ['One Hot Encoder', 'Imputer.x'],
            'Random Forest': ['Random Forest Classifier', 'OneHot_RandomForest.x'],
            'Elastic Net': ['Elastic Net Classifier', 'OneHot_ElasticNet.x'],
            'Logistic Regression': ['Logistic Regression Classifier', 'Random Forest', 'Elastic Net']
        }
    return NonLinearBinaryPipeline


@pytest.fixture
def nonlinear_multiclass_pipeline_class():
    class NonLinearMulticlassPipeline(MulticlassClassificationPipeline):
        component_graph = {
            'Imputer': ['Imputer'],
            'OneHot_RandomForest': ['One Hot Encoder', 'Imputer.x'],
            'OneHot_ElasticNet': ['One Hot Encoder', 'Imputer.x'],
            'Random Forest': ['Random Forest Classifier', 'OneHot_RandomForest.x'],
            'Elastic Net': ['Elastic Net Classifier', 'OneHot_ElasticNet.x'],
            'Logistic Regression': ['Logistic Regression Classifier', 'Random Forest', 'Elastic Net']
        }
    return NonLinearMulticlassPipeline


@pytest.fixture
def nonlinear_regression_pipeline_class():
    class NonLinearRegressionPipeline(RegressionPipeline):
        component_graph = {
            'Imputer': ['Imputer'],
            'OneHot': ['One Hot Encoder', 'Imputer.x'],
            'Random Forest': ['Random Forest Regressor', 'OneHot.x'],
            'Elastic Net': ['Elastic Net Regressor', 'OneHot.x'],
            'Linear Regressor': ['Linear Regressor', 'Random Forest', 'Elastic Net']
        }
    return NonLinearRegressionPipeline


@pytest.fixture
def binary_core_objectives():
    return get_core_objectives(ProblemTypes.BINARY)


@pytest.fixture
def multiclass_core_objectives():
    return get_core_objectives(ProblemTypes.MULTICLASS)


@pytest.fixture
def regression_core_objectives():
    return get_core_objectives(ProblemTypes.REGRESSION)


@pytest.fixture
def time_series_core_objectives():
    return get_core_objectives(ProblemTypes.TIME_SERIES_REGRESSION)


@pytest.fixture
def time_series_non_core_objectives():
    non_core_time_series = [obj_() for obj_ in get_non_core_objectives()
                            if ProblemTypes.TIME_SERIES_REGRESSION in obj_.problem_types]
    return non_core_time_series


@pytest.fixture
def time_series_objectives(time_series_core_objectives, time_series_non_core_objectives):
    return time_series_core_objectives + time_series_non_core_objectives


@pytest.fixture
def stackable_classifiers(helper_functions):
    stackable_classifiers = []
    for estimator_class in _all_estimators():
        supported_problem_types = [handle_problem_types(pt) for pt in estimator_class.supported_problem_types]
        if (set(supported_problem_types) == {ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                                             ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS} and
            estimator_class.model_family not in _nonstackable_model_families and
                estimator_class.model_family != ModelFamily.ENSEMBLE):
            stackable_classifiers.append(helper_functions.safe_init_component_with_njobs_1(estimator_class))
    return stackable_classifiers


@pytest.fixture
def stackable_regressors(helper_functions):
    stackable_regressors = []
    for estimator_class in _all_estimators():
        supported_problem_types = [handle_problem_types(pt) for pt in estimator_class.supported_problem_types]
        if (set(supported_problem_types) == {ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION} and
            estimator_class.model_family not in _nonstackable_model_families and
                estimator_class.model_family != ModelFamily.ENSEMBLE):
            stackable_regressors.append(helper_functions.safe_init_component_with_njobs_1(estimator_class))
    return stackable_regressors


@pytest.fixture
def tree_estimators():
    est_classifier_class = DecisionTreeClassifier()
    est_regressor_class = DecisionTreeRegressor()
    return est_classifier_class, est_regressor_class


@pytest.fixture
def fitted_tree_estimators(tree_estimators, X_y_binary, X_y_regression):
    est_clf, est_reg = tree_estimators
    X_b, y_b = X_y_binary
    X_r, y_r = X_y_regression
    est_clf.fit(X_b, y_b)
    est_reg.fit(X_r, y_r)
    return est_clf, est_reg


@pytest.fixture
def logit_estimator():
    est_class = LogisticRegressionClassifier()
    return est_class


@pytest.fixture
def helper_functions():
    class Helpers:
        @staticmethod
        def safe_init_component_with_njobs_1(component_class):
            try:
                component = component_class(n_jobs=1)
            except TypeError:
                component = component_class()
            return component

        @staticmethod
        def safe_init_pipeline_with_njobs_1(pipeline_class):
            try:
                estimator = pipeline_class.component_graph[-1]
                estimator_name = estimator if isinstance(estimator, str) else estimator.name
                pl = pipeline_class({estimator_name: {'n_jobs': 1}})
            except ValueError:
                pl = pipeline_class({})
            return pl

    return Helpers


@pytest.fixture
def make_data_type():
    """Helper function to convert numpy or pandas input to the appropriate type for tests."""
    def _make_data_type(data_type, data):
        if data_type != "np":
            if len(data.shape) == 1:
                data = pd.Series(data)
            else:
                data = pd.DataFrame(data)
        if data_type == "ww":
            if len(data.shape) == 1:
                data = ww.DataColumn(data)
            else:
                data = ww.DataTable(data)
        return data

    return _make_data_type
