import os

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.objectives.utils import get_core_objectives
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline
)
from evalml.pipelines.components import (
    Estimator,
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
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
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
def binary_core_objectives():
    return get_core_objectives(ProblemTypes.BINARY)


@pytest.fixture
def multiclass_core_objectives():
    return get_core_objectives(ProblemTypes.MULTICLASS)


@pytest.fixture
def regression_core_objectives():
    return get_core_objectives(ProblemTypes.REGRESSION)


@pytest.fixture
def stackable_classifiers():
    stackable_classifiers = []
    for estimator_class in _all_estimators():
        supported_problem_types = [handle_problem_types(pt) for pt in estimator_class.supported_problem_types]
        if (set(supported_problem_types) == {ProblemTypes.BINARY, ProblemTypes.MULTICLASS} and
            estimator_class.model_family not in _nonstackable_model_families and
                estimator_class.model_family != ModelFamily.ENSEMBLE):
            stackable_classifiers.append(estimator_class())
    return stackable_classifiers


@pytest.fixture
def stackable_regressors():
    stackable_regressors = []
    for estimator_class in _all_estimators():
        supported_problem_types = [handle_problem_types(pt) for pt in estimator_class.supported_problem_types]
        if (set(supported_problem_types) == {ProblemTypes.REGRESSION} and
            estimator_class.model_family not in _nonstackable_model_families and
                estimator_class.model_family != ModelFamily.ENSEMBLE):
            stackable_regressors.append(estimator_class())
    return stackable_regressors
