import contextlib
import os
import sys
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import py
import pytest
import woodwork as ww
from sklearn import datasets
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.objectives import BinaryClassificationObjective
from evalml.objectives.utils import (
    get_core_objectives,
    get_non_core_objectives,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline,
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline,
)
from evalml.pipelines.components import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    Estimator,
    LogisticRegressionClassifier,
    SklearnStackedEnsembleClassifier,
    SklearnStackedEnsembleRegressor,
)
from evalml.pipelines.components.ensemble.sklearn_stacked_ensemble_base import (
    _nonstackable_model_families,
)
from evalml.pipelines.components.utils import _all_estimators
from evalml.preprocessing import load_data
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_regression,
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_offline: mark test to be skipped if offline (https://api.featurelabs.com cannot be reached)",
    )


def create_mock_pipeline(estimator, problem_type):
    if problem_type == ProblemTypes.BINARY:

        class MockBinaryPipelineWithOnlyEstimator(BinaryClassificationPipeline):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]

            def __init__(self, parameters, random_seed=0):
                super().__init__(
                    self.component_graph,
                    parameters=parameters,
                    custom_name=self.custom_name,
                    random_seed=random_seed,
                )

        return MockBinaryPipelineWithOnlyEstimator
    elif problem_type == ProblemTypes.MULTICLASS:

        class MockMulticlassPipelineWithOnlyEstimator(MulticlassClassificationPipeline):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]

            def __init__(self, parameters, random_seed=0):
                super().__init__(
                    self.component_graph,
                    parameters=parameters,
                    custom_name=self.custom_name,
                    random_seed=random_seed,
                )

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

        class MockTSRegressionPipelineWithOnlyEstimator(
            TimeSeriesBinaryClassificationPipeline
        ):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]

        return MockTSRegressionPipelineWithOnlyEstimator
    elif problem_type == ProblemTypes.TIME_SERIES_MULTICLASS:

        class MockTSRegressionPipelineWithOnlyEstimator(
            TimeSeriesMulticlassClassificationPipeline
        ):
            custom_name = f"Pipeline with {estimator.name}"
            component_graph = [estimator]

        return MockTSRegressionPipelineWithOnlyEstimator


@pytest.fixture
def all_pipeline_classes():
    all_possible_pipeline_classes = []
    for estimator in [
        estimator
        for estimator in _all_estimators()
        if estimator != SklearnStackedEnsembleClassifier
        and estimator != SklearnStackedEnsembleRegressor
    ]:
        for problem_type in estimator.supported_problem_types:
            all_possible_pipeline_classes.append(
                create_mock_pipeline(estimator, problem_type)
            )
    return all_possible_pipeline_classes


@pytest.fixture
def all_binary_pipeline_classes(all_pipeline_classes):
    return [
        pipeline_class
        for pipeline_class in all_pipeline_classes
        if issubclass(pipeline_class, BinaryClassificationPipeline)
    ]


@pytest.fixture
def all_multiclass_pipeline_classes(all_pipeline_classes):
    return [
        pipeline_class
        for pipeline_class in all_pipeline_classes
        if issubclass(pipeline_class, MulticlassClassificationPipeline)
    ]


def pytest_addoption(parser):
    parser.addoption(
        "--has-minimal-dependencies",
        action="store_true",
        default=False,
        help="If true, tests will assume only the dependencies in"
        "core-requirements.txt have been installed.",
    )
    parser.addoption(
        "--is-using-conda",
        action="store_true",
        default=False,
        help="If true, tests will assume that they are being run as part of"
        "the build_conda_pkg workflow with the feedstock.",
    )


@pytest.fixture
def has_minimal_dependencies(pytestconfig):
    return pytestconfig.getoption("--has-minimal-dependencies")


@pytest.fixture
def is_using_conda(pytestconfig):
    return pytestconfig.getoption("--is-using-conda")


@pytest.fixture
def is_using_windows(pytestconfig):
    return sys.platform in ["win32", "cygwin"]


@pytest.fixture
def is_running_py_39_or_above():
    return sys.version_info >= (3, 9)


@pytest.fixture
def assert_allowed_pipelines_equal_helper():
    def assert_allowed_pipelines_equal_helper(
        actual_allowed_pipelines, expected_allowed_pipelines
    ):
        for actual, expected in zip(
            actual_allowed_pipelines, expected_allowed_pipelines
        ):
            for pipeline_subclass in [
                BinaryClassificationPipeline,
                MulticlassClassificationPipeline,
                RegressionPipeline,
            ]:
                if isinstance(expected, pipeline_subclass):
                    assert isinstance(expected, pipeline_subclass)
                    break
            assert actual.parameters == expected.parameters
            assert actual.name == expected.name
            assert actual.problem_type == expected.problem_type
            assert actual.component_graph == expected.component_graph

    return assert_allowed_pipelines_equal_helper


@pytest.fixture
def X_y_binary():
    X, y = datasets.make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=2, random_state=0
    )

    return X, y


@pytest.fixture(scope="session")
def X_y_binary_cls():
    X, y = datasets.make_classification(
        n_samples=100, n_features=20, n_informative=2, n_redundant=2, random_state=0
    )
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def X_y_regression():
    X, y = datasets.make_regression(
        n_samples=100, n_features=20, n_informative=3, random_state=0
    )
    return X, y


@pytest.fixture
def X_y_multi():
    X, y = datasets.make_classification(
        n_samples=100,
        n_features=20,
        n_classes=3,
        n_informative=3,
        n_redundant=2,
        random_state=0,
    )
    return X, y


@pytest.fixture
def X_y_categorical_regression():
    data_path = os.path.join(os.path.dirname(__file__), "data/tips.csv")
    flights = pd.read_csv(data_path)

    y = flights["tip"]
    X = flights.drop("tip", axis=1)

    # add categorical dtype
    X["smoker"] = X["smoker"].astype("category")
    return X, y


@pytest.fixture
def X_y_categorical_classification():
    data_path = os.path.join(os.path.dirname(__file__), "data/titanic.csv")
    titanic = pd.read_csv(data_path)

    y = titanic["Survived"]
    X = titanic.drop(["Survived", "Name"], axis=1)
    return X, y


@pytest.fixture()
def text_df():
    df = pd.DataFrame(
        {
            "col_1": [
                "I'm singing in the rain! Just singing in the rain, what a glorious feeling, I'm happy again!",
                "In sleep he sang to me, in dreams he came... That voice which calls to me, and speaks my name.",
                "I'm gonna be the main event, like no king was before! I'm brushing up on looking down, I'm working on my ROAR!",
            ],
            "col_2": [
                "do you hear the people sing? Singing the songs of angry men\n\tIt is the music of a people who will NOT be slaves again!",
                "I dreamed a dream in days gone by, when hope was high and life worth living",
                "Red, the blood of angry men - black, the dark of ages past",
            ],
        }
    )
    df.ww.init(logical_types={"col_1": "NaturalLanguage", "col_2": "NaturalLanguage"})
    yield df


@pytest.fixture
def ts_data():
    X, y = pd.DataFrame({"features": range(101, 132)}), pd.Series(range(1, 32))
    y.index = pd.date_range("2020-10-01", "2020-10-31")
    X.index = pd.date_range("2020-10-01", "2020-10-31")
    return X, y


@pytest.fixture
def ts_data_seasonal_train():
    sine_ = np.linspace(-np.pi * 5, np.pi * 5, 25)
    X, y = pd.DataFrame({"features": range(25)}), pd.Series(sine_)
    y.index = pd.date_range(start="1/1/2018", periods=25)
    X.index = pd.date_range(start="1/1/2018", periods=25)
    return X, y


@pytest.fixture
def ts_data_seasonal_test():
    sine_ = np.linspace(-np.pi * 5, np.pi * 5, 25)
    X, y = pd.DataFrame({"features": range(25)}), pd.Series(sine_)
    y.index = pd.date_range(start="1/26/2018", periods=25)
    X.index = pd.date_range(start="1/26/2018", periods=25)
    return X, y


@pytest.fixture
def dummy_pipeline_hyperparameters():
    return {
        "Mock Classifier": {
            "param a": Integer(0, 10),
            "param b": Real(0, 10),
            "param c": ["option a", "option b", "option c"],
            "param d": ["option a", "option b", 100, np.inf],
        }
    }


@pytest.fixture
def dummy_pipeline_hyperparameters_unicode():
    return {
        "Mock Classifier": {
            "param a": Integer(0, 10),
            "param b": Real(0, 10),
            "param c": ["option a 💩", "option b 💩", "option c 💩"],
            "param d": ["option a", "option b", 100, np.inf],
        }
    }


@pytest.fixture
def dummy_pipeline_hyperparameters_small():
    return {
        "Mock Classifier": {
            "param a": ["most_frequent", "median", "mean"],
            "param b": ["a", "b", "c"],
        }
    }


@pytest.fixture
def dummy_classifier_estimator_class():
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.NONE
        supported_problem_types = [
            ProblemTypes.BINARY,
            ProblemTypes.MULTICLASS,
            ProblemTypes.TIME_SERIES_MULTICLASS,
            ProblemTypes.TIME_SERIES_BINARY,
        ]
        hyperparameter_ranges = {"a": Integer(0, 10), "b": Real(0, 10)}

        def __init__(self, a=1, b=0, random_seed=0):
            super().__init__(
                parameters={"a": a, "b": b}, component_obj=None, random_seed=random_seed
            )

        def fit(self, X, y):
            return self

    return MockEstimator


@pytest.fixture
def example_graph():
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot_RandomForest": ["One Hot Encoder", "Imputer.x", "y"],
        "OneHot_ElasticNet": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Classifier", "OneHot_RandomForest.x", "y"],
        "Elastic Net": ["Elastic Net Classifier", "OneHot_ElasticNet.x", "y"],
        "Logistic Regression": [
            "Logistic Regression Classifier",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    return component_graph


@pytest.fixture
def example_regression_graph():
    component_graph = {
        "Imputer": ["Imputer", "X", "y"],
        "OneHot": ["One Hot Encoder", "Imputer.x", "y"],
        "Random Forest": ["Random Forest Regressor", "OneHot.x", "y"],
        "Elastic Net": ["Elastic Net Regressor", "OneHot.x", "y"],
        "Linear Regressor": [
            "Linear Regressor",
            "Random Forest.x",
            "Elastic Net.x",
            "y",
        ],
    }
    return component_graph


@pytest.fixture
def dummy_binary_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockBinaryClassificationPipeline(BinaryClassificationPipeline):
        custom_name = "Mock Binary Classification Pipeline"
        estimator = MockEstimator
        component_graph = [MockEstimator]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return MockBinaryClassificationPipeline


@pytest.fixture
def dummy_multiclass_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockMulticlassClassificationPipeline(MulticlassClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator]
        custom_name = "Mock Multiclass Classification Pipeline"

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return MockMulticlassClassificationPipeline


@pytest.fixture
def dummy_regressor_estimator_class():
    class MockRegressor(Estimator):
        name = "Mock Regressor"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.REGRESSION]
        hyperparameter_ranges = {"a": Integer(0, 10), "b": Real(0, 10)}

        def __init__(self, a=1, b=0, random_seed=0):
            super().__init__(
                parameters={"a": a, "b": b}, component_obj=None, random_seed=random_seed
            )

        def fit(self, X, y):
            return self

    return MockRegressor


@pytest.fixture
def dummy_regression_pipeline_class(dummy_regressor_estimator_class):
    MockRegressor = dummy_regressor_estimator_class

    class MockRegressionPipeline(RegressionPipeline):
        component_graph = [MockRegressor]
        custom_name = "Mock Regression Pipeline"

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return MockRegressionPipeline


@pytest.fixture
def dummy_time_series_regressor_estimator_class():
    class MockTimeSeriesRegressor(Estimator):
        name = "Mock Time Series Regressor"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
        hyperparameter_ranges = {"a": Integer(0, 10), "b": Real(0, 10)}

        def __init__(self, a=1, b=0, random_seed=0):
            super().__init__(
                parameters={"a": a, "b": b}, component_obj=None, random_seed=random_seed
            )

    return MockTimeSeriesRegressor


@pytest.fixture
def dummy_time_series_regression_pipeline_class(
    dummy_time_series_regressor_estimator_class,
):
    MockTimeSeriesRegressor = dummy_time_series_regressor_estimator_class

    class MockTimeSeriesRegressionPipeline(TimeSeriesRegressionPipeline):
        component_graph = [MockTimeSeriesRegressor]
        custom_name = None

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

    return MockTimeSeriesRegressionPipeline


@pytest.fixture
def dummy_ts_binary_pipeline_class(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockBinaryClassificationPipeline(TimeSeriesBinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph, parameters=parameters, random_seed=random_seed
            )

    return MockBinaryClassificationPipeline


@pytest.fixture
def logistic_regression_multiclass_pipeline_class():
    class LogisticRegressionMulticlassPipeline(MulticlassClassificationPipeline):
        """Logistic Regression Pipeline for binary classification."""

        custom_name = "Logistic Regression Multiclass Pipeline"
        component_graph = [
            "Imputer",
            "One Hot Encoder",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return LogisticRegressionMulticlassPipeline


@pytest.fixture
def logistic_regression_binary_pipeline_class():
    class LogisticRegressionBinaryPipeline(BinaryClassificationPipeline):
        component_graph = [
            "Imputer",
            "One Hot Encoder",
            "Standard Scaler",
            "Logistic Regression Classifier",
        ]
        custom_name = "Logistic Regression Binary Pipeline"

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return LogisticRegressionBinaryPipeline


@pytest.fixture
def linear_regression_pipeline_class():
    class LinearRegressionPipeline(RegressionPipeline):
        """Linear Regression Pipeline for regression problems."""

        component_graph = [
            "One Hot Encoder",
            "Imputer",
            "Standard Scaler",
            "Linear Regressor",
        ]
        custom_name = "Linear Regression Pipeline"

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
                random_seed=random_seed,
            )

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return LinearRegressionPipeline


@pytest.fixture
def dummy_stacked_ensemble_binary_estimator(logistic_regression_binary_pipeline_class):
    p1 = logistic_regression_binary_pipeline_class({})
    ensemble_estimator = SklearnStackedEnsembleClassifier(
        input_pipelines=[p1], random_seed=0
    )
    return ensemble_estimator


@pytest.fixture
def dummy_stacked_ensemble_multiclass_estimator(
    logistic_regression_multiclass_pipeline_class,
):
    p1 = logistic_regression_multiclass_pipeline_class({})
    ensemble_estimator = SklearnStackedEnsembleClassifier(
        input_pipelines=[p1], random_seed=0
    )
    return ensemble_estimator


@pytest.fixture
def dummy_stacked_ensemble_regressor_estimator(linear_regression_pipeline_class):
    p1 = linear_regression_pipeline_class({})
    ensemble_estimator = SklearnStackedEnsembleRegressor(
        input_pipelines=[p1], random_seed=0
    )
    return ensemble_estimator


@pytest.fixture
def time_series_regression_pipeline_class():
    class TSRegressionPipeline(TimeSeriesRegressionPipeline):
        """Random Forest Regression Pipeline for time series regression problems."""

        component_graph = ["Delayed Feature Transformer", "Random Forest Regressor"]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph, parameters=parameters, random_seed=random_seed
            )

    return TSRegressionPipeline


@pytest.fixture
def time_series_binary_classification_pipeline_class():
    class TSBinaryPipeline(TimeSeriesBinaryClassificationPipeline):
        """Logistic Regression Pipeline for time series binary classification problems."""

        component_graph = [
            "Delayed Feature Transformer",
            "Logistic Regression Classifier",
        ]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph, parameters=parameters, random_seed=random_seed
            )

    return TSBinaryPipeline


@pytest.fixture
def time_series_multiclass_classification_pipeline_class():
    class TSMultiPipeline(TimeSeriesMulticlassClassificationPipeline):
        """Logistic Regression Pipeline for time series multiclass classification problems."""

        component_graph = [
            "Delayed Feature Transformer",
            "Logistic Regression Classifier",
        ]

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph, parameters=parameters, random_seed=random_seed
            )

    return TSMultiPipeline


@pytest.fixture
def decision_tree_classification_pipeline_class(X_y_categorical_classification):
    pipeline = BinaryClassificationPipeline(
        component_graph={
            "Imputer": ["Imputer", "X", "y"],
            "OneHot": ["One Hot Encoder", "Imputer.x", "y"],
            "Standard Scaler": ["Standard Scaler", "OneHot.x", "y"],
            "Decision Tree Classifier": [
                "Elastic Net Classifier",
                "Standard Scaler.x",
                "y",
            ],
        }
    )
    X, y = X_y_categorical_classification
    X.ww.init(logical_types={"Ticket": "categorical", "Cabin": "categorical"})
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def nonlinear_binary_pipeline_class(example_graph):
    class NonLinearBinaryPipeline(BinaryClassificationPipeline):
        custom_name = "Non Linear Binary Pipeline"
        component_graph = example_graph

        def __init__(self, parameters, random_seed=0):
            super().__init__(
                self.component_graph,
                parameters=parameters,
                custom_name=self.custom_name,
            )

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return NonLinearBinaryPipeline


@pytest.fixture
def nonlinear_multiclass_pipeline_class(example_graph):
    class NonLinearMulticlassPipeline(MulticlassClassificationPipeline):
        component_graph = example_graph

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return NonLinearMulticlassPipeline


@pytest.fixture
def nonlinear_regression_pipeline_class(example_regression_graph):
    class NonLinearRegressionPipeline(RegressionPipeline):
        component_graph = example_regression_graph

        def __init__(self, parameters, random_seed=0):
            super().__init__(self.component_graph, parameters=parameters)

        def new(self, parameters, random_seed=0):
            return self.__class__(parameters, random_seed=random_seed)

        def clone(self):
            return self.__class__(self.parameters, random_seed=self.random_seed)

    return NonLinearRegressionPipeline


@pytest.fixture
def binary_test_objectives():
    return [
        o
        for o in get_core_objectives(ProblemTypes.BINARY)
        if o.name in {"Log Loss Binary", "F1", "AUC"}
    ]


@pytest.fixture
def multiclass_test_objectives():
    return [
        o
        for o in get_core_objectives(ProblemTypes.MULTICLASS)
        if o.name in {"Log Loss Multiclass", "AUC Micro", "F1 Micro"}
    ]


@pytest.fixture
def regression_test_objectives():
    return [
        o
        for o in get_core_objectives(ProblemTypes.REGRESSION)
        if o.name in {"R2", "Root Mean Squared Error", "MAE"}
    ]


@pytest.fixture
def time_series_core_objectives():
    return get_core_objectives(ProblemTypes.TIME_SERIES_REGRESSION)


@pytest.fixture
def time_series_non_core_objectives():
    non_core_time_series = [
        obj_()
        for obj_ in get_non_core_objectives()
        if ProblemTypes.TIME_SERIES_REGRESSION in obj_.problem_types
    ]
    return non_core_time_series


@pytest.fixture
def time_series_objectives(
    time_series_core_objectives, time_series_non_core_objectives
):
    return time_series_core_objectives + time_series_non_core_objectives


@pytest.fixture
def stackable_classifiers(helper_functions):
    stackable_classifiers = []
    for estimator_class in _all_estimators():
        supported_problem_types = [
            handle_problem_types(pt) for pt in estimator_class.supported_problem_types
        ]
        if (
            set(supported_problem_types)
            == {
                ProblemTypes.BINARY,
                ProblemTypes.MULTICLASS,
                ProblemTypes.TIME_SERIES_BINARY,
                ProblemTypes.TIME_SERIES_MULTICLASS,
            }
            and estimator_class.model_family not in _nonstackable_model_families
            and estimator_class.model_family != ModelFamily.ENSEMBLE
        ):
            stackable_classifiers.append(estimator_class)
    return stackable_classifiers


@pytest.fixture
def stackable_regressors(helper_functions):
    stackable_regressors = []
    for estimator_class in _all_estimators():
        supported_problem_types = [
            handle_problem_types(pt) for pt in estimator_class.supported_problem_types
        ]
        if (
            set(supported_problem_types)
            == {ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION}
            and estimator_class.model_family not in _nonstackable_model_families
            and estimator_class.model_family != ModelFamily.ENSEMBLE
        ):
            stackable_regressors.append(estimator_class)
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
    X_b = pd.DataFrame(X_b, columns=[f"Testing_{col}" for col in range(len(X_b[0]))])
    X_r = pd.DataFrame(X_r, columns=[f"Testing_{col}" for col in range(len(X_r[0]))])
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
                estimator_name = (
                    estimator if isinstance(estimator, str) else estimator.name
                )
                pl = pipeline_class({estimator_name: {"n_jobs": 1}})
            except ValueError:
                pl = pipeline_class({})
            return pl

    return Helpers


@pytest.fixture
def make_data_type():
    """Helper function to convert numpy or pandas input to the appropriate type for tests."""

    def _make_data_type(data_type, data):
        if data_type == "li":
            if isinstance(data, pd.DataFrame):
                data = data.to_numpy()
            data = data.tolist()
            return data
        if data_type != "np":
            if len(data.shape) == 1:
                data = pd.Series(data)
            else:
                data = pd.DataFrame(data)
        if data_type == "ww":
            if len(data.shape) == 1:
                data = ww.init_series(data)
            else:
                data.ww.init()
        return data

    return _make_data_type


def load_fraud_local(n_rows=None):
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    fraud_data_path = os.path.join(data_folder_path, "fraud_transactions.csv.gz")
    X, y = load_data(
        path=fraud_data_path,
        index="id",
        target="fraud",
        compression="gzip",
        n_rows=n_rows,
    )
    return X, y


@pytest.fixture
def fraud_local():
    X, y = load_fraud_local()
    X.ww.set_types(logical_types={"provider": "Categorical", "region": "Categorical"})
    return X, y


@pytest.fixture
def fraud_100():
    X, y = load_fraud_local(n_rows=100)
    X.ww.set_types(
        logical_types={
            "provider": "Categorical",
            "region": "Categorical",
            "currency": "categorical",
            "expiration_date": "categorical",
        }
    )
    return X, y


@pytest.fixture
def breast_cancer_local():
    data = datasets.load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    y = y.map(lambda x: data["target_names"][x])
    X.ww.init()
    y = ww.init_series(y)
    return X, y


@pytest.fixture
def wine_local():
    data = datasets.load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    y = y.map(lambda x: data["target_names"][x])
    X.ww.init()
    y = ww.init_series(y)
    return X, y


@pytest.fixture
def diabetes_local():
    data = datasets.load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    X.ww.init()
    y = ww.init_series(y)
    return X, y


@pytest.fixture
def churn_local():
    currdir_path = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(currdir_path, "data")
    churn_data_path = os.path.join(data_folder_path, "churn.csv")
    return load_data(
        path=churn_data_path,
        index="customerID",
        target="Churn",
    )


@pytest.fixture
def mock_imbalanced_data_X_y():
    """Helper function to return an imbalanced binary or multiclass dataset"""

    def _imbalanced_data_X_y(problem_type, categorical_columns, size):
        """ "Generates a dummy classification dataset with particular amounts of class imbalance and categorical input columns.
        For our targets, we maintain a 1:5, or 0.2, class ratio of minority : majority.
        We only generate minimum amount for X to set the logical_types, so the length of X and y will be different.

        Args:
            problem_type (str): Either 'binary' or 'multiclass'
            categorical_columns (str): Determines how many categorical cols to use. Either 'all', 'some', or 'none'.
            size (str): Either 'large' or 'small'. 'large' returns a dataset of size 21,000, while 'small' returns a size of 4200
        """
        multiplier = 5 if size == "large" else 1
        col_names = [f"col_{i}" for i in range(100)]
        # generate X to be all int values
        X_dict = {
            col_name: [i % (j + 1) for i in range(1, 100)]
            for j, col_name in enumerate(col_names)
        }
        X = pd.DataFrame(X_dict)
        if categorical_columns == "all":
            X.ww.init(logical_types={col_name: "Categorical" for col_name in col_names})
        elif categorical_columns == "some":
            X.ww.init(
                logical_types={
                    col_name: "Categorical"
                    for col_name in col_names[: len(col_names) // 2]
                }
            )
        else:
            X.ww.init()
        if problem_type == "binary":
            targets = [0] * 3500 + [1] * 700
        else:
            targets = [0] * 3000 + [1] * 600 + [2] * 600
        targets *= multiplier
        y = ww.init_series(pd.Series(targets))
        return X, y

    return _imbalanced_data_X_y


class _AutoMLTestEnv:
    """A test environment that makes it easy to test automl behavior with patched pipeline computations.

    This class provides a context manager that will automatically patch pipeline fit/score/predict_proba methods,
    as well as _encode_targets, BinaryClassificationObjective.optimize_threshold, and skopt.Optimizer.tell. These are
    the most time consuming operations during search, so your test will run as fast as possible.

    This class is ideal for tests that verify some behavior of AutoMLSearch that can be controlled via the side_effect
    or return_value parameters exposed to the patched methods but it may not be suitable for all tests, such as
    tests that patch Estimator.fit instead of Pipeline.fit or tests that only want to patch a selective
    subset of the methods listed above.

    Example:
        >>> env = _AutoMLTestEnv(problem_type="binary")
        >>> # run_search is short-hand for creating the context manager and then running search
        >>> # env.run_search(automl, score_return_value={automl.objective.name: 1.0})
        >>> # with env.test_context(score_return_value={automl.objective.name: 1.0}):
        >>> #    automl.search()
        >>> # env.mock_fit.assert_called_once()
        >>> # env.mock_score.assert_called_once()
    """

    def __init__(self, problem_type):
        """Create a test environment.

        Args:
            problem_type (str): The problem type corresponding to the search class you want to test.

        Attributes:
            mock_fit (MagicMock): MagicMock corresponding to the pipeline.fit method for the latest automl computation.
                Set to None until the first computation is run in the test environment.
            mock_tell (MagicMock): Magic mock corresponding to the skopt.Optimizer.tell method. Set to None unil the
                first computation is run in the test environment.
            mock_score (MagicMock): MagicMock corresponding to the pipeline.score method for the latest automl computation.
                Set to None until the first computation is run in the test environment.
            mock_encode_targets (MagicMock): MagicMock corresponding to the pipeline._encode_targets method for the latest automl computation.
                Set to None until the first computation is run in the test environment.
            mock_predict_proba (MagicMock): MagicMock corresponding to the pipeline.predict_proba method for the latest automl computation.
                Set to None until the first computation is run in the test environment.
            mock_optimize_threshold (MagicMock): MagicMock corresponding to the BinaryClassificationObjective.optimize_threshold for the latest automl computation.
                Set to None until the first computation is run in the test environment.
        """
        self.problem_type = handle_problem_types(problem_type)
        self._mock_fit = None
        self._mock_tell = None
        self._mock_score = None
        self._mock_get_names = None
        self._mock_encode_targets = None
        self._mock_predict_proba = None
        self._mock_optimize_threshold = None

    @property
    def _pipeline_class(self):
        return {
            ProblemTypes.REGRESSION: "evalml.pipelines.RegressionPipeline",
            ProblemTypes.BINARY: "evalml.pipelines.BinaryClassificationPipeline",
            ProblemTypes.MULTICLASS: "evalml.pipelines.MulticlassClassificationPipeline",
            ProblemTypes.TIME_SERIES_REGRESSION: "evalml.pipelines.TimeSeriesRegressionPipeline",
            ProblemTypes.TIME_SERIES_MULTICLASS: "evalml.pipelines.TimeSeriesMulticlassClassificationPipeline",
            ProblemTypes.TIME_SERIES_BINARY: "evalml.pipelines.TimeSeriesBinaryClassificationPipeline",
        }[self.problem_type]

    def _patch_method(self, method, side_effect, return_value, pipeline_class_str=None):
        kwargs = {}
        if pipeline_class_str is None:
            pipeline_class_str = self._pipeline_class
        if side_effect is not None:
            kwargs = {"side_effect": side_effect}
        elif return_value is not None:
            kwargs = {"return_value": return_value}
        return patch(pipeline_class_str + "." + method, **kwargs)

    def _reset_mocks(self):
        """Set the mocks to None before running a computation so that we can prevent users from trying to access
        them before leaving the context manager."""
        self._mock_fit = None
        self._mock_tell = None
        self._mock_score = None
        self._mock_get_names = None
        self._mock_encode_targets = None
        self._mock_predict_proba = None
        self._mock_optimize_threshold = None

    def _get_mock(self, mock_name):
        mock = getattr(self, f"_mock_{mock_name}")
        if mock is None:
            raise ValueError(
                f"mock_{mock_name} cannot be accessed before leaving the test_context! "
                "Access it after leaving test_context."
            )
        return mock

    @property
    def mock_fit(self):
        return self._get_mock("fit")

    @property
    def mock_tell(self):
        return self._get_mock("tell")

    @property
    def mock_score(self):
        return self._get_mock("score")

    @property
    def mock_encode_targets(self):
        return self._get_mock("encode_targets")

    @property
    def mock_predict_proba(self):
        return self._get_mock("predict_proba")

    @property
    def mock_optimize_threshold(self):
        return self._get_mock("optimize_threshold")

    @contextlib.contextmanager
    def test_context(
        self,
        score_return_value=None,
        mock_score_side_effect=None,
        mock_fit_side_effect=None,
        mock_fit_return_value=None,
        predict_proba_return_value=None,
        optimize_threshold_return_value=0.2,
    ):
        """A context manager for creating an environment that patches time-consuming pipeline methods.
        Sets the mock_fit, mock_score, mock_encode_targets, mock_predict_proba, mock_optimize_threshold attributes.

        Args:
            score_return_value: Passed as the return_value argument of the pipeline.score patch.
            mock_score_side_effect: Passed as the side_effect argument of the pipeline.score patch. Takes precedence over
                score_return_value.
            mock_fit_side_effect: Passed as the side_effect argument of the pipeline.fit patch. Takes precedence over mock_fit_return_value.
            mock_fit_return_value: Passed as the return_value argument of the pipeline.fit patch.
            predict_proba_return_value: Passed as the return_value argument of the pipeline.predict_proba patch.
            optimize_threshold_return_value: Passed as the return value of BinaryClassificationObjective.optimize_threshold patch.
        """
        mock_fit = self._patch_method(
            "fit", side_effect=mock_fit_side_effect, return_value=mock_fit_return_value
        )
        mock_score = self._patch_method(
            "score", side_effect=mock_score_side_effect, return_value=score_return_value
        )
        mock_get_names = patch(
            "evalml.pipelines.components.FeatureSelector.get_names", return_value=[]
        )

        # For simplicity, we will always mock predict_proba and _encode_targets even if the problem is not a
        # classification problem. For regression problems, we'll mock BinaryClassificationPipeline but it doesn't
        # matter which one we mock since those methods won't be called for regression.
        pipeline_to_mock = self._pipeline_class
        if is_regression(self.problem_type):
            pipeline_to_mock = "evalml.pipelines.BinaryClassificationPipeline"

        mock_encode_targets = self._patch_method(
            "_encode_targets",
            side_effect=lambda y: y,
            return_value=None,
            pipeline_class_str=pipeline_to_mock,
        )
        mock_predict_proba = self._patch_method(
            "predict_proba",
            side_effect=None,
            return_value=predict_proba_return_value,
            pipeline_class_str=pipeline_to_mock,
        )

        mock_optimize = patch(
            "evalml.objectives.BinaryClassificationObjective.optimize_threshold",
            return_value=optimize_threshold_return_value,
        )

        mock_tell = patch("evalml.tuners.skopt_tuner.Optimizer.tell")

        # Reset the mocks from a previous computation so that ValueError can be properly raised if
        # user tries to access mocks before leaving the context
        self._reset_mocks()

        # Unfortunately, in order to set the MagicMock instances as class attributes we need to use the
        # `with ... ` syntax.
        sleep_time = PropertyMock(return_value=0.00000001)
        mock_sleep = patch(
            "evalml.automl.AutoMLSearch._sleep_time", new_callable=sleep_time
        )

        with mock_sleep, mock_fit as fit, mock_score as score, mock_get_names as get_names, mock_encode_targets as encode, mock_predict_proba as proba, mock_tell as tell, mock_optimize as optimize:
            # Can think of `yield` as blocking this method until the computation finishes running
            yield
            self._mock_fit = fit
            self._mock_tell = tell
            self._mock_score = score
            self._mock_get_names = get_names
            self._mock_encode_targets = encode
            self._mock_predict_proba = proba
            self._mock_optimize_threshold = optimize


@pytest.fixture
def AutoMLTestEnv():
    return _AutoMLTestEnv


@pytest.fixture
def tmpdir(tmp_path):
    dir = py.path.local(tmp_path)
    yield dir
    dir.remove(ignore_errors=True)


@pytest.fixture
def df_with_url_and_email():
    X = pd.DataFrame(
        {
            "categorical": ["a", "b", "b", "a", "c"],
            "numeric": [1, 2, 3, 4, 5],
            "email": [
                "abalone_0@gmail.com",
                "AbaloneRings@yahoo.com",
                "abalone_2@abalone.com",
                "$titanic_data%&@hotmail.com",
                "foo*EMAIL@email.org",
            ],
            "integer": [1, 2, 3, 4, 5],
            "boolean": [True, False, True, False, False],
            "nat_lang": ["natural", "language", "understanding", "is", "difficult"],
            "url": [
                "https://evalml.alteryx.com/en/stable/",
                "https://woodwork.alteryx.com/en/stable/guides/statistical_insights.html",
                "https://twitter.com/AlteryxOSS",
                "https://www.twitter.com/AlteryxOSS",
                "https://www.evalml.alteryx.com/en/stable/demos/text_input.html",
            ],
        }
    )
    X.ww.init(
        logical_types={
            "categorical": "Categorical",
            "numeric": "Double",
            "email": "EmailAddress",
            "boolean": "Boolean",
            "nat_lang": "NaturalLanguage",
            "integer": "Integer",
            "url": "URL",
        }
    )
    return X


def CustomClassificationObjectiveRanges(ranges):
    class CustomClassificationObjectiveRanges(BinaryClassificationObjective):
        """Accuracy score for binary and multiclass classification."""

        name = "Classification Accuracy"
        greater_is_better = True
        score_needs_proba = False
        perfect_score = 1.0
        is_bounded_like_percentage = False
        expected_range = ranges
        problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

        def objective_function(self, y_true, y_predicted, X=None):
            """Not implementing since mocked in our tests."""

    return CustomClassificationObjectiveRanges()
