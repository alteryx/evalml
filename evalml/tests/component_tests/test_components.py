import importlib
import inspect
import os
import warnings
from unittest.mock import patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from skopt.space import Categorical

from evalml.exceptions import (
    ComponentNotYetFittedError,
    EnsembleMissingPipelinesError,
    MethodPropertyNotFoundError,
)
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline, RegressionPipeline
from evalml.pipelines.components import (
    LSA,
    PCA,
    BaselineClassifier,
    BaselineRegressor,
    CatBoostClassifier,
    CatBoostRegressor,
    ComponentBase,
    DateTimeFeaturizer,
    DelayedFeatureTransformer,
    DFSTransformer,
    DropColumns,
    DropNullColumns,
    DropRowsTransformer,
    ElasticNetClassifier,
    ElasticNetRegressor,
    Estimator,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    Imputer,
    LightGBMClassifier,
    LightGBMRegressor,
    LinearDiscriminantAnalysis,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    Oversampler,
    PerColumnImputer,
    PolynomialDetrender,
    ProphetRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RFClassifierSelectFromModel,
    RFRegressorSelectFromModel,
    SelectByType,
    SelectColumns,
    SimpleImputer,
    StandardScaler,
    SVMClassifier,
    SVMRegressor,
    TargetImputer,
    TextFeaturizer,
    TimeSeriesBaselineEstimator,
    Transformer,
    Undersampler,
    XGBoostClassifier,
    XGBoostRegressor,
)
from evalml.pipelines.components.ensemble import (
    SklearnStackedEnsembleClassifier,
    SklearnStackedEnsembleRegressor,
    StackedEnsembleBase,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor,
)
from evalml.pipelines.components.transformers.encoders.label_encoder import (
    LabelEncoder,
)
from evalml.pipelines.components.transformers.preprocessing.log_transformer import (
    LogTransformer,
)
from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseSampler,
)
from evalml.pipelines.components.utils import (
    _all_estimators,
    _all_transformers,
    all_components,
    generate_component_code,
)
from evalml.problem_types import ProblemTypes


@pytest.fixture(scope="module")
def test_classes():
    class MockComponent(ComponentBase):
        name = "Mock Component"
        modifies_features = True
        modifies_target = False
        training_only = False

    class MockEstimator(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ["binary"]

    class MockTransformer(Transformer):
        name = "Mock Transformer"

        def transform(self, X, y=None):
            return X

    return MockComponent, MockEstimator, MockTransformer


@pytest.fixture(scope="module")
def test_estimator_needs_fitting_false():
    class MockEstimatorNeedsFittingFalse(Estimator):
        name = "Mock Estimator Needs Fitting False"
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ["binary"]
        needs_fitting = False

        def predict(self, X):
            pass

    return MockEstimatorNeedsFittingFalse


class MockFitComponent(ComponentBase):
    name = "Mock Fit Component"
    modifies_features = True
    modifies_target = False
    training_only = False

    def __init__(self, param_a=2, param_b=10, random_seed=0):
        parameters = {"param_a": param_a, "param_b": param_b}
        super().__init__(parameters=parameters, component_obj=None, random_seed=0)

    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.array(
            [self.parameters["param_a"] * 2, self.parameters["param_b"] * 10]
        )


def test_init(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    assert MockComponent().name == "Mock Component"
    assert MockEstimator().name == "Mock Estimator"
    assert MockTransformer().name == "Mock Transformer"


def test_describe(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    params = {"param_a": "value_a", "param_b": 123}
    component = MockComponent(parameters=params)
    assert component.describe(return_dict=True) == {
        "name": "Mock Component",
        "parameters": params,
    }
    estimator = MockEstimator(parameters=params)
    assert estimator.describe(return_dict=True) == {
        "name": "Mock Estimator",
        "parameters": params,
    }
    transformer = MockTransformer(parameters=params)
    assert transformer.describe(return_dict=True) == {
        "name": "Mock Transformer",
        "parameters": params,
    }


def test_describe_component():
    enc = OneHotEncoder()
    imputer = Imputer()
    simple_imputer = SimpleImputer("mean")
    column_imputer = PerColumnImputer({"a": "mean", "b": ("constant", 100)})
    scaler = StandardScaler()
    feature_selection_clf = RFClassifierSelectFromModel(
        n_estimators=10, number_features=5, percent_features=0.3, threshold=-np.inf
    )
    feature_selection_reg = RFRegressorSelectFromModel(
        n_estimators=10, number_features=5, percent_features=0.3, threshold=-np.inf
    )
    drop_col_transformer = DropColumns(columns=["col_one", "col_two"])
    drop_null_transformer = DropNullColumns()
    datetime = DateTimeFeaturizer()
    text_featurizer = TextFeaturizer()
    lsa = LSA()
    pca = PCA()
    lda = LinearDiscriminantAnalysis()
    ft = DFSTransformer()
    us = Undersampler()
    assert enc.describe(return_dict=True) == {
        "name": "One Hot Encoder",
        "parameters": {
            "top_n": 10,
            "features_to_encode": None,
            "categories": None,
            "drop": "if_binary",
            "handle_unknown": "ignore",
            "handle_missing": "error",
        },
    }
    assert imputer.describe(return_dict=True) == {
        "name": "Imputer",
        "parameters": {
            "categorical_impute_strategy": "most_frequent",
            "categorical_fill_value": None,
            "numeric_impute_strategy": "mean",
            "numeric_fill_value": None,
        },
    }
    assert simple_imputer.describe(return_dict=True) == {
        "name": "Simple Imputer",
        "parameters": {"impute_strategy": "mean", "fill_value": None},
    }
    assert column_imputer.describe(return_dict=True) == {
        "name": "Per Column Imputer",
        "parameters": {
            "impute_strategies": {"a": "mean", "b": ("constant", 100)},
            "default_impute_strategy": "most_frequent",
        },
    }
    assert scaler.describe(return_dict=True) == {
        "name": "Standard Scaler",
        "parameters": {},
    }
    assert feature_selection_clf.describe(return_dict=True) == {
        "name": "RF Classifier Select From Model",
        "parameters": {
            "number_features": 5,
            "n_estimators": 10,
            "max_depth": None,
            "percent_features": 0.3,
            "threshold": -np.inf,
            "n_jobs": -1,
        },
    }
    assert feature_selection_reg.describe(return_dict=True) == {
        "name": "RF Regressor Select From Model",
        "parameters": {
            "number_features": 5,
            "n_estimators": 10,
            "max_depth": None,
            "percent_features": 0.3,
            "threshold": -np.inf,
            "n_jobs": -1,
        },
    }
    assert drop_col_transformer.describe(return_dict=True) == {
        "name": "Drop Columns Transformer",
        "parameters": {"columns": ["col_one", "col_two"]},
    }
    assert drop_null_transformer.describe(return_dict=True) == {
        "name": "Drop Null Columns Transformer",
        "parameters": {"pct_null_threshold": 1.0},
    }
    assert datetime.describe(return_dict=True) == {
        "name": "DateTime Featurization Component",
        "parameters": {
            "features_to_extract": ["year", "month", "day_of_week", "hour"],
            "encode_as_categories": False,
            "date_index": None,
        },
    }
    assert text_featurizer.describe(return_dict=True) == {
        "name": "Text Featurization Component",
        "parameters": {},
    }
    assert lsa.describe(return_dict=True) == {
        "name": "LSA Transformer",
        "parameters": {},
    }
    assert pca.describe(return_dict=True) == {
        "name": "PCA Transformer",
        "parameters": {"n_components": None, "variance": 0.95},
    }
    assert lda.describe(return_dict=True) == {
        "name": "Linear Discriminant Analysis Transformer",
        "parameters": {"n_components": None},
    }
    assert ft.describe(return_dict=True) == {
        "name": "DFS Transformer",
        "parameters": {"index": "index"},
    }
    assert us.describe(return_dict=True) == {
        "name": "Undersampler",
        "parameters": {
            "sampling_ratio": 0.25,
            "sampling_ratio_dict": None,
            "min_samples": 100,
            "min_percentage": 0.1,
        },
    }
    try:
        oversampler = Oversampler()
        assert oversampler.describe(return_dict=True) == {
            "name": "Oversampler",
            "parameters": {
                "sampling_ratio": 0.25,
                "sampling_ratio_dict": None,
                "k_neighbors_default": 5,
                "n_jobs": -1,
            },
        }
    except ImportError:
        pass
    # testing estimators
    base_classifier = BaselineClassifier()
    base_regressor = BaselineRegressor()
    lr_classifier = LogisticRegressionClassifier()
    en_classifier = ElasticNetClassifier()
    en_regressor = ElasticNetRegressor()
    et_classifier = ExtraTreesClassifier(n_estimators=10, max_features="auto")
    et_regressor = ExtraTreesRegressor(n_estimators=10, max_features="auto")
    rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=3)
    rf_regressor = RandomForestRegressor(n_estimators=10, max_depth=3)
    linear_regressor = LinearRegressor()
    svm_classifier = SVMClassifier()
    svm_regressor = SVMRegressor()
    assert base_classifier.describe(return_dict=True) == {
        "name": "Baseline Classifier",
        "parameters": {"strategy": "mode"},
    }
    assert base_regressor.describe(return_dict=True) == {
        "name": "Baseline Regressor",
        "parameters": {"strategy": "mean"},
    }
    assert lr_classifier.describe(return_dict=True) == {
        "name": "Logistic Regression Classifier",
        "parameters": {
            "penalty": "l2",
            "C": 1.0,
            "n_jobs": -1,
            "multi_class": "auto",
            "solver": "lbfgs",
        },
    }
    assert en_classifier.describe(return_dict=True) == {
        "name": "Elastic Net Classifier",
        "parameters": {
            "C": 1.0,
            "l1_ratio": 0.15,
            "n_jobs": -1,
            "multi_class": "auto",
            "solver": "saga",
            "penalty": "elasticnet",
        },
    }
    assert en_regressor.describe(return_dict=True) == {
        "name": "Elastic Net Regressor",
        "parameters": {
            "alpha": 0.0001,
            "l1_ratio": 0.15,
            "max_iter": 1000,
            "normalize": False,
        },
    }
    assert et_classifier.describe(return_dict=True) == {
        "name": "Extra Trees Classifier",
        "parameters": {
            "n_estimators": 10,
            "max_features": "auto",
            "max_depth": 6,
            "min_samples_split": 2,
            "min_weight_fraction_leaf": 0.0,
            "n_jobs": -1,
        },
    }
    assert et_regressor.describe(return_dict=True) == {
        "name": "Extra Trees Regressor",
        "parameters": {
            "n_estimators": 10,
            "max_features": "auto",
            "max_depth": 6,
            "min_samples_split": 2,
            "min_weight_fraction_leaf": 0.0,
            "n_jobs": -1,
        },
    }
    assert rf_classifier.describe(return_dict=True) == {
        "name": "Random Forest Classifier",
        "parameters": {"n_estimators": 10, "max_depth": 3, "n_jobs": -1},
    }
    assert rf_regressor.describe(return_dict=True) == {
        "name": "Random Forest Regressor",
        "parameters": {"n_estimators": 10, "max_depth": 3, "n_jobs": -1},
    }
    assert linear_regressor.describe(return_dict=True) == {
        "name": "Linear Regressor",
        "parameters": {"fit_intercept": True, "normalize": False, "n_jobs": -1},
    }
    assert svm_classifier.describe(return_dict=True) == {
        "name": "SVM Classifier",
        "parameters": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "auto",
            "probability": True,
        },
    }
    assert svm_regressor.describe(return_dict=True) == {
        "name": "SVM Regressor",
        "parameters": {"C": 1.0, "kernel": "rbf", "gamma": "auto"},
    }
    try:
        xgb_classifier = XGBoostClassifier(
            eta=0.1, min_child_weight=1, max_depth=3, n_estimators=75
        )
        xgb_regressor = XGBoostRegressor(
            eta=0.1, min_child_weight=1, max_depth=3, n_estimators=75
        )
        assert xgb_classifier.describe(return_dict=True) == {
            "name": "XGBoost Classifier",
            "parameters": {
                "eta": 0.1,
                "max_depth": 3,
                "min_child_weight": 1,
                "n_estimators": 75,
                "n_jobs": 12,
                "eval_metric": "logloss",
            },
        }
        assert xgb_regressor.describe(return_dict=True) == {
            "name": "XGBoost Regressor",
            "parameters": {
                "eta": 0.1,
                "max_depth": 3,
                "min_child_weight": 1,
                "n_estimators": 75,
                "n_jobs": 12,
            },
        }
    except ImportError:
        pass
    try:
        cb_classifier = CatBoostClassifier()
        cb_regressor = CatBoostRegressor()
        assert cb_classifier.describe(return_dict=True) == {
            "name": "CatBoost Classifier",
            "parameters": {
                "allow_writing_files": False,
                "n_estimators": 10,
                "eta": 0.03,
                "max_depth": 6,
                "bootstrap_type": None,
                "silent": True,
                "n_jobs": -1,
            },
        }
        assert cb_regressor.describe(return_dict=True) == {
            "name": "CatBoost Regressor",
            "parameters": {
                "allow_writing_files": False,
                "n_estimators": 10,
                "eta": 0.03,
                "max_depth": 6,
                "bootstrap_type": None,
                "silent": False,
                "n_jobs": -1,
            },
        }
    except ImportError:
        pass
    try:
        lg_classifier = LightGBMClassifier()
        lg_regressor = LightGBMRegressor()
        assert lg_classifier.describe(return_dict=True) == {
            "name": "LightGBM Classifier",
            "parameters": {
                "boosting_type": "gbdt",
                "learning_rate": 0.1,
                "n_estimators": 100,
                "max_depth": 0,
                "num_leaves": 31,
                "min_child_samples": 20,
                "n_jobs": -1,
                "bagging_fraction": 0.9,
                "bagging_freq": 0,
            },
        }
        assert lg_regressor.describe(return_dict=True) == {
            "name": "LightGBM Regressor",
            "parameters": {
                "boosting_type": "gbdt",
                "learning_rate": 0.1,
                "n_estimators": 20,
                "max_depth": 0,
                "num_leaves": 31,
                "min_child_samples": 20,
                "n_jobs": -1,
                "bagging_fraction": 0.9,
                "bagging_freq": 0,
            },
        }
    except ImportError:
        pass
    try:
        prophet_regressor = ProphetRegressor()
        assert prophet_regressor.describe(return_dict=True) == {
            "name": "Prophet Regressor",
            "parameters": {
                "changepoint_prior_scale": 0.05,
                "date_index": None,
                "holidays_prior_scale": 10,
                "seasonality_mode": "additive",
                "seasonality_prior_scale": 10,
                "stan_backend": "CMDSTANPY",
            },
        }
    except ImportError:
        pass


def test_missing_attributes(X_y_binary):
    class MockComponentName(ComponentBase):
        pass

    with pytest.raises(TypeError):
        MockComponentName()

    class MockComponentModelFamily(ComponentBase):
        name = "Mock Component"

    with pytest.raises(TypeError):
        MockComponentModelFamily()

    class MockEstimatorWithoutAttribute(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL

    with pytest.raises(TypeError):
        MockEstimatorWithoutAttribute()


def test_missing_methods_on_components(X_y_binary, test_classes):
    X, y = X_y_binary
    MockComponent, MockEstimator, MockTransformer = test_classes

    component = MockComponent()
    with pytest.raises(
        MethodPropertyNotFoundError,
        match="Component requires a fit method or a component_obj that implements fit",
    ):
        component.fit(X)

    estimator = MockEstimator()
    estimator._is_fitted = True
    with pytest.raises(
        MethodPropertyNotFoundError,
        match="Estimator requires a predict method or a component_obj that implements predict",
    ):
        estimator.predict(X)
    with pytest.raises(
        MethodPropertyNotFoundError,
        match="Estimator requires a predict_proba method or a component_obj that implements predict_proba",
    ):
        estimator.predict_proba(X)
    with pytest.raises(
        MethodPropertyNotFoundError,
        match="Estimator requires a feature_importance property or a component_obj that implements feature_importances_",
    ):
        estimator.feature_importance

    transformer = MockTransformer()
    transformer._is_fitted = True
    with pytest.raises(
        MethodPropertyNotFoundError,
        match="Component requires a fit method or a component_obj that implements fit",
    ):
        transformer.fit(X, y)
        transformer.transform(X)
    with pytest.raises(
        MethodPropertyNotFoundError,
        match="Component requires a fit method or a component_obj that implements fit",
    ):
        transformer.fit_transform(X)


def test_component_fit(X_y_binary):
    X, y = X_y_binary

    class MockEstimator:
        def fit(self, X, y):
            pass

    class MockComponent(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ["binary"]
        hyperparameter_ranges = {}

        def __init__(self):
            parameters = {}
            est = MockEstimator()
            super().__init__(parameters=parameters, component_obj=est, random_seed=0)

    est = MockComponent()
    assert isinstance(est.fit(X, y), ComponentBase)


def test_component_fit_transform(X_y_binary):
    X, y = X_y_binary

    class MockTransformerWithFitTransform(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit_transform(self, X, y=None):
            return X

        def transform(self, X, y=None):
            return X

        def __init__(self):
            parameters = {}
            super().__init__(parameters=parameters, component_obj=None, random_seed=0)

    class MockTransformerWithFitTransformButError(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit_transform(self, X, y=None):
            raise RuntimeError

        def transform(self, X, y=None):
            return X

        def __init__(self):
            parameters = {}
            super().__init__(parameters=parameters, component_obj=None, random_seed=0)

    class MockTransformerWithFitAndTransform(Transformer):
        name = "Mock Transformer"
        hyperparameter_ranges = {}

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            return X

        def __init__(self):
            parameters = {}
            super().__init__(parameters=parameters, component_obj=None, random_seed=0)

    # convert data to pd DataFrame, because the component classes don't
    # standardize to pd DataFrame
    X = pd.DataFrame(X)
    y = pd.Series(y)

    component = MockTransformerWithFitTransform()
    assert isinstance(component.fit_transform(X, y), pd.DataFrame)

    component = MockTransformerWithFitTransformButError()
    with pytest.raises(RuntimeError):
        component.fit_transform(X, y)

    component = MockTransformerWithFitAndTransform()
    assert isinstance(component.fit_transform(X, y), pd.DataFrame)


def test_model_family_components(test_classes):
    _, MockEstimator, _ = test_classes

    assert MockEstimator.model_family == ModelFamily.LINEAR_MODEL


def test_regressor_call_predict_proba(test_classes):
    X = np.array([])
    _, MockEstimator, _ = test_classes
    component = MockEstimator()
    component._is_fitted = True
    with pytest.raises(MethodPropertyNotFoundError):
        component.predict_proba(X)


def test_component_describe(test_classes, caplog):
    MockComponent, _, _ = test_classes
    component = MockComponent()
    component.describe(print_name=True)
    out = caplog.text
    assert "Mock Component" in out


def test_component_parameters_getter(test_classes):
    MockComponent, _, _ = test_classes
    component = MockComponent({"test": "parameter"})
    assert component.parameters == {"test": "parameter"}
    component.parameters["test"] = "new"
    assert component.parameters == {"test": "parameter"}


def test_component_parameters_init(
    logistic_regression_binary_pipeline_class, linear_regression_pipeline_class
):
    for component_class in all_components():
        print("Testing component {}".format(component_class.name))
        try:
            component = component_class()
        except EnsembleMissingPipelinesError:
            if component_class == SklearnStackedEnsembleClassifier:
                component = component_class(
                    input_pipelines=[
                        logistic_regression_binary_pipeline_class(parameters={})
                    ]
                )
            elif component_class == SklearnStackedEnsembleRegressor:
                component = component_class(
                    input_pipelines=[linear_regression_pipeline_class(parameters={})]
                )
        parameters = component.parameters

        component2 = component_class(**parameters)
        parameters2 = component2.parameters

        assert parameters == parameters2


def test_clone_init():
    params = {"param_a": 2, "param_b": 11}
    clf = MockFitComponent(**params)
    clf_clone = clf.clone()
    assert clf.parameters == clf_clone.parameters
    assert clf_clone.random_seed == clf.random_seed


def test_clone_fitted(X_y_binary):
    X, y = X_y_binary
    params = {"param_a": 3, "param_b": 7}
    clf = MockFitComponent(**params)
    clf.fit(X, y)
    predicted = clf.predict(X)

    clf_clone = clf.clone()
    assert clf_clone.random_seed == clf.random_seed
    assert clf.parameters == clf_clone.parameters

    with pytest.raises(ComponentNotYetFittedError, match="You must fit"):
        clf_clone.predict(X)

    clf_clone.fit(X, y)
    predicted_clone = clf_clone.predict(X)
    np.testing.assert_almost_equal(predicted, predicted_clone)


def test_components_init_kwargs():
    for component_class in all_components():
        try:
            component = component_class()
        except EnsembleMissingPipelinesError:
            continue
        if component._component_obj is None:
            continue
        if isinstance(component, StackedEnsembleBase):
            continue

        obj_class = component._component_obj.__class__.__name__
        module = component._component_obj.__module__
        importlib.import_module(module, obj_class)
        patched = module + "." + obj_class + ".__init__"
        if component_class == LabelEncoder:
            # scikit-learn's LabelEncoder found in different module than where we import from
            patched = module[: module.rindex(".")] + "." + obj_class + ".__init__"

        def all_init(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        with patch(patched, new=all_init) as _:
            component = component_class(test_arg="test")
            component_with_different_kwargs = component_class(diff_test_arg="test")
            assert component.parameters["test_arg"] == "test"
            if not isinstance(component, (PolynomialDetrender, LabelEncoder)):
                assert component._component_obj.test_arg == "test"
            # Test equality of different components with same or different kwargs
            assert component == component_class(test_arg="test")
            assert component != component_with_different_kwargs


def test_component_has_random_seed():
    for component_class in all_components():
        params = inspect.signature(component_class.__init__).parameters
        assert "random_seed" in params


def test_transformer_transform_output_type(X_y_binary):
    X_np, y_np = X_y_binary
    assert isinstance(X_np, np.ndarray)
    assert isinstance(y_np, np.ndarray)
    y_list = list(y_np)
    X_df_no_col_names = pd.DataFrame(X_np)
    range_index = pd.RangeIndex(start=0, stop=X_np.shape[1], step=1)
    X_df_with_col_names = pd.DataFrame(
        X_np, columns=["x" + str(i) for i in range(X_np.shape[1])]
    )
    y_series_no_name = pd.Series(y_np)
    y_series_with_name = pd.Series(y_np, name="target")
    datatype_combos = [
        (X_np, y_np, range_index),
        (X_np, y_list, range_index),
        (X_df_no_col_names, y_series_no_name, range_index),
        (X_df_with_col_names, y_series_with_name, X_df_with_col_names.columns),
    ]

    for component_class in _all_transformers():
        if component_class in [PolynomialDetrender, LogTransformer, LabelEncoder]:
            # Skipping because these tests are handled in their respective test files
            continue
        print("Testing transformer {}".format(component_class.name))
        for X, y, X_cols_expected in datatype_combos:
            print(
                'Checking output of transform for transformer "{}" on X type {} cols {}, y type {} name {}'.format(
                    component_class.name,
                    type(X),
                    X.columns if isinstance(X, pd.DataFrame) else None,
                    type(y),
                    y.name if isinstance(y, pd.Series) else None,
                )
            )

            component = component_class()
            # SMOTE will throw an error if we pass a ratio lower than the current class balance
            if "Oversampler" == component_class.name:
                component = component_class(sampling_ratio=1)
                # we cover this case in test_oversamplers
                continue

            component.fit(X, y=y)
            transform_output = component.transform(X, y=y)

            if component.modifies_target:
                assert isinstance(transform_output[0], pd.DataFrame)
                assert isinstance(transform_output[1], pd.Series)
            else:
                assert isinstance(transform_output, pd.DataFrame)

            if isinstance(component, SelectColumns) or isinstance(
                component, SelectByType
            ):
                assert transform_output.shape == (X.shape[0], 0)
            elif isinstance(component, PCA) or isinstance(
                component, LinearDiscriminantAnalysis
            ):
                assert transform_output.shape[0] == X.shape[0]
                assert transform_output.shape[1] <= X.shape[1]
            elif isinstance(component, DFSTransformer):
                assert transform_output.shape[0] == X.shape[0]
                assert transform_output.shape[1] >= X.shape[1]
            elif isinstance(component, DelayedFeatureTransformer):
                # We just want to check that DelayedFeaturesTransformer outputs a DataFrame
                # The dataframe shape and index are checked in test_delayed_features_transformer.py
                continue
            elif component.modifies_target:
                assert transform_output[0].shape == X.shape
                assert transform_output[1].shape[0] == X.shape[0]
                assert len(transform_output[1].shape) == 1
            else:
                assert transform_output.shape == X.shape
                assert list(transform_output.columns) == list(X_cols_expected)

            transform_output = component.fit_transform(X, y=y)
            if component.modifies_target:
                assert isinstance(transform_output[0], pd.DataFrame)
                assert isinstance(transform_output[1], pd.Series)
            else:
                assert isinstance(transform_output, pd.DataFrame)

            if isinstance(component, SelectColumns) or isinstance(
                component, SelectByType
            ):
                assert transform_output.shape == (X.shape[0], 0)
            elif isinstance(component, PCA) or isinstance(
                component, LinearDiscriminantAnalysis
            ):
                assert transform_output.shape[0] == X.shape[0]
                assert transform_output.shape[1] <= X.shape[1]
            elif isinstance(component, DFSTransformer):
                assert transform_output.shape[0] == X.shape[0]
                assert transform_output.shape[1] >= X.shape[1]
            elif component.modifies_target:
                assert transform_output[0].shape == X.shape
                assert transform_output[1].shape[0] == X.shape[0]
                assert len(transform_output[1].shape) == 1

            else:
                assert transform_output.shape == X.shape
                assert list(transform_output.columns) == list(X_cols_expected)


@pytest.mark.parametrize(
    "cls",
    [
        cls
        for cls in all_components()
        if cls
        not in [
            SklearnStackedEnsembleRegressor,
            SklearnStackedEnsembleClassifier,
            StackedEnsembleClassifier,
            StackedEnsembleRegressor,
        ]
    ],
)
def test_default_parameters(cls):
    assert (
        cls.default_parameters == cls().parameters
    ), f"{cls.__name__}'s default parameters don't match __init__."


@pytest.mark.parametrize(
    "cls",
    [
        cls
        for cls in all_components()
        if cls
        not in [SklearnStackedEnsembleRegressor, SklearnStackedEnsembleClassifier]
    ],
)
def test_default_parameters_raise_no_warnings(cls):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cls()
        assert len(w) == 0


def test_estimator_check_for_fit(X_y_binary):
    class MockEstimatorObj:
        def __init__(self):
            pass

        def fit(self, X, y):
            return self

        def predict(self, X):
            series = pd.Series(dtype="string")
            series.ww.init()
            return series

        def predict_proba(self, X):
            df = pd.DataFrame()
            df.ww.init()
            return df

    class MockEstimator(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ["binary"]

        def __init__(self, parameters=None, component_obj=None, random_seed=0):
            est = MockEstimatorObj()
            super().__init__(
                parameters=parameters, component_obj=est, random_seed=random_seed
            )

    X, y = X_y_binary
    est = MockEstimator()
    with pytest.raises(ComponentNotYetFittedError, match="You must fit"):
        est.predict(X)
    with pytest.raises(ComponentNotYetFittedError, match="You must fit"):
        est.predict_proba(X)

    est.fit(X, y)
    est.predict(X)
    est.predict_proba(X)


def test_transformer_check_for_fit(X_y_binary):
    class MockTransformerObj:
        def __init__(self):
            pass

        def fit(self, X, y):
            return self

        def transform(self, X, y=None):
            return X

        def fit_transform(self, X, y=None):
            return X

    class MockTransformer(Transformer):
        name = "Mock Transformer"

        def __init__(self, parameters=None, component_obj=None, random_seed=0):
            transformer = MockTransformerObj()
            super().__init__(
                parameters=parameters,
                component_obj=transformer,
                random_seed=random_seed,
            )

        def transform(self, X, y=None):
            return X

        def inverse_transform(self, X, y=None):
            return X, y

    X, y = X_y_binary
    trans = MockTransformer()
    with pytest.raises(ComponentNotYetFittedError, match="You must fit"):
        trans.transform(X)

    with pytest.raises(ComponentNotYetFittedError, match="You must fit"):
        trans.inverse_transform(X, y)

    trans.fit(X, y)
    trans.transform(X)
    trans.fit_transform(X, y)
    trans.inverse_transform(X, y)


def test_transformer_check_for_fit_with_overrides(X_y_binary):
    class MockTransformerWithOverride(Transformer):
        name = "Mock Transformer"

        def fit(self, X, y):
            return self

        def transform(self, X, y=None):
            df = pd.DataFrame()
            df.ww.init()
            return df

    class MockTransformerWithOverrideSubclass(Transformer):
        name = "Mock Transformer Subclass"

        def fit(self, X, y):
            return self

        def transform(self, X, y=None):
            df = pd.DataFrame()
            df.ww.init()
            return df

    X, y = X_y_binary
    transformer = MockTransformerWithOverride()
    transformer_subclass = MockTransformerWithOverrideSubclass()

    with pytest.raises(ComponentNotYetFittedError, match="You must fit"):
        transformer.transform(X)
    with pytest.raises(ComponentNotYetFittedError, match="You must fit"):
        transformer_subclass.transform(X)

    transformer.fit(X, y)
    transformer.transform(X)
    transformer_subclass.fit(X, y)
    transformer_subclass.transform(X)


def test_all_transformers_needs_fitting():
    for component_class in _all_transformers() + _all_estimators():
        if component_class.__name__ in [
            "DropColumns",
            "SelectColumns",
            "SelectByType",
            "DelayedFeatureTransformer",
        ]:
            assert not component_class.needs_fitting
        else:
            assert component_class.needs_fitting


def test_all_transformers_check_fit(X_y_binary):
    X, y = X_y_binary
    for component_class in _all_transformers():
        if not component_class.needs_fitting:
            continue

        component = component_class()
        # SMOTE will throw errors if we call it but cannot oversample
        if "Oversampler" == component_class.name:
            component = component_class(sampling_ratio=1)

        with pytest.raises(
            ComponentNotYetFittedError, match=f"You must fit {component_class.__name__}"
        ):
            component.transform(X, y)

        component.fit(X, y)
        component.transform(X, y)

        component = component_class()
        if "Oversampler" == component_class.name:
            component = component_class(sampling_ratio=1)
        component.fit_transform(X, y)
        component.transform(X, y)


def test_all_estimators_check_fit(
    X_y_binary, ts_data, test_estimator_needs_fitting_false, helper_functions
):
    estimators_to_check = [
        estimator
        for estimator in _all_estimators()
        if estimator
        not in [
            SklearnStackedEnsembleClassifier,
            SklearnStackedEnsembleRegressor,
            StackedEnsembleClassifier,
            StackedEnsembleRegressor,
            TimeSeriesBaselineEstimator,
        ]
    ] + [test_estimator_needs_fitting_false]
    for component_class in estimators_to_check:
        if not component_class.needs_fitting:
            continue

        if (
            ProblemTypes.TIME_SERIES_REGRESSION
            in component_class.supported_problem_types
        ):
            X, y = ts_data
        else:
            X, y = X_y_binary

        component = helper_functions.safe_init_component_with_njobs_1(component_class)
        with patch.object(component, "_component_obj"):
            with pytest.raises(
                ComponentNotYetFittedError,
                match=f"You must fit {component_class.__name__}",
            ):
                component.predict(X)
            if (
                ProblemTypes.BINARY in component.supported_problem_types
                or ProblemTypes.MULTICLASS in component.supported_problem_types
            ):
                with pytest.raises(
                    ComponentNotYetFittedError,
                    match=f"You must fit {component_class.__name__}",
                ):
                    component.predict_proba(X)

            with pytest.raises(
                ComponentNotYetFittedError,
                match=f"You must fit {component_class.__name__}",
            ):
                component.feature_importance

            component.fit(X, y)

            if (
                ProblemTypes.BINARY in component.supported_problem_types
                or ProblemTypes.MULTICLASS in component.supported_problem_types
            ):
                component.predict_proba(X)

            component.predict(X)
            component.feature_importance


@pytest.mark.parametrize("data_type", ["li", "np", "pd", "ww"])
def test_all_transformers_check_fit_input_type(data_type, X_y_binary, make_data_type):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    for component_class in _all_transformers():
        if not component_class.needs_fitting or "Oversampler" in component_class.name:
            # since SMOTE determines categorical columns through the logical type, it can only accept ww data
            continue

        component = component_class()
        component.fit(X, y)


def test_no_fitting_required_components(
    X_y_binary, test_estimator_needs_fitting_false, helper_functions
):
    X, y = X_y_binary
    for component_class in all_components() + [test_estimator_needs_fitting_false]:
        if not component_class.needs_fitting:
            component = helper_functions.safe_init_component_with_njobs_1(
                component_class
            )
            if issubclass(component_class, Estimator):
                component.predict(X)
            else:
                component.transform(X, y)


def test_serialization(X_y_binary, ts_data, tmpdir, helper_functions):
    path = os.path.join(str(tmpdir), "component.pkl")
    for component_class in all_components():
        print("Testing serialization of component {}".format(component_class.name))
        try:
            component = helper_functions.safe_init_component_with_njobs_1(
                component_class
            )
        except EnsembleMissingPipelinesError:
            if component_class == SklearnStackedEnsembleClassifier:
                component = component_class(
                    input_pipelines=[
                        BinaryClassificationPipeline(
                            [RandomForestClassifier],
                            parameters={"Random Forest Classifier": {"n_jobs": 1}},
                        )
                    ]
                )
            elif component_class == SklearnStackedEnsembleRegressor:
                component = component_class(
                    input_pipelines=[
                        RegressionPipeline(
                            [RandomForestRegressor],
                            parameters={"Random Forest Regressor": {"n_jobs": 1}},
                        )
                    ]
                )
        if (
            isinstance(component, Estimator)
            and ProblemTypes.TIME_SERIES_REGRESSION in component.supported_problem_types
        ):
            X, y = ts_data
        else:
            X, y = X_y_binary

        component.fit(X, y)

        for pickle_protocol in range(cloudpickle.DEFAULT_PROTOCOL + 1):
            component.save(path, pickle_protocol=pickle_protocol)
            loaded_component = ComponentBase.load(path)
            assert component.parameters == loaded_component.parameters
            assert component.describe(return_dict=True) == loaded_component.describe(
                return_dict=True
            )
            if issubclass(component_class, Estimator) and not (
                isinstance(component, SklearnStackedEnsembleClassifier)
                or isinstance(component, SklearnStackedEnsembleRegressor)
                or isinstance(component, StackedEnsembleClassifier)
                or isinstance(component, StackedEnsembleRegressor)
            ):
                assert (
                    component.feature_importance == loaded_component.feature_importance
                ).all()


@patch("cloudpickle.dump")
def test_serialization_protocol(mock_cloudpickle_dump, tmpdir):
    path = os.path.join(str(tmpdir), "pipe.pkl")
    component = LogisticRegressionClassifier()

    component.save(path)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert (
        mock_cloudpickle_dump.call_args_list[0][1]["protocol"]
        == cloudpickle.DEFAULT_PROTOCOL
    )

    mock_cloudpickle_dump.reset_mock()

    component.save(path, pickle_protocol=42)
    assert len(mock_cloudpickle_dump.call_args_list) == 1
    assert mock_cloudpickle_dump.call_args_list[0][1]["protocol"] == 42


@pytest.mark.parametrize("estimator_class", _all_estimators())
def test_estimators_accept_all_kwargs(
    estimator_class,
    logistic_regression_binary_pipeline_class,
    linear_regression_pipeline_class,
):
    try:
        estimator = estimator_class()
    except EnsembleMissingPipelinesError:
        if estimator_class == SklearnStackedEnsembleClassifier:
            estimator = estimator_class(
                input_pipelines=[
                    logistic_regression_binary_pipeline_class(parameters={})
                ]
            )
        elif estimator_class == SklearnStackedEnsembleRegressor:
            estimator = estimator_class(
                input_pipelines=[linear_regression_pipeline_class(parameters={})]
            )
    if estimator._component_obj is None:
        pytest.skip(
            f"Skipping {estimator_class} because does not have component object."
        )
    if estimator_class.model_family == ModelFamily.ENSEMBLE:
        params = estimator.parameters
    elif estimator_class.model_family == ModelFamily.PROPHET:
        params = estimator.get_params()
    else:
        params = estimator._component_obj.get_params()
        if "random_state" in params:
            del params["random_state"]
    estimator_class(**params)


def test_component_equality_different_classes():
    # Tests that two classes which are equivalent are not equal
    class MockComponent(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        modifies_features = True
        modifies_target = False
        training_only = False

    class MockComponentWithADifferentName(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        modifies_features = True
        modifies_target = False
        training_only = False

    assert MockComponent() != MockComponentWithADifferentName()


def test_component_equality_subclasses():
    class MockComponent(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        modifies_features = True
        modifies_target = False
        training_only = False

    class MockEstimatorSubclass(MockComponent):
        pass

    assert MockComponent() != MockEstimatorSubclass()


def test_component_equality():
    class MockComponent(ComponentBase):
        name = "Mock Component"
        model_family = ModelFamily.NONE
        modifies_features = True
        modifies_target = False
        training_only = False

        def __init__(self, param_1=0, param_2=0, random_seed=0, **kwargs):
            parameters = {"param_1": param_1, "param_2": param_2}
            parameters.update(kwargs)
            super().__init__(
                parameters=parameters, component_obj=None, random_seed=random_seed
            )

        def fit(self, X, y=None):
            return self

    # Test self-equality
    mock_component = MockComponent()
    assert mock_component == mock_component

    # Test defaults
    assert MockComponent() == MockComponent()

    # Test random_state and random_seed
    assert MockComponent(random_seed=10) == MockComponent(random_seed=10)
    assert MockComponent(random_seed=10) != MockComponent(random_seed=0)

    # Test parameters
    assert MockComponent(1, 2) == MockComponent(1, 2)
    assert MockComponent(1, 2) != MockComponent(1, 0)
    assert MockComponent(0, 2) != MockComponent(1, 2)

    # Test fitted equality
    mock_component.fit(pd.DataFrame({}))
    assert mock_component != MockComponent()


@pytest.mark.parametrize("component_class", all_components())
def test_component_equality_all_components(
    component_class,
    logistic_regression_binary_pipeline_class,
    linear_regression_pipeline_class,
):
    if component_class == SklearnStackedEnsembleClassifier:
        component = component_class(
            input_pipelines=[logistic_regression_binary_pipeline_class(parameters={})]
        )
    elif component_class == SklearnStackedEnsembleRegressor:
        component = component_class(
            input_pipelines=[linear_regression_pipeline_class(parameters={})]
        )
    else:
        component = component_class()
    parameters = component.parameters
    equal_component = component_class(**parameters)
    assert component == equal_component


def test_component_equality_with_subclasses(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes
    mock_component = MockComponent()
    mock_estimator = MockEstimator()
    mock_transformer = MockTransformer()
    assert mock_component != mock_estimator
    assert mock_component != mock_transformer
    assert mock_estimator != mock_component
    assert mock_estimator != mock_transformer
    assert mock_transformer != mock_component
    assert mock_transformer != mock_estimator


def test_mock_component_str(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes

    assert str(MockComponent()) == "Mock Component"
    assert str(MockEstimator()) == "Mock Estimator"
    assert str(MockTransformer()) == "Mock Transformer"


def test_mock_component_repr():
    component = MockFitComponent()
    assert repr(component) == "MockFitComponent(param_a=2, param_b=10)"

    component_with_params = MockFitComponent(param_a=29, param_b=None, random_seed=42)
    assert repr(component_with_params) == "MockFitComponent(param_a=29, param_b=None)"

    component_with_nan = MockFitComponent(param_a=np.nan, param_b=float("nan"))
    assert (
        repr(component_with_nan) == "MockFitComponent(param_a=np.nan, param_b=np.nan)"
    )

    component_with_inf = MockFitComponent(param_a=np.inf, param_b=float("-inf"))
    assert (
        repr(component_with_inf)
        == "MockFitComponent(param_a=float('inf'), param_b=float('-inf'))"
    )


@pytest.mark.parametrize("component_class", all_components())
def test_component_str(
    component_class,
    logistic_regression_binary_pipeline_class,
    linear_regression_pipeline_class,
):
    try:
        component = component_class()
    except EnsembleMissingPipelinesError:
        if component_class == SklearnStackedEnsembleClassifier:
            component = component_class(
                input_pipelines=[
                    logistic_regression_binary_pipeline_class(parameters={})
                ]
            )
        elif component_class == SklearnStackedEnsembleRegressor:
            component = component_class(
                input_pipelines=[linear_regression_pipeline_class(parameters={})]
            )
    assert str(component) == component.name


@pytest.mark.parametrize(
    "categorical",
    [
        {
            "type": Categorical(["mean", "median", "mode"]),
            "categories": Categorical(["blue", "green"]),
        },
        {"type": ["mean", "median", "mode"], "categories": ["blue", "green"]},
    ],
)
def test_categorical_hyperparameters(X_y_binary, categorical):
    X, y = X_y_binary

    class MockEstimator:
        def fit(self, X, y):
            pass

    class MockComponent(Estimator):
        name = "Mock Estimator"
        model_family = ModelFamily.LINEAR_MODEL
        supported_problem_types = ["binary"]
        hyperparameter_ranges = categorical

        def __init__(self, agg_type, category="green"):
            parameters = {"type": agg_type, "categories": category}
            est = MockEstimator()
            super().__init__(parameters=parameters, component_obj=est, random_seed=0)

    assert MockComponent(agg_type="mean").fit(X, y)
    assert MockComponent(agg_type="moat", category="blue").fit(X, y)


def test_generate_code_errors():
    with pytest.raises(ValueError, match="Element must be a component instance"):
        generate_component_code(BinaryClassificationPipeline([RandomForestClassifier]))

    with pytest.raises(ValueError, match="Element must be a component instance"):
        generate_component_code(LinearRegressor)

    with pytest.raises(ValueError, match="Element must be a component instance"):
        generate_component_code(Imputer)

    with pytest.raises(ValueError, match="Element must be a component instance"):
        generate_component_code(ComponentBase)


def test_generate_code():
    expected_code = (
        "from evalml.pipelines.components.estimators.classifiers.logistic_regression_classifier import LogisticRegressionClassifier"
        "\n\nlogisticRegressionClassifier = LogisticRegressionClassifier(**{'penalty': 'l2', 'C': 1.0, 'n_jobs': -1, 'multi_class': 'auto', 'solver': 'lbfgs'})"
    )
    component_code = generate_component_code(LogisticRegressionClassifier())
    assert component_code == expected_code

    expected_code = (
        "from evalml.pipelines.components.estimators.regressors.et_regressor import ExtraTreesRegressor"
        "\n\nextraTreesRegressor = ExtraTreesRegressor(**{'n_estimators': 50, 'max_features': 'auto', 'max_depth': 6, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_jobs': -1})"
    )
    component_code = generate_component_code(ExtraTreesRegressor(n_estimators=50))
    assert component_code == expected_code

    expected_code = (
        "from evalml.pipelines.components.transformers.imputers.imputer import Imputer"
        "\n\nimputer = Imputer(**{'categorical_impute_strategy': 'most_frequent', 'numeric_impute_strategy': 'mean', 'categorical_fill_value': None, 'numeric_fill_value': None})"
    )
    component_code = generate_component_code(Imputer())
    assert component_code == expected_code


def test_generate_code_custom(test_classes):
    MockComponent, MockEstimator, MockTransformer = test_classes

    expected_code = "mockComponent = MockComponent(**{})"
    component_code = generate_component_code(MockComponent())
    assert component_code == expected_code

    expected_code = "mockEstimator = MockEstimator(**{})"
    component_code = generate_component_code(MockEstimator())
    assert component_code == expected_code

    expected_code = "mockTransformer = MockTransformer(**{})"
    component_code = generate_component_code(MockTransformer())
    assert component_code == expected_code


@pytest.mark.parametrize("transformer_class", _all_transformers())
@pytest.mark.parametrize("use_custom_index", [True, False])
def test_transformer_fit_and_transform_respect_custom_indices(
    use_custom_index, transformer_class, X_y_binary
):
    check_names = True
    if transformer_class == DFSTransformer:
        check_names = False
        if use_custom_index:
            pytest.skip("The DFSTransformer changes the index so we skip it.")
    if transformer_class == PolynomialDetrender:
        pytest.skip(
            "Skipping PolynomialDetrender because we test that it respects custom indices in "
            "test_polynomial_detrender.py"
        )

    X, y = X_y_binary

    X = pd.DataFrame(X)
    y = pd.Series(y)

    if use_custom_index:
        gen = np.random.default_rng(seed=0)
        custom_index = gen.permutation(range(200, 200 + X.shape[0]))
        X.index = custom_index
        y.index = custom_index

    X_original_index = X.index.copy()
    y_original_index = y.index.copy()

    transformer = transformer_class()

    transformer.fit(X, y)
    pd.testing.assert_index_equal(X.index, X_original_index)
    pd.testing.assert_index_equal(y.index, y_original_index)

    if isinstance(transformer, BaseSampler):
        return
    elif transformer_class.modifies_target:
        X_t, y_t = transformer.transform(X, y)
        pd.testing.assert_index_equal(
            y_t.index, y_original_index, check_names=check_names
        )
    else:
        X_t = transformer.transform(X, y)
        pd.testing.assert_index_equal(
            y.index, y_original_index, check_names=check_names
        )
    pd.testing.assert_index_equal(X_t.index, X_original_index, check_names=check_names)


@pytest.mark.parametrize("estimator_class", _all_estimators())
@pytest.mark.parametrize("use_custom_index", [True, False])
def test_estimator_fit_respects_custom_indices(
    use_custom_index,
    estimator_class,
    X_y_binary,
    X_y_regression,
    ts_data,
    logistic_regression_binary_pipeline_class,
    linear_regression_pipeline_class,
    helper_functions,
):

    if estimator_class == SklearnStackedEnsembleRegressor:
        input_pipelines = [
            helper_functions.safe_init_pipeline_with_njobs_1(
                linear_regression_pipeline_class
            )
        ]
    elif estimator_class == SklearnStackedEnsembleClassifier:
        input_pipelines = [
            helper_functions.safe_init_pipeline_with_njobs_1(
                logistic_regression_binary_pipeline_class
            )
        ]
    else:
        input_pipelines = []

    supported_problem_types = estimator_class.supported_problem_types

    ts_problem = False
    if ProblemTypes.REGRESSION in supported_problem_types:
        X, y = X_y_regression
    elif ProblemTypes.TIME_SERIES_REGRESSION in supported_problem_types:
        X, y = ts_data
        ts_problem = True
    else:
        X, y = X_y_binary

    X = pd.DataFrame(X)
    y = pd.Series(y)

    if use_custom_index and ts_problem:
        X.index = pd.date_range("2020-10-01", "2020-10-31")
        y.index = pd.date_range("2020-10-01", "2020-10-31")
    elif use_custom_index and not ts_problem:
        gen = np.random.default_rng(seed=0)
        custom_index = gen.permutation(range(200, 200 + X.shape[0]))
        X.index = custom_index
        y.index = custom_index

    X_original_index = X.index.copy()
    y_original_index = y.index.copy()

    if input_pipelines:
        estimator = estimator_class(n_jobs=1, input_pipelines=input_pipelines)
    else:
        estimator = helper_functions.safe_init_component_with_njobs_1(estimator_class)

    estimator.fit(X, y)
    pd.testing.assert_index_equal(X.index, X_original_index)
    pd.testing.assert_index_equal(y.index, y_original_index)


def test_component_modifies_feature_or_target():
    for component_class in all_components():
        if (
            issubclass(component_class, BaseSampler)
            or hasattr(component_class, "inverse_transform")
            or component_class in [TargetImputer, DropRowsTransformer]
        ):
            assert component_class.modifies_target
        else:
            assert not component_class.modifies_target
        if hasattr(component_class, "inverse_transform") or component_class in [
            TargetImputer
        ]:
            assert not component_class.modifies_features
        else:
            assert component_class.modifies_features


def test_component_parameters_supported_by_list_API():
    for component_class in all_components():
        if (
            issubclass(component_class, BaseSampler)
            or hasattr(component_class, "inverse_transform")
            or component_class in [TargetImputer, DropRowsTransformer]
        ):
            assert not component_class._supported_by_list_API
        else:
            assert component_class._supported_by_list_API
