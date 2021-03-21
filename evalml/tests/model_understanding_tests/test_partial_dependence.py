import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from skopt.space import Real

from evalml.demos import load_breast_cancer, load_fraud, load_wine
from evalml.exceptions import NullsInColumnWarning
from evalml.model_family import ModelFamily
from evalml.model_understanding import (
    graph_partial_dependence,
    partial_dependence
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    ClassificationPipeline,
    MulticlassClassificationPipeline,
    RegressionPipeline
)
from evalml.problem_types import ProblemTypes


@pytest.fixture
def test_pipeline():
    class TestPipeline(BinaryClassificationPipeline):
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']

        hyperparameters = {
            "penalty": ["l2"],
            "C": Real(.01, 10),
            "impute_strategy": ["mean", "median", "most_frequent"],
        }

        def __init__(self, parameters):
            super().__init__(parameters=parameters)

        @property
        def feature_importance(self):
            importance = [1.0, 0.2, 0.0002, 0.0, 0.0, -1.0]
            feature_names = range(len(importance))
            f_i = list(zip(feature_names, importance))
            df = pd.DataFrame(f_i, columns=["feature", "importance"])
            return df

    return TestPipeline(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})


def check_partial_dependence_dataframe(pipeline, part_dep, grid_size=20):
    columns = ["feature_values", "partial_dependence"]
    if isinstance(pipeline, ClassificationPipeline):
        columns.append("class_label")
    n_rows_for_class = len(pipeline.classes_) if isinstance(pipeline, MulticlassClassificationPipeline) else 1
    assert list(part_dep.columns) == columns
    assert len(part_dep["partial_dependence"]) == grid_size * n_rows_for_class
    assert len(part_dep["feature_values"]) == grid_size * n_rows_for_class
    if isinstance(pipeline, ClassificationPipeline):
        per_class_counts = part_dep['class_label'].value_counts()
        assert all(value == grid_size for value in per_class_counts.values)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
def test_partial_dependence_problem_types(data_type, problem_type, X_y_binary, X_y_multi, X_y_regression,
                                          logistic_regression_binary_pipeline_class,
                                          logistic_regression_multiclass_pipeline_class,
                                          linear_regression_pipeline_class, make_data_type):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})

    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        pipeline = logistic_regression_multiclass_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})

    elif problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression
        pipeline = linear_regression_pipeline_class(parameters={"Linear Regressor": {"n_jobs": 1}})

    X = make_data_type(data_type, X)
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features=0, grid_resolution=20)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().any(axis=None)


def test_partial_dependence_string_feature_name(logistic_regression_binary_pipeline_class):
    X, y = load_breast_cancer()
    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features="mean radius", grid_resolution=20)
    assert list(part_dep.columns) == ["feature_values", "partial_dependence", "class_label"]
    assert len(part_dep["partial_dependence"]) == 20
    assert len(part_dep["feature_values"]) == 20
    assert not part_dep.isnull().any(axis=None)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_partial_dependence_with_non_numeric_columns(data_type, linear_regression_pipeline_class, logistic_regression_binary_pipeline_class):
    X = pd.DataFrame({'numeric': [1, 2, 3, 0],
                      'also numeric': [2, 3, 4, 1],
                      'string': ['a', 'b', 'a', 'c'],
                      'also string': ['c', 'b', 'a', 'd']})
    if data_type == "ww":
        X = ww.DataTable(X)
    y = [0, 0.2, 1.4, 1]
    pipeline = linear_regression_pipeline_class(parameters={"Linear Regressor": {"n_jobs": 1}})
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features='numeric')
    assert list(part_dep.columns) == ["feature_values", "partial_dependence"]
    assert len(part_dep["partial_dependence"]) == 4
    assert len(part_dep["feature_values"]) == 4
    assert not part_dep.isnull().any(axis=None)

    part_dep = partial_dependence(pipeline, X, features='string')
    assert list(part_dep.columns) == ["feature_values", "partial_dependence"]
    assert len(part_dep["partial_dependence"]) == 3
    assert len(part_dep["feature_values"]) == 3
    assert not part_dep.isnull().any(axis=None)


def test_partial_dependence_baseline():
    X = pd.DataFrame([[1, 0], [0, 1]])
    y = pd.Series([0, 1])

    class BaselineTestPipeline(BinaryClassificationPipeline):
        component_graph = ["Baseline Classifier"]
    pipeline = BaselineTestPipeline({})
    pipeline.fit(X, y)
    with pytest.raises(ValueError, match="Partial dependence plots are not supported for Baseline pipelines"):
        partial_dependence(pipeline, X, features=0, grid_resolution=20)


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_partial_dependence_catboost(problem_type, X_y_binary, X_y_multi, has_minimal_dependencies):
    if not has_minimal_dependencies:

        if problem_type == ProblemTypes.BINARY:
            X, y = X_y_binary
            y_small = ['a', 'b', 'a']

            class CatBoostTestPipeline(BinaryClassificationPipeline):
                component_graph = ["CatBoost Classifier"]
        else:
            X, y = X_y_multi
            y_small = ['a', 'b', 'c']

            class CatBoostTestPipeline(MulticlassClassificationPipeline):
                component_graph = ["CatBoost Classifier"]

        pipeline = CatBoostTestPipeline({"CatBoost Classifier": {'thread_count': 1}})
        pipeline.fit(X, y)
        part_dep = partial_dependence(pipeline, X, features=0, grid_resolution=20)
        check_partial_dependence_dataframe(pipeline, part_dep)
        assert not part_dep.isnull().all().all()

        # test that CatBoost can natively handle non-numerical columns as feature passed to partial_dependence
        X = pd.DataFrame({'numeric': [1, 2, 3], 'also numeric': [2, 3, 4], 'string': ['a', 'b', 'c'], 'also string': ['c', 'b', 'a']})
        pipeline = CatBoostTestPipeline({"CatBoost Classifier": {'thread_count': 1}})
        pipeline.fit(X, y_small)
        part_dep = partial_dependence(pipeline, X, features='string')
        check_partial_dependence_dataframe(pipeline, part_dep, grid_size=3)
        assert not part_dep.isnull().all().all()


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
def test_partial_dependence_xgboost_feature_names(problem_type, has_minimal_dependencies,
                                                  X_y_binary, X_y_multi, X_y_regression):
    if has_minimal_dependencies:
        pytest.skip("Skipping because XGBoost not installed for minimal dependencies")
    if problem_type == ProblemTypes.REGRESSION:
        class XGBoostPipeline(RegressionPipeline):
            component_graph = ['Simple Imputer', 'XGBoost Regressor']
            model_family = ModelFamily.XGBOOST
        X, y = X_y_regression
    elif problem_type == ProblemTypes.BINARY:
        class XGBoostPipeline(BinaryClassificationPipeline):
            component_graph = ['Simple Imputer', 'XGBoost Classifier']
            model_family = ModelFamily.XGBOOST
        X, y = X_y_binary
    elif problem_type == ProblemTypes.MULTICLASS:
        class XGBoostPipeline(MulticlassClassificationPipeline):
            component_graph = ['Simple Imputer', 'XGBoost Classifier']
            model_family = ModelFamily.XGBOOST
        X, y = X_y_multi

    X = pd.DataFrame(X)
    X = X.rename(columns={0: '<[0]'})
    pipeline = XGBoostPipeline({'XGBoost Classifier': {'nthread': 1}})
    pipeline.fit(X, y)
    part_dep = partial_dependence(pipeline, X, features="<[0]", grid_resolution=20)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().all().all()

    part_dep = partial_dependence(pipeline, X, features=1, grid_resolution=20)
    check_partial_dependence_dataframe(pipeline, part_dep)
    assert not part_dep.isnull().all().all()


def test_partial_dependence_multiclass(logistic_regression_multiclass_pipeline_class):
    X, y = load_wine()
    pipeline = logistic_regression_multiclass_pipeline_class(
        parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    pipeline.fit(X, y)

    num_classes = y.to_series().nunique()
    grid_resolution = 20

    one_way_part_dep = partial_dependence(pipeline=pipeline,
                                          X=X,
                                          features="magnesium",
                                          grid_resolution=grid_resolution)
    assert "class_label" in one_way_part_dep.columns
    assert one_way_part_dep["class_label"].nunique() == num_classes
    assert len(one_way_part_dep.index) == num_classes * grid_resolution
    assert list(one_way_part_dep.columns) == ["feature_values", "partial_dependence", "class_label"]

    two_way_part_dep = partial_dependence(pipeline=pipeline,
                                          X=X,
                                          features=("magnesium", "alcohol"),
                                          grid_resolution=grid_resolution)

    assert "class_label" in two_way_part_dep.columns
    assert two_way_part_dep["class_label"].nunique() == num_classes
    assert len(two_way_part_dep.index) == num_classes * grid_resolution
    assert len(two_way_part_dep.columns) == grid_resolution + 1


def test_partial_dependence_not_fitted(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    with pytest.raises(ValueError, match="Pipeline to calculate partial dependence for must be fitted"):
        partial_dependence(pipeline, X, features=0, grid_resolution=20)


def test_partial_dependence_warning(logistic_regression_binary_pipeline_class):
    X = pd.DataFrame({'a': [1, 2, None, 2, 2], 'b': [1, 1, 2, 2, 1]})
    y = pd.Series([0, 1, 0, 1, 0])
    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    pipeline.fit(X, y)
    with pytest.warns(NullsInColumnWarning, match="There are null values in the features, which will cause NaN values in the partial dependence output"):
        partial_dependence(pipeline, X, features=0, grid_resolution=20)

    with pytest.warns(NullsInColumnWarning, match="There are null values in the features, which will cause NaN values in the partial dependence output"):
        partial_dependence(pipeline, X, features='a', grid_resolution=20)


def test_partial_dependence_errors(logistic_regression_binary_pipeline_class):
    X = pd.DataFrame({'a': [2, None, 2, 2], 'b': [1, 2, 2, 1], 'c': [0, 0, 0, 0]})
    y = pd.Series([0, 1, 0, 1])
    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    pipeline.fit(X, y)

    with pytest.raises(ValueError, match="Too many features given to graph_partial_dependence.  Only one or two-way partial dependence is supported."):
        partial_dependence(pipeline, X, features=('a', 'b', 'c'), grid_resolution=20)

    with pytest.raises(ValueError, match="Features provided must be a tuple entirely of integers or strings, not a mixture of both."):
        partial_dependence(pipeline, X, features=(0, 'b'))


def test_partial_dependence_more_categories_than_grid_resolution(logistic_regression_binary_pipeline_class):
    def round_dict_keys(dictionary, places=6):
        """ Function to round all keys of a dictionary that has floats as keys. """
        dictionary_rounded = {}
        for key in dictionary:
            dictionary_rounded[round(key, places)] = dictionary[key]
        return dictionary_rounded

    X, y = load_fraud(1000)
    X = X.drop(columns=['datetime', 'expiration_date', 'country', 'region', 'provider'])
    pipeline = logistic_regression_binary_pipeline_class({})
    pipeline.fit(X, y)
    num_cat_features = len(set(X["currency"].to_series()))
    assert num_cat_features == 164

    part_dep_ans = {0.1432616813857269: 154, 0.1502346349971562: 1, 0.14487916687594762: 1,
                    0.1573183451314127: 1, 0.11695462432136654: 1, 0.07950579532536253: 1, 0.006794444792966759: 1,
                    0.17745270478939879: 1, 0.1666874487986626: 1, 0.13357573073236878: 1, 0.06778096366056789: 1}
    part_dep_ans_rounded = round_dict_keys(part_dep_ans)

    # Check the case where grid_resolution < number of categorical features
    part_dep = partial_dependence(pipeline, X, 'currency', grid_resolution=round(num_cat_features / 2))
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)

    # Check the case where grid_resolution == number of categorical features
    part_dep = partial_dependence(pipeline, X, 'currency', grid_resolution=round(num_cat_features))
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)

    # Check the case where grid_resolution > number of categorical features
    part_dep = partial_dependence(pipeline, X, 'currency', grid_resolution=round(num_cat_features * 2))
    part_dep_dict = dict(part_dep["partial_dependence"].value_counts())
    assert part_dep_ans_rounded == round_dict_keys(part_dep_dict)


def test_graph_partial_dependence(test_pipeline):
    X, y = load_breast_cancer()

    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_partial_dependence(clf, X, features='mean radius', grid_resolution=20)
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == "Partial Dependence of 'mean radius'"
    assert len(fig_dict['data']) == 1
    assert fig_dict['data'][0]['name'] == "Partial Dependence"

    part_dep_data = partial_dependence(clf, X, features='mean radius', grid_resolution=20)
    assert np.array_equal(fig_dict['data'][0]['x'], part_dep_data['feature_values'])
    assert np.array_equal(fig_dict['data'][0]['y'], part_dep_data['partial_dependence'].values)


def test_graph_two_way_partial_dependence(test_pipeline):
    X, y = load_breast_cancer()

    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    clf = test_pipeline
    clf.fit(X, y)
    fig = graph_partial_dependence(clf, X, features=('mean radius', 'mean area'), grid_resolution=20)
    assert isinstance(fig, go.Figure)
    fig_dict = fig.to_dict()
    assert fig_dict['layout']['title']['text'] == "Partial Dependence of 'mean radius' vs. 'mean area'"
    assert len(fig_dict['data']) == 1
    assert fig_dict['data'][0]['name'] == "Partial Dependence"

    part_dep_data = partial_dependence(clf, X, features=('mean radius', 'mean area'), grid_resolution=20)
    assert np.array_equal(fig_dict['data'][0]['x'], part_dep_data.index)
    assert np.array_equal(fig_dict['data'][0]['y'], part_dep_data.columns)
    assert np.array_equal(fig_dict['data'][0]['z'], part_dep_data.values)


def test_graph_partial_dependence_multiclass(logistic_regression_multiclass_pipeline_class):
    go = pytest.importorskip('plotly.graph_objects', reason='Skipping plotting test because plotly not installed')
    X, y = load_wine()
    pipeline = logistic_regression_multiclass_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    pipeline.fit(X, y)

    # Test one-way without class labels
    fig_one_way_no_class_labels = graph_partial_dependence(pipeline, X, features='magnesium', grid_resolution=20)
    assert isinstance(fig_one_way_no_class_labels, go.Figure)
    fig_dict = fig_one_way_no_class_labels.to_dict()
    assert len(fig_dict['data']) == len(pipeline.classes_)
    for data, label in zip(fig_dict['data'], pipeline.classes_):
        assert len(data['x']) == 20
        assert len(data['y']) == 20
        assert data['name'] == label

    # Check that all the subplots axes have the same range
    for suplot_1_axis, suplot_2_axis in [('axis2', 'axis3'), ('axis2', 'axis4'), ('axis3', 'axis4')]:
        for axis_type in ['x', 'y']:
            assert fig_dict['layout'][axis_type + suplot_1_axis]['range'] == fig_dict['layout'][axis_type + suplot_2_axis]['range']

    # Test one-way with class labels
    fig_one_way_class_labels = graph_partial_dependence(pipeline, X, features='magnesium', class_label='class_1', grid_resolution=20)
    assert isinstance(fig_one_way_class_labels, go.Figure)
    fig_dict = fig_one_way_class_labels.to_dict()
    assert len(fig_dict['data']) == 1
    assert len(fig_dict['data'][0]['x']) == 20
    assert len(fig_dict['data'][0]['y']) == 20
    assert fig_dict['data'][0]['name'] == 'class_1'

    msg = "Class wine is not one of the classes the pipeline was fit on: class_0, class_1, class_2"
    with pytest.raises(ValueError, match=msg):
        graph_partial_dependence(pipeline, X, features='alcohol', class_label='wine')

    # Test two-way without class labels
    fig_two_way_no_class_labels = graph_partial_dependence(pipeline, X, features=('magnesium', 'alcohol'), grid_resolution=20)
    assert isinstance(fig_two_way_no_class_labels, go.Figure)
    fig_dict = fig_two_way_no_class_labels.to_dict()
    assert len(fig_dict['data']) == 3, "Figure does not have partial dependence data for each class."
    assert all([len(fig_dict["data"][i]['x']) == 20 for i in range(3)])
    assert all([len(fig_dict["data"][i]['y']) == 20 for i in range(3)])
    assert [fig_dict["data"][i]['name'] for i in range(3)] == ["class_0", "class_1", "class_2"]

    # Check that all the subplots axes have the same range
    for suplot_1_axis, suplot_2_axis in [('axis2', 'axis3'), ('axis2', 'axis4'), ('axis3', 'axis4')]:
        for axis_type in ['x', 'y']:
            assert fig_dict['layout'][axis_type + suplot_1_axis]['range'] == fig_dict['layout'][axis_type + suplot_2_axis]['range']

    # Test two-way with class labels
    fig_two_way_class_labels = graph_partial_dependence(pipeline, X, features=('magnesium', 'alcohol'), class_label='class_1', grid_resolution=20)
    assert isinstance(fig_two_way_class_labels, go.Figure)
    fig_dict = fig_two_way_class_labels.to_dict()
    assert len(fig_dict['data']) == 1
    assert len(fig_dict['data'][0]['x']) == 20
    assert len(fig_dict['data'][0]['y']) == 20
    assert fig_dict['data'][0]['name'] == 'class_1'

    msg = "Class wine is not one of the classes the pipeline was fit on: class_0, class_1, class_2"
    with pytest.raises(ValueError, match=msg):
        graph_partial_dependence(pipeline, X, features='alcohol', class_label='wine')


def test_partial_dependence_percentile_errors(logistic_regression_binary_pipeline_class):
    # random_col will be 5% 0, 95% 1
    X = pd.DataFrame({"A": [i % 3 for i in range(1000)], "B": [(j + 3) % 5 for j in range(1000)], "random_col": [0 if i < 50 else 1 for i in range(1000)]})
    y = pd.Series([i % 2 for i in range(1000)])
    pipeline = logistic_regression_binary_pipeline_class(parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
    pipeline.fit(X, y)
    with pytest.raises(ValueError, match="Feature 'random_col' is mostly one value, 1, and cannot be"):
        partial_dependence(pipeline, X, features="random_col", grid_resolution=20)
    with pytest.raises(ValueError, match="Feature 'random_col' is mostly one value, 1, and cannot be"):
        partial_dependence(pipeline, X, features="random_col", percentiles=(0.01, 0.955), grid_resolution=20)
    with pytest.raises(ValueError, match="Feature 'random_col' is mostly one value, 1, and cannot be"):
        partial_dependence(pipeline, X, features=2, percentiles=(0.01, 0.955), grid_resolution=20)

    part_dep = partial_dependence(pipeline, X, features="random_col", percentiles=(0.01, 0.96), grid_resolution=20)
    assert list(part_dep.columns) == ["feature_values", "partial_dependence", "class_label"]
    assert len(part_dep["partial_dependence"]) == 2
    assert len(part_dep["feature_values"]) == 2
    assert not part_dep.isnull().any(axis=None)
