from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal
)

from evalml.exceptions import MissingComponentError
from evalml.pipelines import ComponentGraph
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    ElasticNetClassifier,
    Estimator,
    Imputer,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    StandardScaler,
    TargetImputer,
    Transformer,
    Undersampler
)
from evalml.utils import infer_feature_types


class DummyTransformer(Transformer):
    name = "Dummy Transformer"

    def __init__(self, parameters={}, random_seed=0):
        super().__init__(parameters=parameters, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        return self


class TransformerA(DummyTransformer):
    """copy class"""


class TransformerB(DummyTransformer):
    """copy class"""


class TransformerC(DummyTransformer):
    """copy class"""


class DummyEstimator(Estimator):
    name = "Dummy Estimator"
    model_family = None
    supported_problem_types = None

    def __init__(self, parameters={}, random_seed=0):
        super().__init__(parameters=parameters, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        return self


class EstimatorA(DummyEstimator):
    """copy class"""


class EstimatorB(DummyEstimator):
    """copy class"""


class EstimatorC(DummyEstimator):
    """copy class"""


@pytest.fixture
def dummy_components():
    return TransformerA, TransformerB, TransformerC, EstimatorA, EstimatorB, EstimatorC


@pytest.fixture
def example_graph():
    graph = {'Imputer': [Imputer],
             'OneHot_RandomForest': [OneHotEncoder, 'Imputer.x'],
             'OneHot_ElasticNet': [OneHotEncoder, 'Imputer.x'],
             'Random Forest': [RandomForestClassifier, 'OneHot_RandomForest.x'],
             'Elastic Net': [ElasticNetClassifier, 'OneHot_ElasticNet.x'],
             'Logistic Regression': [LogisticRegressionClassifier, 'Random Forest', 'Elastic Net']}
    return graph


def test_init(example_graph):
    comp_graph = ComponentGraph()
    assert len(comp_graph.component_dict) == 0

    graph = example_graph
    comp_graph = ComponentGraph(graph)
    assert len(comp_graph.component_dict) == 6

    expected_order = ['Imputer', 'OneHot_ElasticNet', 'Elastic Net', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
    assert comp_graph.compute_order == expected_order


def test_init_str_components():
    graph = {'Imputer': ['Imputer'],
             'OneHot_RandomForest': ['One Hot Encoder', 'Imputer.x'],
             'OneHot_ElasticNet': ['One Hot Encoder', 'Imputer.x'],
             'Random Forest': ['Random Forest Classifier', 'OneHot_RandomForest.x'],
             'Elastic Net': ['Elastic Net Classifier', 'OneHot_ElasticNet.x'],
             'Logistic Regression': ['Logistic Regression Classifier', 'Random Forest', 'Elastic Net']}
    comp_graph = ComponentGraph(graph)
    assert len(comp_graph.component_dict) == 6

    expected_order = ['Imputer', 'OneHot_ElasticNet', 'Elastic Net', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
    assert comp_graph.compute_order == expected_order


def test_invalid_init():
    invalid_graph = {'Imputer': [Imputer], 'OHE': OneHotEncoder}
    with pytest.raises(ValueError, match='All component information should be passed in as a list'):
        ComponentGraph(invalid_graph)

    with pytest.raises(ValueError, match='may only contain str or ComponentBase subclasses'):
        ComponentGraph({'Imputer': [Imputer(numeric_impute_strategy="most_frequent")], 'OneHot': [OneHotEncoder]})

    graph = {'Imputer': [Imputer(numeric_impute_strategy='constant', numeric_fill_value=0)]}
    with pytest.raises(ValueError, match='may only contain str or ComponentBase subclasses'):
        ComponentGraph(graph)

    graph = {'Imputer': ['Imputer', 'Fake'],
             'Fake': ['Fake Component', 'Estimator'],
             'Estimator': [ElasticNetClassifier]}
    with pytest.raises(MissingComponentError):
        ComponentGraph(graph)


def test_init_bad_graphs():
    graph = {'Imputer': [Imputer],
             'OHE': [OneHotEncoder, 'Imputer.x', 'Estimator'],
             'Estimator': [RandomForestClassifier, 'OHE.x']}
    with pytest.raises(ValueError, match='given graph contains a cycle'):
        ComponentGraph(graph)

    graph = {'Imputer': [Imputer],
             'OneHot_RandomForest': [OneHotEncoder, 'Imputer.x'],
             'OneHot_ElasticNet': [OneHotEncoder, 'Imputer.x'],
             'Random Forest': [RandomForestClassifier],
             'Elastic Net': [ElasticNetClassifier],
             'Logistic Regression': [LogisticRegressionClassifier, 'Random Forest', 'Elastic Net']}
    with pytest.raises(ValueError, match='graph is not completely connected'):
        ComponentGraph(graph)

    graph = {'Imputer': ['Imputer'],
             'OneHot_RandomForest': ['One Hot Encoder', 'Imputer.x'],
             'OneHot_ElasticNet': ['One Hot Encoder', 'Imputer.x'],
             'Random Forest': ['Random Forest Classifier', 'OneHot_RandomForest.x'],
             'Elastic Net': ['Elastic Net Classifier'],
             'Logistic Regression': ['Logistic Regression Classifier', 'Random Forest', 'Elastic Net']}
    with pytest.raises(ValueError, match='graph has more than one final'):
        ComponentGraph(graph)


def test_order_x_and_y():
    graph = {'Imputer': [Imputer],
             'OHE': [OneHotEncoder, 'Imputer.x', 'Imputer.y'],
             'Random Forest': [RandomForestClassifier, 'OHE.x']}
    component_graph = ComponentGraph(graph).instantiate({})
    assert component_graph.compute_order == ['Imputer', 'OHE', 'Random Forest']


def test_from_list():
    component_list = ['Imputer', 'One Hot Encoder', RandomForestClassifier]

    component_graph = ComponentGraph.from_list(component_list)

    assert len(component_graph.component_dict) == 3
    assert component_graph.get_component('Imputer') == Imputer
    assert component_graph.get_component('One Hot Encoder') == OneHotEncoder
    assert component_graph.get_component('Random Forest Classifier') == RandomForestClassifier

    expected_order = ['Imputer', 'One Hot Encoder', 'Random Forest Classifier']
    assert component_graph.compute_order == expected_order
    assert component_graph.component_dict == {
        'Imputer': [Imputer],
        'One Hot Encoder': [OneHotEncoder, 'Imputer.x'],
        'Random Forest Classifier': [RandomForestClassifier, 'One Hot Encoder.x']
    }

    bad_component_list = ['Imputer', 'Fake Estimator']
    with pytest.raises(MissingComponentError, match='was not found'):
        ComponentGraph.from_list(bad_component_list)


def test_from_list_repeat_component():
    component_list = ['Imputer', 'One Hot Encoder', 'One Hot Encoder', RandomForestClassifier]
    component_graph = ComponentGraph.from_list(component_list)

    expected_order = ['Imputer', 'One Hot Encoder', 'One Hot Encoder_2', 'Random Forest Classifier']
    assert component_graph.compute_order == expected_order

    component_graph.instantiate({'One Hot Encoder': {'top_n': 2},
                                 'One Hot Encoder_2': {'top_n': 11}})
    assert component_graph.get_component('One Hot Encoder').parameters['top_n'] == 2
    assert component_graph.get_component('One Hot Encoder_2').parameters['top_n'] == 11


def test_instantiate_with_parameters(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert not isinstance(component_graph.get_component('Imputer'), Imputer)
    assert not isinstance(component_graph.get_component('Elastic Net'), ElasticNetClassifier)

    parameters = {'OneHot_RandomForest': {'top_n': 3},
                  'OneHot_ElasticNet': {'top_n': 5},
                  'Elastic Net': {'max_iter': 100}}
    component_graph.instantiate(parameters)

    expected_order = ['Imputer', 'OneHot_ElasticNet', 'Elastic Net', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
    assert component_graph.compute_order == expected_order

    assert isinstance(component_graph.get_component('Imputer'), Imputer)
    assert isinstance(component_graph.get_component('Random Forest'), RandomForestClassifier)
    assert isinstance(component_graph.get_component('Logistic Regression'), LogisticRegressionClassifier)
    assert component_graph.get_component('OneHot_RandomForest').parameters['top_n'] == 3
    assert component_graph.get_component('OneHot_ElasticNet').parameters['top_n'] == 5
    assert component_graph.get_component('Elastic Net').parameters['max_iter'] == 100


def test_instantiate_without_parameters(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert component_graph.get_component('OneHot_RandomForest').parameters['top_n'] == 10
    assert component_graph.get_component('OneHot_ElasticNet').parameters['top_n'] == 10
    assert component_graph.get_component('OneHot_RandomForest') is not component_graph.get_component('OneHot_ElasticNet')

    expected_order = ['Imputer', 'OneHot_ElasticNet', 'Elastic Net', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
    assert component_graph.compute_order == expected_order


def test_instantiate_from_list():
    component_list = ['Imputer', 'One Hot Encoder', 'Random Forest Classifier']
    component_graph = ComponentGraph().from_list(component_list)

    parameters = {'One Hot Encoder': {'top_n': 7}}
    component_graph.instantiate(parameters)
    assert isinstance(component_graph.get_component('Imputer'), Imputer)
    assert isinstance(component_graph.get_component('Random Forest Classifier'), RandomForestClassifier)
    assert component_graph.get_component('One Hot Encoder').parameters['top_n'] == 7


def test_reinstantiate(example_graph):
    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate({})
    with pytest.raises(ValueError, match='Cannot reinstantiate a component graph'):
        component_graph.instantiate({'OneHot': {'top_n': 7}})


def test_bad_instantiate_can_reinstantiate(example_graph):
    component_graph = ComponentGraph(example_graph)
    with pytest.raises(ValueError, match='Error received when instantiating component'):
        component_graph.instantiate(parameters={'Elastic Net': {'max_iter': 100, 'fake_param': None}})

    component_graph.instantiate({'Elastic Net': {'max_iter': 22}})
    assert component_graph.get_component('Elastic Net').parameters['max_iter'] == 22


def test_get_component(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert component_graph.get_component('OneHot_ElasticNet') == OneHotEncoder
    assert component_graph.get_component('Logistic Regression') == LogisticRegressionClassifier

    with pytest.raises(ValueError, match='not in the graph'):
        component_graph.get_component('Fake Component')

    component_graph.instantiate({'OneHot_RandomForest': {'top_n': 3},
                                 'Random Forest': {'max_depth': 4, 'n_estimators': 50}})
    assert component_graph.get_component('OneHot_ElasticNet') == OneHotEncoder()
    assert component_graph.get_component('OneHot_RandomForest') == OneHotEncoder(top_n=3)
    assert component_graph.get_component('Random Forest') == RandomForestClassifier(n_estimators=50, max_depth=4)


def test_get_estimators(example_graph):
    component_graph = ComponentGraph(example_graph)
    with pytest.raises(ValueError, match='Cannot get estimators until'):
        component_graph.get_estimators()

    component_graph.instantiate({})
    assert component_graph.get_estimators() == [RandomForestClassifier(), ElasticNetClassifier(), LogisticRegressionClassifier()]

    component_graph = ComponentGraph.from_list(['Imputer', 'One Hot Encoder'])
    component_graph.instantiate({})
    assert component_graph.get_estimators() == []


def test_parents(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert component_graph.get_parents('Imputer') == []
    assert component_graph.get_parents('OneHot_RandomForest') == ['Imputer.x']
    assert component_graph.get_parents('OneHot_ElasticNet') == ['Imputer.x']
    assert component_graph.get_parents('Random Forest') == ['OneHot_RandomForest.x']
    assert component_graph.get_parents('Elastic Net') == ['OneHot_ElasticNet.x']
    assert component_graph.get_parents('Logistic Regression') == ['Random Forest', 'Elastic Net']

    with pytest.raises(ValueError, match='not in the graph'):
        component_graph.get_parents('Fake component')

    component_graph.instantiate({})
    assert component_graph.get_parents('Imputer') == []
    assert component_graph.get_parents('OneHot_RandomForest') == ['Imputer.x']
    assert component_graph.get_parents('OneHot_ElasticNet') == ['Imputer.x']
    assert component_graph.get_parents('Random Forest') == ['OneHot_RandomForest.x']
    assert component_graph.get_parents('Elastic Net') == ['OneHot_ElasticNet.x']
    assert component_graph.get_parents('Logistic Regression') == ['Random Forest', 'Elastic Net']

    with pytest.raises(ValueError, match='not in the graph'):
        component_graph.get_parents('Fake component')


def test_get_last_component(example_graph):
    component_graph = ComponentGraph()
    with pytest.raises(ValueError, match='Cannot get last component from edgeless graph'):
        component_graph.get_last_component()

    component_graph = ComponentGraph(example_graph)
    assert component_graph.get_last_component() == LogisticRegressionClassifier

    component_graph.instantiate({})
    assert component_graph.get_last_component() == LogisticRegressionClassifier()

    component_graph = ComponentGraph({'Imputer': [Imputer]})
    assert component_graph.get_last_component() == Imputer

    component_graph = ComponentGraph({'Imputer': [Imputer], 'OneHot': [OneHotEncoder, 'Imputer']})
    assert component_graph.get_last_component() == OneHotEncoder

    component_graph = ComponentGraph({'Imputer': [Imputer], 'OneHot': [OneHotEncoder]})
    with pytest.raises(ValueError, match='Cannot get last component from edgeless graph'):
        component_graph.get_last_component()


@patch('evalml.pipelines.components.Transformer.fit_transform')
@patch('evalml.pipelines.components.Estimator.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_fit(mock_predict, mock_fit, mock_fit_transform, example_graph, X_y_binary):
    X, y = X_y_binary
    mock_fit_transform.return_value = ww.DataTable(X)
    mock_predict.return_value = ww.DataColumn(y)
    component_graph = ComponentGraph(example_graph).instantiate({})
    component_graph.fit(X, y)

    assert mock_fit_transform.call_count == 3
    assert mock_fit.call_count == 3
    assert mock_predict.call_count == 2


@patch('evalml.pipelines.components.Imputer.fit_transform')
@patch('evalml.pipelines.components.OneHotEncoder.fit_transform')
def test_fit_correct_inputs(mock_ohe_fit_transform, mock_imputer_fit_transform, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    y = pd.Series(y)
    graph = {'Imputer': [Imputer], 'OHE': [OneHotEncoder, 'Imputer.x', 'Imputer.y']}
    expected_x = ww.DataTable(pd.DataFrame(index=X.index, columns=X.index).fillna(1))
    expected_y = ww.DataColumn(pd.Series(index=y.index).fillna(0))
    mock_imputer_fit_transform.return_value = tuple((expected_x, expected_y))
    mock_ohe_fit_transform.return_value = expected_x
    component_graph = ComponentGraph(graph).instantiate({})
    component_graph.fit(X, y)
    expected_x_df = expected_x.to_dataframe().astype("Int64")
    assert_frame_equal(expected_x_df, mock_ohe_fit_transform.call_args[0][0].to_dataframe())
    assert_series_equal(expected_y.to_series(), mock_ohe_fit_transform.call_args[0][1].to_series())


@patch('evalml.pipelines.components.Transformer.fit_transform')
@patch('evalml.pipelines.components.Estimator.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_fit_features(mock_predict, mock_fit, mock_fit_transform, X_y_binary):
    X, y = X_y_binary
    component_list = ['Imputer', 'One Hot Encoder', 'Random Forest Classifier']
    component_graph = ComponentGraph.from_list(component_list)
    component_graph.instantiate({})

    mock_fit_transform.return_value = ww.DataTable(np.ones(X.shape))
    mock_fit.return_value = Estimator
    mock_predict.return_value = ww.DataColumn(y)

    component_graph.fit_features(X, y)

    assert mock_fit_transform.call_count == 2
    assert mock_fit.call_count == 0
    assert mock_predict.call_count == 0


@patch('evalml.pipelines.components.Transformer.fit_transform')
@patch('evalml.pipelines.components.Estimator.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_fit_features_nonlinear(mock_predict, mock_fit, mock_fit_transform, example_graph, X_y_binary):
    X, y = X_y_binary
    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate({})

    mock_X_t = ww.DataTable(np.ones(pd.DataFrame(X).shape))
    mock_fit_transform.return_value = mock_X_t
    mock_fit.return_value = Estimator
    mock_predict.return_value = ww.DataColumn(pd.Series(y))

    component_graph.fit_features(X, y)

    assert mock_fit_transform.call_count == 3
    assert mock_fit.call_count == 2
    assert mock_predict.call_count == 2


@patch('evalml.pipelines.components.Estimator.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_predict(mock_predict, mock_fit, example_graph, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = ww.DataColumn(pd.Series(y))
    component_graph = ComponentGraph(example_graph).instantiate({})
    component_graph.fit(X, y)

    component_graph.predict(X)
    assert mock_predict.call_count == 5  # Called twice when fitting pipeline, thrice when predicting
    assert mock_fit.call_count == 3  # Only called during fit, not predict


@patch('evalml.pipelines.components.Estimator.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_predict_repeat_estimator(mock_predict, mock_fit, X_y_binary):
    X, y = X_y_binary
    mock_predict.return_value = ww.DataColumn(pd.Series(y))

    graph = {'Imputer': [Imputer],
             'OneHot_RandomForest': [OneHotEncoder, 'Imputer.x'],
             'OneHot_Logistic': [OneHotEncoder, 'Imputer.x'],
             'Random Forest': [RandomForestClassifier, 'OneHot_RandomForest.x'],
             'Logistic Regression': [LogisticRegressionClassifier, 'OneHot_Logistic.x'],
             'Final Estimator': [LogisticRegressionClassifier, 'Random Forest', 'Logistic Regression']}
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    component_graph.fit(X, y)

    assert not component_graph.get_component('Logistic Regression')._component_obj == component_graph.get_component('Final Estimator')._component_obj

    component_graph.predict(X)
    assert mock_predict.call_count == 5
    assert mock_fit.call_count == 3


@patch('evalml.pipelines.components.Imputer.transform')
@patch('evalml.pipelines.components.OneHotEncoder.transform')
def test_compute_final_component_features_linear(mock_ohe, mock_imputer, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X_expected = X.fillna(0)
    mock_imputer.return_value = ww.DataTable(X)
    mock_ohe.return_value = ww.DataTable(X_expected)

    component_list = ['Imputer', 'One Hot Encoder', 'Random Forest Classifier']
    component_graph = ComponentGraph().from_list(component_list)
    component_graph.instantiate({})
    component_graph.fit(X, y)

    X_t = component_graph.compute_final_component_features(X)
    assert_frame_equal(X_expected, X_t.to_dataframe())
    assert mock_imputer.call_count == 2
    assert mock_ohe.call_count == 2


@patch('evalml.pipelines.components.Imputer.transform')
@patch('evalml.pipelines.components.OneHotEncoder.transform')
@patch('evalml.pipelines.components.RandomForestClassifier.predict')
@patch('evalml.pipelines.components.ElasticNetClassifier.predict')
def test_compute_final_component_features_nonlinear(mock_en_predict, mock_rf_predict, mock_ohe, mock_imputer, example_graph, X_y_binary):
    X, y = X_y_binary
    mock_imputer.return_value = ww.DataTable(pd.DataFrame(X))
    mock_ohe.return_value = ww.DataTable(pd.DataFrame(X))
    mock_en_predict.return_value = ww.DataColumn(pd.Series(np.ones(X.shape[0])))
    mock_rf_predict.return_value = ww.DataColumn(pd.Series(np.zeros(X.shape[0])))
    X_expected = pd.DataFrame({'Random Forest': np.zeros(X.shape[0]), 'Elastic Net': np.ones(X.shape[0])})
    component_graph = ComponentGraph(example_graph).instantiate({})
    component_graph.fit(X, y)

    X_t = component_graph.compute_final_component_features(X)
    assert_frame_equal(X_expected, X_t.to_dataframe())
    assert mock_imputer.call_count == 2
    assert mock_ohe.call_count == 4


@patch(f'{__name__}.DummyTransformer.transform')
def test_compute_final_component_features_single_component(mock_transform, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    mock_transform.return_value = ww.DataTable(X)
    component_graph = ComponentGraph({'Dummy Component': [DummyTransformer]}).instantiate({})
    component_graph.fit(X, y)

    X_t = component_graph.compute_final_component_features(X)
    assert_frame_equal(X, X_t.to_dataframe())


@patch('evalml.pipelines.components.Imputer.fit_transform')
def test_fit_y_parent(mock_fit_transform, X_y_binary):
    X, y = X_y_binary
    graph = {'Imputer': [Imputer],
             'OHE': [OneHotEncoder, 'Imputer.x', 'Imputer.y'],
             'Random Forest': [RandomForestClassifier, 'OHE.x']}
    component_graph = ComponentGraph(graph).instantiate({})
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))

    component_graph.fit(X, y)
    mock_fit_transform.assert_called_once()


def test_predict_empty_graph(X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    component_graph = ComponentGraph()
    component_graph.instantiate({})

    component_graph.fit(X, y)
    X_t = component_graph.predict(X)
    assert_frame_equal(X, X_t.to_dataframe())


@patch('evalml.pipelines.components.OneHotEncoder.fit_transform')
@patch('evalml.pipelines.components.OneHotEncoder.transform')
def test_predict_transformer_end(mock_fit_transform, mock_transform, X_y_binary):
    X, y = X_y_binary
    graph = {'Imputer': [Imputer], 'OHE': [OneHotEncoder, 'Imputer.x']}
    component_graph = ComponentGraph(graph).instantiate({})
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))
    mock_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))

    component_graph.fit(X, y)
    output = component_graph.predict(X)
    assert_frame_equal(pd.DataFrame(X), output.to_dataframe())


def test_no_instantiate_before_fit(X_y_binary):
    X, y = X_y_binary
    graph = {'Imputer': [Imputer],
             'OHE': [OneHotEncoder, 'Imputer.x'],
             'Estimator': [RandomForestClassifier, 'OHE.x']}
    component_graph = ComponentGraph(graph)
    with pytest.raises(ValueError, match='All components must be instantiated before fitting or predicting'):
        component_graph.fit(X, y)


@patch('evalml.pipelines.components.Imputer.fit_transform')
def test_multiple_y_parents(mock_fit_transform, X_y_binary):
    X, y = X_y_binary
    graph = {'Imputer': [Imputer],
             'OHE': [OneHotEncoder, 'Imputer.x'],
             'Estimator': [RandomForestClassifier, 'Imputer.y', 'OHE.y']}
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))
    with pytest.raises(ValueError, match='Cannot have multiple `y` parents for a single component'):
        component_graph.fit(X, y)


def test_component_graph_order(example_graph):
    component_graph = ComponentGraph(example_graph)
    expected_order = ['Imputer', 'OneHot_ElasticNet', 'Elastic Net', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
    assert expected_order == component_graph.compute_order

    component_graph = ComponentGraph({'Imputer': [Imputer]})
    expected_order = ['Imputer']
    assert expected_order == component_graph.compute_order


@pytest.mark.parametrize("index", [list(range(-5, 0)),
                                   list(range(100, 105)),
                                   [f"row_{i}" for i in range(5)],
                                   pd.date_range("2020-09-08", periods=5)])
def test_computation_input_custom_index(index):
    graph = {'OneHot': [OneHotEncoder],
             'Random Forest': [RandomForestClassifier, 'OneHot.x'],
             'Elastic Net': [ElasticNetClassifier, 'OneHot.x'],
             'Logistic Regression': [LogisticRegressionClassifier, 'Random Forest', 'Elastic Net']}

    X = pd.DataFrame({"categories": [f"cat_{i}" for i in range(5)], "numbers": np.arange(5)},
                     index=index)
    y = pd.Series([1, 2, 1, 2, 1])
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    component_graph.fit(X, y)

    X_t = component_graph.predict(X).to_series()
    assert_index_equal(X_t.index, pd.RangeIndex(start=0, stop=5, step=1))
    assert not X_t.isna().any(axis=None)


@patch(f'{__name__}.EstimatorC.predict')
@patch(f'{__name__}.EstimatorB.predict')
@patch(f'{__name__}.EstimatorA.predict')
@patch(f'{__name__}.TransformerC.transform')
@patch(f'{__name__}.TransformerB.transform')
@patch(f'{__name__}.TransformerA.transform')
def test_component_graph_evaluation_plumbing(mock_transa, mock_transb, mock_transc, mock_preda, mock_predb, mock_predc, dummy_components):
    TransformerA, TransformerB, TransformerC, EstimatorA, EstimatorB, EstimatorC = dummy_components
    mock_transa.return_value = ww.DataTable(pd.DataFrame({'feature trans': [1, 0, 0, 0, 0, 0], 'feature a': np.ones(6)}))
    mock_transb.return_value = ww.DataTable(pd.DataFrame({'feature b': np.ones(6) * 2}))
    mock_transc.return_value = ww.DataTable(pd.DataFrame({'feature c': np.ones(6) * 3}))
    mock_preda.return_value = ww.DataColumn(pd.Series([0, 0, 0, 1, 0, 0]))
    mock_predb.return_value = ww.DataColumn(pd.Series([0, 0, 0, 0, 1, 0]))
    mock_predc.return_value = ww.DataColumn(pd.Series([0, 0, 0, 0, 0, 1]))
    graph = {
        'transformer a': [TransformerA],
        'transformer b': [TransformerB, 'transformer a'],
        'transformer c': [TransformerC, 'transformer a', 'transformer b'],
        'estimator a': [EstimatorA],
        'estimator b': [EstimatorB, 'transformer a'],
        'estimator c': [EstimatorC, 'transformer a', 'estimator a', 'transformer b', 'estimator b', 'transformer c']
    }
    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    X = pd.DataFrame({'feature1': np.zeros(6), 'feature2': np.zeros(6)})
    y = pd.Series(np.zeros(6))
    component_graph.fit(X, y)
    predict_out = component_graph.predict(X)

    assert_frame_equal(mock_transa.call_args[0][0].to_dataframe(), X)
    assert_frame_equal(mock_transb.call_args[0][0].to_dataframe(), pd.DataFrame({'feature trans': pd.Series([1, 0, 0, 0, 0, 0], dtype="Int64"),
                                                                                 'feature a': np.ones(6)}, columns=['feature trans', 'feature a']))
    assert_frame_equal(mock_transc.call_args[0][0].to_dataframe(), pd.DataFrame({'feature trans': pd.Series([1, 0, 0, 0, 0, 0], dtype="Int64"),
                                                                                 'feature a': np.ones(6),
                                                                                 'feature b': np.ones(6) * 2},
                                                                                columns=['feature trans', 'feature a', 'feature b']))
    assert_frame_equal(mock_preda.call_args[0][0].to_dataframe(), X)
    assert_frame_equal(mock_predb.call_args[0][0].to_dataframe(), pd.DataFrame({'feature trans': pd.Series([1, 0, 0, 0, 0, 0], dtype="Int64"),
                                                                                'feature a': np.ones(6)},
                                                                               columns=['feature trans', 'feature a']))
    assert_frame_equal(mock_predc.call_args[0][0].to_dataframe(), pd.DataFrame({'feature trans': pd.Series([1, 0, 0, 0, 0, 0], dtype="Int64"),
                                                                                'feature a': np.ones(6),
                                                                                'estimator a': pd.Series([0, 0, 0, 1, 0, 0], dtype="Int64"),
                                                                                'feature b': np.ones(6) * 2,
                                                                                'estimator b': pd.Series([0, 0, 0, 0, 1, 0], dtype="Int64"),
                                                                                'feature c': np.ones(6) * 3},
                                                                               columns=['feature trans', 'feature a', 'estimator a', 'feature b', 'estimator b', 'feature c']))
    assert_series_equal(pd.Series([0, 0, 0, 0, 0, 1], dtype="Int64"), predict_out.to_series())


def test_input_feature_names(example_graph):
    X = pd.DataFrame({'column_1': ['a', 'b', 'c', 'd', 'a', 'a', 'b', 'c', 'b'],
                      'column_2': [1, 2, 3, 4, 5, 6, 5, 4, 3]})
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])

    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate({'OneHot_RandomForest': {'top_n': 2},
                                 'OneHot_ElasticNet': {'top_n': 3}})
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names['Imputer'] == ['column_1', 'column_2']
    assert input_feature_names['OneHot_RandomForest'] == ['column_1', 'column_2']
    assert input_feature_names['OneHot_ElasticNet'] == ['column_1', 'column_2']
    assert input_feature_names['Random Forest'] == ['column_2', 'column_1_a', 'column_1_b']
    assert input_feature_names['Elastic Net'] == ['column_2', 'column_1_a', 'column_1_b', 'column_1_c']
    assert input_feature_names['Logistic Regression'] == ['Random Forest', 'Elastic Net']


def test_iteration(example_graph):
    component_graph = ComponentGraph(example_graph)

    expected = [Imputer, OneHotEncoder, ElasticNetClassifier, OneHotEncoder, RandomForestClassifier, LogisticRegressionClassifier]
    iteration = [component for component in component_graph]
    assert iteration == expected

    component_graph.instantiate({'OneHot_RandomForest': {'top_n': 32}})
    expected = [Imputer(), OneHotEncoder(), ElasticNetClassifier(), OneHotEncoder(top_n=32), RandomForestClassifier(), LogisticRegressionClassifier()]
    iteration = [component for component in component_graph]
    assert iteration == expected


def test_custom_input_feature_types(example_graph):
    X = pd.DataFrame({'column_1': ['a', 'b', 'c', 'd', 'a', 'a', 'b', 'c', 'b'],
                      'column_2': [1, 2, 3, 4, 5, 6, 5, 4, 3]})
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X = infer_feature_types(X, {"column_2": "categorical"})

    component_graph = ComponentGraph(example_graph)
    component_graph.instantiate({'OneHot_RandomForest': {'top_n': 2},
                                 'OneHot_ElasticNet': {'top_n': 3}})
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names['Imputer'] == ['column_1', 'column_2']
    assert input_feature_names['OneHot_RandomForest'] == ['column_1', 'column_2']
    assert input_feature_names['OneHot_ElasticNet'] == ['column_1', 'column_2']
    assert input_feature_names['Random Forest'] == ['column_1_a', 'column_1_b', 'column_2_4', 'column_2_5']
    assert input_feature_names['Elastic Net'] == ['column_1_a', 'column_1_b', 'column_1_c', 'column_2_3', 'column_2_4', 'column_2_5']
    assert input_feature_names['Logistic Regression'] == ['Random Forest', 'Elastic Net']


def test_component_graph_dataset_with_different_types():
    # Checks that types are converted correctly by Woodwork. Specifically, the standard scaler
    # should convert column_3 to float, so our code to try to convert back to the original boolean type
    # will catch the TypeError thrown and not convert the column.
    graph = {'Imputer': [Imputer],
             'OneHot': [OneHotEncoder, 'Imputer.x'],
             'DateTime': [DateTimeFeaturizer, 'OneHot.x'],
             'Scaler': [StandardScaler, 'DateTime.x'],
             'Random Forest': [RandomForestClassifier, 'Scaler.x'],
             'Elastic Net': [ElasticNetClassifier, 'Scaler.x'],
             'Logistic Regression': [LogisticRegressionClassifier, 'Random Forest', 'Elastic Net']}

    X = pd.DataFrame({'column_1': ['a', 'b', 'c', 'd', 'a', 'a', 'b', 'c', 'b'],
                      'column_2': [1, 2, 3, 4, 5, 6, 5, 4, 3],
                      'column_3': [True, False, True, False, True, False, True, False, False]})
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, 0])
    X = infer_feature_types(X, {"column_2": "categorical"})

    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert component_graph.input_feature_names == {}
    component_graph.fit(X, y)

    input_feature_names = component_graph.input_feature_names
    assert input_feature_names['Imputer'] == ['column_1', 'column_2', 'column_3']
    assert input_feature_names['OneHot'] == ['column_1', 'column_2', 'column_3']
    assert input_feature_names['DateTime'] == ['column_3', 'column_1_a', 'column_1_b', 'column_1_c', 'column_1_d',
                                               'column_2_1', 'column_2_2', 'column_2_3', 'column_2_4', 'column_2_5', 'column_2_6']
    assert input_feature_names['Scaler'] == ['column_3', 'column_1_a', 'column_1_b', 'column_1_c', 'column_1_d',
                                             'column_2_1', 'column_2_2', 'column_2_3', 'column_2_4', 'column_2_5', 'column_2_6']
    assert input_feature_names['Random Forest'] == ['column_3', 'column_1_a', 'column_1_b', 'column_1_c', 'column_1_d',
                                                    'column_2_1', 'column_2_2', 'column_2_3', 'column_2_4', 'column_2_5', 'column_2_6']
    assert input_feature_names['Elastic Net'] == ['column_3', 'column_1_a', 'column_1_b', 'column_1_c', 'column_1_d',
                                                  'column_2_1', 'column_2_2', 'column_2_3', 'column_2_4', 'column_2_5', 'column_2_6']
    assert input_feature_names['Logistic Regression'] == ['Random Forest', 'Elastic Net']


def test_component_graph_sampler():
    graph = {'Imputer': [Imputer],
             'OneHot': [OneHotEncoder, 'Imputer.x'],
             'Undersampler': [Undersampler, 'OneHot.x'],
             'Random Forest': [RandomForestClassifier, 'Undersampler.x', 'Undersampler.y'],
             'Elastic Net': [ElasticNetClassifier, 'Undersampler.x', 'Undersampler.y'],
             'Logistic Regression': [LogisticRegressionClassifier, 'Random Forest', 'Elastic Net']}

    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert component_graph.get_parents('Imputer') == []
    assert component_graph.get_parents('OneHot') == ['Imputer.x']
    assert component_graph.get_parents('Undersampler') == ['OneHot.x']
    assert component_graph.get_parents('Random Forest') == ['Undersampler.x', 'Undersampler.y']
    assert component_graph.get_parents('Elastic Net') == ['Undersampler.x', 'Undersampler.y']
    assert component_graph.get_parents('Logistic Regression') == ['Random Forest', 'Elastic Net']


def test_component_graph_sampler_list():
    component_list = ['Imputer', 'One Hot Encoder', 'Undersampler', 'Random Forest Classifier']
    component_graph = ComponentGraph.from_list(component_list)

    assert len(component_graph.component_dict) == 4
    assert component_graph.get_component('Imputer') == Imputer
    assert component_graph.get_component('One Hot Encoder') == OneHotEncoder
    assert component_graph.get_component('Undersampler') == Undersampler
    assert component_graph.get_component('Random Forest Classifier') == RandomForestClassifier

    assert component_graph.compute_order == component_list
    assert component_graph.component_dict == {
        'Imputer': [Imputer],
        'One Hot Encoder': [OneHotEncoder, 'Imputer.x'],
        'Undersampler': [Undersampler, 'One Hot Encoder.x'],
        'Random Forest Classifier': [RandomForestClassifier, 'Undersampler.x', 'Undersampler.y']
    }
    assert component_graph.get_parents('Imputer') == []
    assert component_graph.get_parents('One Hot Encoder') == ['Imputer.x']
    assert component_graph.get_parents('Undersampler') == ['One Hot Encoder.x']
    assert component_graph.get_parents('Random Forest Classifier') == ['Undersampler.x', 'Undersampler.y']


def test_component_graph_dataset_with_target_imputer():
    X = pd.DataFrame({'column_1': ['a', 'b', 'c', 'd', 'a', 'a', 'b', 'c', 'b'],
                      'column_2': [1, 2, 3, 4, 5, 6, 5, 4, 3]})
    y = pd.Series([1, 0, 1, 0, 1, 1, 0, 0, np.nan])
    graph = {'Target Imputer': [TargetImputer],
             'OneHot': [OneHotEncoder, 'Target Imputer.x', 'Target Imputer.y'],
             'Random Forest': [RandomForestClassifier, 'OneHot.x', 'Target Imputer.y'],
             'Elastic Net': [ElasticNetClassifier, 'OneHot.x', 'Target Imputer.y'],
             'Logistic Regression': [LogisticRegressionClassifier, 'Random Forest', 'Elastic Net', 'Target Imputer.y']}

    component_graph = ComponentGraph(graph)
    component_graph.instantiate({})
    assert component_graph.get_parents('Target Imputer') == []
    assert component_graph.get_parents('OneHot') == ['Target Imputer.x', 'Target Imputer.y']
    assert component_graph.get_parents('Random Forest') == ['OneHot.x', 'Target Imputer.y']
    assert component_graph.get_parents('Elastic Net') == ['OneHot.x', 'Target Imputer.y']

    component_graph.fit(X, y)
    predictions = component_graph.predict(X)
    assert not pd.isnull(predictions.to_series()).any()


@patch('evalml.pipelines.components.estimators.LogisticRegressionClassifier.fit')
def test_component_graph_sampler_y_passes(mock_estimator_fit):
    pytest.importorskip("imblearn.over_sampling", reason="Cannot import imblearn, skipping tests")
    # makes sure the y value from oversampler gets passed to the estimator, even though StandardScaler has no y output
    X = pd.DataFrame({"a": [i for i in range(100)],
                      "b": [i % 3 for i in range(100)]})
    y = pd.Series([0] * 90 + [1] * 10)
    component_list = ['Imputer', 'SMOTE Oversampler', 'Standard Scaler', 'Logistic Regression Classifier']
    component_graph = ComponentGraph.from_list(component_list)
    component_graph.instantiate({})
    component_graph.fit(X, y)
    assert len(mock_estimator_fit.call_args[0][0]) == len(mock_estimator_fit.call_args[0][1])
    assert len(mock_estimator_fit.call_args[0][0]) == int(1.25 * 90)


@patch('evalml.pipelines.components.estimators.RandomForestClassifier.fit')
@patch('evalml.pipelines.components.estimators.DecisionTreeClassifier.fit')
def test_component_graph_sampler_same_given_components(mock_dt_fit, mock_rf_fit):
    pytest.importorskip("imblearn.over_sampling", reason="Cannot import imblearn, skipping tests")
    X = pd.DataFrame({"a": [i for i in range(100)],
                      "b": [i % 3 for i in range(100)]})
    y = pd.Series([0] * 90 + [1] * 10)
    component_list = ['Imputer', 'SMOTE Oversampler', 'Random Forest Classifier']
    component_graph = ComponentGraph.from_list(component_list)
    component_graph.instantiate({})
    component_graph.fit(X, y)

    component_list2 = ['Imputer', 'SMOTE Oversampler', 'Decision Tree Classifier']
    component_graph2 = ComponentGraph.from_list(component_list2)
    component_graph2.instantiate({})
    component_graph2.fit(X, y)
    assert mock_dt_fit.call_args[0][0] == mock_rf_fit.call_args[0][0]
    assert mock_dt_fit.call_args[0][1] == mock_rf_fit.call_args[0][1]
