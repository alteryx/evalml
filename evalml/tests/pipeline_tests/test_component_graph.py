from unittest.mock import patch

import pandas as pd
import pytest

from evalml.exceptions import MissingComponentError
from evalml.pipelines import ComponentGraph
from evalml.pipelines.components import (
    ElasticNetClassifier,
    Estimator,
    Imputer,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier
)


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

    invalid_graph = {'Imputer': [Imputer], 'OHE': OneHotEncoder}
    with pytest.raises(ValueError, match='All component information should be passed in as a list'):
        ComponentGraph(invalid_graph)


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

    bad_component_list = ['Imputer', 'Fake Estimator']
    with pytest.raises(MissingComponentError, match='was not found'):
        ComponentGraph.from_list(bad_component_list)


def test_instantiate_with_parameters(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert not isinstance(component_graph.get_component('Imputer'), Imputer)
    assert not isinstance(component_graph.get_component('Elastic Net'), ElasticNetClassifier)

    parameters = {'OneHot_RandomForest': {'top_n': 3},
                  'OneHot_ElasticNet': {'top_n': 5},
                  'Elastic Net': {'max_iter': 100}}
    component_graph.instantiate(parameters)

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


def test_instantiate_from_list():
    component_list = ['Imputer', 'One Hot Encoder', 'Random Forest Classifier']
    component_graph = ComponentGraph().from_list(component_list)

    parameters = {'One Hot Encoder': {'top_n': 7}}
    component_graph.instantiate(parameters)
    assert isinstance(component_graph.get_component('Imputer'), Imputer)
    assert isinstance(component_graph.get_component('Random Forest Classifier'), RandomForestClassifier)
    assert component_graph.get_component('One Hot Encoder').parameters['top_n'] == 7


def test_invalid_instantiate():

    component_graph = ComponentGraph({'Imputer': [Imputer(numeric_impute_strategy="most_frequent")], 'OneHot': [OneHotEncoder]})
    with pytest.raises(ValueError, match='Cannot reinstantiate component'):
        component_graph.instantiate({})
    with pytest.raises(ValueError, match='Cannot reinstantiate component'):
        component_graph.instantiate({'OneHot': {'top_n': 7}})

    graph = {'Imputer': ['Imputer', 'Fake'],
             'Fake': ['Fake Component', 'Estimator'],
             'Estimator': [ElasticNetClassifier]}
    component_graph = ComponentGraph(graph)
    with pytest.raises(MissingComponentError):
        component_graph.instantiate(parameters={})

    graph = {'Imputer': ['Imputer', 'OHE'],
             'OHE': [OneHotEncoder, 'Estimator'],
             'Estimator': [ElasticNetClassifier]}
    component_graph = ComponentGraph(graph)
    with pytest.raises(ValueError, match='Error received when instantiating component'):
        component_graph.instantiate(parameters={'Estimator': {'max_iter': 100, 'fake_param': None}})

    graph = {'Imputer': [Imputer(numeric_impute_strategy='constant', numeric_fill_value=0)]}
    component_graph = ComponentGraph(graph)
    with pytest.raises(ValueError, match='Cannot reinstantiate component'):
        component_graph.instantiate({'Imputer': {'numeric_fill_value': 1}})

    component = OneHotEncoder()
    component_graph = ComponentGraph({'OneHot': [component]})
    with pytest.raises(ValueError, match='Cannot reinstantiate component'):
        component_graph.instantiate({'OneHot': {'top_n': 3}})


def test_add_node():
    component_graph = ComponentGraph()
    component_graph.add_node('OneHot', OneHotEncoder)
    component_graph.add_node('Random Forest', RandomForestClassifier, parents=['OneHot'])
    assert len(component_graph.component_dict) == 2

    component_graph.add_node('Final', Imputer, parents=['OneHot', 'Random Forest'])
    assert len(component_graph.component_dict) == 3

    assert component_graph.get_parents('Random Forest') == ['OneHot']
    assert component_graph.get_parents('Final') == ['OneHot', 'Random Forest']

    expected_order = ['OneHot', 'Random Forest', 'Final']
    assert component_graph.compute_order == expected_order


def test_add_node_invalid():
    component_graph = ComponentGraph()
    with pytest.raises(ValueError, match='Cannot add parent that is not yet in the graph'):
        component_graph.add_node('OneHot', OneHotEncoder, parents=['Imputer'])

    component_graph = ComponentGraph({'OneHot': [OneHotEncoder]})
    with pytest.raises(ValueError, match='Cannot add a component that already exists'):
        component_graph.add_node('OneHot', OneHotEncoder)


def test_add_edge():
    component_dict = {'Imputer': [Imputer],
                      'OneHot': [OneHotEncoder],
                      'OneHot_2': [OneHotEncoder],
                      'Random Forest': [RandomForestClassifier]}
    component_graph = ComponentGraph(component_dict)
    assert len(component_graph.component_dict) == 4
    assert component_graph.compute_order == []

    component_graph.add_edge('Imputer', 'OneHot')
    component_graph.add_edge('Imputer', 'OneHot_2')
    assert len(component_graph.compute_order) == 3
    assert list(component_graph.get_parents('OneHot')) == ['Imputer']

    component_graph.add_edge('OneHot', 'Random Forest')
    assert len(component_graph.compute_order) == 4
    component_graph.add_edge('OneHot_2', 'Random Forest')
    assert len(component_graph.compute_order) == 4
    assert list(component_graph.get_parents('Random Forest')) == ['OneHot', 'OneHot_2']


def test_add_invalid_edge():
    component_dict = {'Imputer': [Imputer],
                      'OneHot': [OneHotEncoder],
                      'OneHot_2': [OneHotEncoder],
                      'Random Forest': [RandomForestClassifier]}
    component_graph = ComponentGraph(component_dict)
    with pytest.raises(ValueError, match='component not in the graph yet'):
        component_graph.add_edge('Imputer', 'Fake Component')
    with pytest.raises(ValueError, match='component not in the graph yet'):
        component_graph.add_edge('Fake Component', 'Random Forest')


def test_get_component(example_graph):
    graph = example_graph
    component_graph = ComponentGraph(graph)

    assert component_graph.get_component('OneHot_ElasticNet') == OneHotEncoder
    assert component_graph.get_component('Logistic Regression') == LogisticRegressionClassifier

    with pytest.raises(ValueError, match='not in the graph'):
        component_graph.get_component('Fake Component')


def test_get_estimators(example_graph):
    component_graph = ComponentGraph()
    assert component_graph.get_estimators() == []

    component_list = ['Imputer', 'One Hot Encoder']
    component_graph.from_list(component_list)
    assert component_graph.get_estimators() == []

    component_graph = ComponentGraph(example_graph)
    assert component_graph.get_estimators() == [RandomForestClassifier, ElasticNetClassifier, LogisticRegressionClassifier]


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


def test_get_last_component(example_graph):
    component_graph = ComponentGraph()
    assert component_graph.get_last_component() is None

    component_graph = ComponentGraph(example_graph)
    assert component_graph.get_last_component() == LogisticRegressionClassifier

    component_graph.instantiate({})
    assert component_graph.get_last_component() == LogisticRegressionClassifier()

    component_graph = ComponentGraph({'Imputer': [Imputer]})
    assert component_graph.get_last_component() == Imputer

    component_graph = ComponentGraph({'Imputer': [Imputer], 'OneHot': [OneHotEncoder, 'Imputer']})
    assert component_graph.get_last_component() == OneHotEncoder

    component_graph = ComponentGraph({'Imputer': [Imputer], 'OneHot': [OneHotEncoder]})
    assert component_graph.get_last_component() is None


@patch('evalml.pipelines.components.Transformer.fit_transform')
@patch('evalml.pipelines.components.Estimator.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_compute_final_features_fit_true(mock_fit_transform, mock_fit, mock_predict, example_graph, X_y_binary):
    X, y = X_y_binary
    mock_fit_transform.return_value = pd.DataFrame(X)
    mock_fit.return_value = Estimator
    mock_predict.return_value = pd.Series(y)
    component_graph = ComponentGraph(example_graph).instantiate({})
    component_graph.compute_final_features(X, y, fit=True)

    assert mock_fit_transform.call_count == 3
    assert mock_fit.call_count == 3
    assert mock_predict.call_count == 3


@patch('evalml.pipelines.components.Transformer.transform')
@patch('evalml.pipelines.components.Estimator.fit')
@patch('evalml.pipelines.components.Estimator.predict')
def test_compute_final_features_fit_false(mock_transform, mock_fit, mock_predict, example_graph, X_y_binary):
    X, y = X_y_binary
    mock_transform.return_value = pd.DataFrame(X)
    mock_fit.return_value = Estimator
    mock_predict.return_value = pd.Series(y)
    component_graph = ComponentGraph(example_graph).instantiate({})
    component_graph.compute_final_features(X, y, fit=True)

    component_graph.compute_final_features(X, fit=False)
    assert mock_transform.call_count == 6  # Called thrice when fitting pipeline, thrice when predicting
    assert mock_fit.call_count == 3  # Only called during fit, not predict


@patch('evalml.pipelines.components.Imputer.fit_transform')
def test_compute_final_features_y_parent(mock_fit_transform, X_y_binary):
    X, y = X_y_binary
    graph = {'Imputer': [Imputer],
             'OHE': [OneHotEncoder, 'Imputer.x', 'Imputer.y'],
             'Random Forest': [RandomForestClassifier, 'OHE.x']}
    component_graph = ComponentGraph(graph).instantiate({})
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))

    component_graph.compute_final_features(X, y, fit=True)
    mock_fit_transform.assert_called_once()


@patch('evalml.pipelines.components.OneHotEncoder.fit_transform')
@patch('evalml.pipelines.components.OneHotEncoder.transform')
def test_compute_final_features_transformer_end(mock_fit_transform, mock_transform, X_y_binary):
    X, y = X_y_binary
    graph = {'Imputer': [Imputer], 'OHE': [OneHotEncoder, 'Imputer.x']}
    component_graph = ComponentGraph(graph).instantiate({})
    mock_fit_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))
    mock_transform.return_value = tuple((pd.DataFrame(X), pd.Series(y)))

    component_graph.compute_final_features(X, y, fit=True)
    output = component_graph.compute_final_features(X)

    pd.testing.assert_frame_equal(output[0], pd.DataFrame(X))
    pd.testing.assert_series_equal(output[1], pd.Series(y))


def test_no_instantiate_before_fit(X_y_binary):
    X, y = X_y_binary
    graph = {'Imputer': [Imputer],
             'OHE': [OneHotEncoder, 'Imputer.x'],
             'Estimator': [RandomForestClassifier, 'OHE.x']}
    component_graph = ComponentGraph(graph)
    with pytest.raises(ValueError, match='All components must be instantiated before fitting or predicting'):
        component_graph.compute_final_features(X, y, fit=True)


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
        component_graph.compute_final_features(X, y, fit=True)


def test_component_graph_order(example_graph):
    component_graph = ComponentGraph(example_graph)
    expected_order = ['Imputer', 'OneHot_ElasticNet', 'Elastic Net', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
    assert expected_order == component_graph.compute_order

    component_graph = ComponentGraph({'Imputer': [Imputer]})
    expected_order = ['Imputer']
    assert expected_order == component_graph.compute_order
