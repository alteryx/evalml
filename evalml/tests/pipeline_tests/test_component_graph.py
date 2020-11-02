import pytest

from evalml.exceptions import MissingComponentError
from evalml.pipelines import ComponentGraph
from evalml.pipelines.components import (
    ElasticNetClassifier,
    Imputer,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier
)


@pytest.fixture
def example_graph():
    graph = {'Imputer': Imputer,
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

    order = [comp_name for comp_name, _ in comp_graph]
    expected_order = ['Imputer', 'OneHot_ElasticNet', 'Elastic Net', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
    assert order == expected_order


def test_from_list():
    component_list = ['Imputer', 'One Hot Encoder', RandomForestClassifier]

    component_graph = ComponentGraph()
    component_graph.from_list(component_list)

    assert len(component_graph.component_dict) == 3
    assert component_graph.get_component('Imputer') == Imputer
    assert component_graph.get_component('One Hot Encoder') == OneHotEncoder
    assert component_graph.get_component('Random Forest Classifier') == RandomForestClassifier

    order = [comp_name for comp_name, _ in component_graph]
    expected_order = ['Imputer', 'One Hot Encoder', 'Random Forest Classifier']
    assert order == expected_order

    bad_component_list = ['Imputer', 'Fake Estimator']
    component_graph = ComponentGraph()
    with pytest.raises(MissingComponentError, match='was not found'):
        component_graph.from_list(bad_component_list)


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


def test_instantiate_mixed():
    component = OneHotEncoder()
    component_graph = ComponentGraph({'OneHot': component})
    component_graph.instantiate({})
    assert isinstance(component_graph.get_component('OneHot'), OneHotEncoder)

    component_graph = ComponentGraph({'Imputer': Imputer(numeric_impute_strategy="most_frequent"), 'OneHot': OneHotEncoder})
    component_graph.instantiate({'OneHot': {'top_n': 7}})
    assert component_graph.get_component('Imputer').parameters['numeric_impute_strategy'] == 'most_frequent'
    assert component_graph.get_component('OneHot').parameters['top_n'] == 7


def test_invalid_instantiate():
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
    with pytest.raises(ValueError, match='component already instantiated'):
        component_graph.instantiate({'Imputer': {'numeric_fill_value': 1}})

    component = OneHotEncoder()
    component_graph = ComponentGraph({'OneHot': [component]})
    with pytest.raises(ValueError, match='component already instantiated'):
        component_graph.instantiate({'OneHot': {'top_n': 3}})


def test_add_node():
    component_graph = ComponentGraph()
    component_graph.add_node('OneHot', OneHotEncoder)
    component_graph.add_node('Random Forest', RandomForestClassifier, parents=['OneHot'])
    assert len(component_graph.component_dict) == 2

    component_graph.add_node('Final', Imputer, parents=['OneHot', 'Random Forest'])
    assert len(component_graph.component_dict) == 3

    assert component_graph.parents('Random Forest') == ['OneHot']
    assert component_graph.parents('Final') == ['OneHot', 'Random Forest']

    order = [comp_name for comp_name, _ in component_graph]
    expected_order = ['OneHot', 'Random Forest', 'Final']
    assert order == expected_order


def test_add_node_invalid():
    component_graph = ComponentGraph()
    with pytest.raises(ValueError, match='Cannot add parent that is not yet in the graph'):
        component_graph.add_node('OneHot', OneHotEncoder, parents=['Imputer'])

    component_graph = ComponentGraph({'OneHot': [OneHotEncoder]})
    with pytest.raises(ValueError, match='Cannot add a component that already exists'):
        component_graph.add_node('OneHot', OneHotEncoder)


def test_add_edge():
    component_dict = {'Imputer': Imputer,
                      'OneHot': OneHotEncoder,
                      'OneHot_2': OneHotEncoder,
                      'Random Forest': RandomForestClassifier}
    component_graph = ComponentGraph(component_dict)
    order = [comp_name for comp_name, _ in component_graph]
    assert len(component_graph.component_dict) == 4
    assert order == []

    component_graph.add_edge('Imputer', 'OneHot')
    component_graph.add_edge('Imputer', 'OneHot_2')
    order = [comp_name for comp_name, _ in component_graph]
    assert len(order) == 3
    assert list(component_graph.parents('OneHot')) == ['Imputer']

    component_graph.add_edge('OneHot', 'Random Forest')
    order = [comp_name for comp_name, _ in component_graph]
    assert len(order) == 4
    component_graph.add_edge('OneHot_2', 'Random Forest')
    order = [comp_name for comp_name, _ in component_graph]
    assert len(order) == 4
    assert list(component_graph.parents('Random Forest')) == ['OneHot', 'OneHot_2']


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

    assert component_graph.parents('Imputer') == []
    assert component_graph.parents('OneHot_RandomForest') == ['Imputer.x']
    assert component_graph.parents('OneHot_ElasticNet') == ['Imputer.x']
    assert component_graph.parents('Random Forest') == ['OneHot_RandomForest.x']
    assert component_graph.parents('Elastic Net') == ['OneHot_ElasticNet.x']
    assert component_graph.parents('Logistic Regression') == ['Random Forest', 'Elastic Net']

    with pytest.raises(ValueError, match='not in the graph'):
        component_graph.parents('Fake component')