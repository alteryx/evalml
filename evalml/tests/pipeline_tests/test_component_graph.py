import pytest

from evalml.exceptions import MissingComponentError
from evalml.pipelines import ComponentGraph
from evalml.pipelines.components import (
    Imputer,
    OneHotEncoder,
    RandomForestClassifier,
    CatBoostClassifier,
    LogisticRegressionClassifier
)


@pytest.fixture
def example_graph():
    components = {'Imputer': Imputer,
                  'OneHot_RandomForest': OneHotEncoder,
                  'OneHot_CatBoost': OneHotEncoder,
                  'Random Forest': RandomForestClassifier,
                  'CatBoost': CatBoostClassifier,
                  'Logistic Regression': LogisticRegressionClassifier}
    edges = [('Imputer', 'OneHot_RandomForest'),
             ('Imputer', 'OneHot_CatBoost'),
             ('OneHot_RandomForest', 'Random Forest'),
             ('OneHot_CatBoost', 'CatBoost'),
             ('Random Forest', 'Logistic Regression'),
             ('CatBoost', 'Logistic Regression')]
    return components, edges


def test_init(example_graph):
    comp_graph = ComponentGraph()
    assert len(comp_graph.component_names) == 0

    components, edges = example_graph
    comp_graph = ComponentGraph(components, edges)
    assert len(comp_graph.component_names) == 6

    order = [comp_name for comp_name, _ in comp_graph]
    expected_order = ['Imputer', 'OneHot_CatBoost', 'CatBoost', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
    assert order == expected_order


def test_from_list():
    component_list = ['Imputer', 'One Hot Encoder', RandomForestClassifier]

    component_graph = ComponentGraph()
    component_graph.from_list(component_list)

    assert len(component_graph.component_names) == 3
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
    components, edges = example_graph
    component_graph = ComponentGraph(components, edges)

    assert not isinstance(component_graph.get_component('Imputer'), Imputer)
    assert not isinstance(component_graph.get_component('CatBoost'), CatBoostClassifier)

    parameters = {'OneHot_RandomForest': {'top_n': 3},
                  'OneHot_CatBoost': {'top_n': 5},
                  'CatBoost': {'n_estimators': 12}}
    component_graph.instantiate(parameters)

    assert isinstance(component_graph.get_component('Imputer'), Imputer)
    assert isinstance(component_graph.get_component('Random Forest'), RandomForestClassifier)
    assert isinstance(component_graph.get_component('Logistic Regression'), LogisticRegressionClassifier)
    assert component_graph.get_component('OneHot_RandomForest').parameters['top_n'] == 3
    assert component_graph.get_component('OneHot_CatBoost').parameters['top_n'] == 5
    assert component_graph.get_component('CatBoost').parameters['n_estimators'] == 12


def test_instantiate_without_parameters(example_graph):
    components, edges = example_graph
    component_graph = ComponentGraph(components, edges)
    component_graph.instantiate({})
    assert component_graph.get_component('OneHot_RandomForest').parameters['top_n'] == 10
    assert component_graph.get_component('OneHot_CatBoost').parameters['top_n'] == 10
    assert component_graph.get_component('OneHot_RandomForest') is not component_graph.get_component('OneHot_CatBoost')


def test_instantiate_from_list():
    component_list = ['Imputer', 'One Hot Encoder', 'Random Forest Classifier']
    component_graph = ComponentGraph().from_list(component_list)

    parameters = {'One Hot Encoder': {'top_n': 7}}
    component_graph.instantiate(parameters)
    assert isinstance(component_graph.get_component('Imputer'), Imputer)
    assert isinstance(component_graph.get_component('Random Forest Classifier'), RandomForestClassifier)
    assert component_graph.get_component('One Hot Encoder').parameters['top_n'] == 7


def test_invalid_instantiate():
    component = OneHotEncoder()
    component_graph = ComponentGraph(component_names={'OneHot': component})
    with pytest.raises(ValueError, match='Cannot instantiate already instantiated component'):
        component_graph.instantiate({'OneHot': {'top_n': 3}})

    components = {'Imputer': 'Imputer', 'Fake': 'Fake Component', 'Estimator': CatBoostClassifier}
    edges = [('Imputer', 'Fake'), ('Fake', 'Estimator')]
    component_graph = ComponentGraph(components, edges)
    with pytest.raises(MissingComponentError):
        component_graph.instantiate(parameters={})


def test_add_node():
    component_graph = ComponentGraph()
    component_graph.add_node('OneHot', OneHotEncoder)
    component_graph.add_node('Random Forest', RandomForestClassifier, parents=['OneHot'])
    component_graph.add_node('OneHot_2', OneHotEncoder, children=['Random Forest'])
    assert len(component_graph.component_names) == 3

    component_graph.add_node('Imputer', Imputer, children=['OneHot', 'OneHot_2'])
    assert len(component_graph.component_names) == 4

    assert list(component_graph.parents('OneHot')) == ['Imputer']
    assert list(component_graph.parents('Random Forest')) == ['OneHot', 'OneHot_2']

    order = [comp_name for comp_name, _ in component_graph]
    expected_order = ['Imputer', 'OneHot_2', 'OneHot', 'Random Forest']
    assert order == expected_order


def test_add_edge():
    component_names = {'Imputer': Imputer,
                       'OneHot': OneHotEncoder,
                       'OneHot_2': OneHotEncoder,
                       'Random Forest': RandomForestClassifier}
    component_graph = ComponentGraph(component_names=component_names)
    order = [comp_name for comp_name, _ in component_graph]
    assert len(component_graph.component_names) == 4
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
    component_names = {'Imputer': Imputer,
                       'OneHot': OneHotEncoder,
                       'OneHot_2': OneHotEncoder,
                       'Random Forest': RandomForestClassifier}
    component_graph = ComponentGraph(component_names=component_names)
    with pytest.raises(ValueError, match='component not in the graph yet'):
        component_graph.add_edge('Imputer', 'Fake Component')
    with pytest.raises(ValueError, match='component not in the graph yet'):
        component_graph.add_edge('Fake Component', 'Random Forest')


def test_merge_graph():
    component_graph = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot_RandomForest': OneHotEncoder, 'Random Forest': RandomForestClassifier},
                                     edges=[('Imputer', 'OneHot_RandomForest'), ('OneHot_RandomForest', 'Random Forest')])
    component_graph_2 = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot_CatBoost': OneHotEncoder, 'CatBoost': CatBoostClassifier},
                                       edges=[('Imputer', 'OneHot_CatBoost'), ('OneHot_CatBoost', 'CatBoost')])

    assert len(component_graph.component_names) == 3
    assert len([comp_name for comp_name, _ in component_graph]) == 3
    component_graph.merge_graph(component_graph_2)
    assert len(component_graph.component_names) == 5
    assert len([comp_name for comp_name, _ in component_graph]) == 5
    order = [comp_name for comp_name, _ in component_graph]
    expected_order = ['Imputer', 'OneHot_CatBoost', 'CatBoost', 'OneHot_RandomForest', 'Random Forest']
    assert order == expected_order

    parameters = {'OneHot_RandomForest': {'top_n': 3},
                  'OneHot_CatBoost': {'top_n': 5}}
    component_graph.instantiate(parameters)
    assert component_graph.get_component('OneHot_RandomForest') != component_graph.get_component('OneHot_CatBoost')


def test_get_component(example_graph):
    components, edges = example_graph
    component_graph = ComponentGraph(components, edges)

    assert component_graph.get_component('OneHot_CatBoost') == OneHotEncoder
    assert component_graph.get_component('Logistic Regression') == LogisticRegressionClassifier

    with pytest.raises(ValueError, match='not in the graph'):
        component_graph.get_component('Fake Component')


def test_get_estimators(example_graph):
    component_graph = ComponentGraph()
    assert component_graph.get_estimators() == []

    component_list = ['Imputer', 'One Hot Encoder']
    component_graph.from_list(component_list)
    assert component_graph.get_estimators() == []

    component_graph = ComponentGraph(example_graph[0], example_graph[1])
    assert component_graph.get_estimators() == [RandomForestClassifier, CatBoostClassifier, LogisticRegressionClassifier]


def test_parents(example_graph):
    components, edges = example_graph
    component_graph = ComponentGraph(components, edges)

    assert list(component_graph.parents('Imputer')) == []
    assert list(component_graph.parents('OneHot_RandomForest')) == ['Imputer']
    assert list(component_graph.parents('OneHot_CatBoost')) == ['Imputer']
    assert list(component_graph.parents('Random Forest')) == ['OneHot_RandomForest']
    assert list(component_graph.parents('CatBoost')) == ['OneHot_CatBoost']
    assert list(component_graph.parents('Logistic Regression')) == ['Random Forest', 'CatBoost']

    with pytest.raises(ValueError, match='not in the graph'):
        component_graph.parents('Fake component')
