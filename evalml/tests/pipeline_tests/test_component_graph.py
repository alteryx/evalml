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
    components = {'Imputer': Imputer,
                  'OneHot_RandomForest': OneHotEncoder,
                  'OneHot_ElasticNet': OneHotEncoder,
                  'Random Forest': RandomForestClassifier,
                  'Elastic Net': ElasticNetClassifier,
                  'Logistic Regression': LogisticRegressionClassifier}
    edges = [('Imputer', 'OneHot_RandomForest'),
             ('Imputer', 'OneHot_ElasticNet'),
             ('OneHot_RandomForest', 'Random Forest'),
             ('OneHot_ElasticNet', 'Elastic Net'),
             ('Random Forest', 'Logistic Regression'),
             ('Elastic Net', 'Logistic Regression')]
    return components, edges


def test_init(example_graph):
    comp_graph = ComponentGraph()
    assert len(comp_graph.component_names) == 0

    components, edges = example_graph
    comp_graph = ComponentGraph(components, edges)
    assert len(comp_graph.component_names) == 6

    order = [comp_name for comp_name, _ in comp_graph]
    expected_order = ['Imputer', 'OneHot_ElasticNet', 'Elastic Net', 'OneHot_RandomForest', 'Random Forest', 'Logistic Regression']
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
    components, edges = example_graph
    component_graph = ComponentGraph(components, edges)
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
    component_graph = ComponentGraph(component_names={'OneHot': component})
    component_graph.instantiate({})
    assert isinstance(component_graph.get_component('OneHot'), OneHotEncoder)

    component_graph = ComponentGraph(component_names={'Imputer': Imputer(numeric_impute_strategy="most_frequent"), 'OneHot': OneHotEncoder})
    component_graph.instantiate({'OneHot': {'top_n': 7}})
    assert component_graph.get_component('Imputer').parameters['numeric_impute_strategy'] == 'most_frequent'
    assert component_graph.get_component('OneHot').parameters['top_n'] == 7


def test_invalid_instantiate():
    components = {'Imputer': 'Imputer', 'Fake': 'Fake Component', 'Estimator': ElasticNetClassifier}
    edges = [('Imputer', 'Fake'), ('Fake', 'Estimator')]
    component_graph = ComponentGraph(components, edges)
    with pytest.raises(MissingComponentError):
        component_graph.instantiate(parameters={})

    components = {'Imputer': 'Imputer', 'OneHot': 'One Hot Encoder', 'Estimator': ElasticNetClassifier}
    edges = [('Imputer', 'OneHot'), ('OneHot', 'Estimator')]
    component_graph = ComponentGraph(components, edges)
    with pytest.raises(ValueError, match='Error received when instantiating component'):
        component_graph.instantiate(parameters={'Estimator': {'max_iter': 100, 'fake_param': None}})

    component_graph = ComponentGraph(component_names={'Imputer': Imputer(numeric_impute_strategy='constant', numeric_fill_value=0)})
    with pytest.raises(ValueError, match='component already instantiated'):
        component_graph.instantiate({'Imputer': {'numeric_fill_value': 1}})

    component = OneHotEncoder()
    component_graph = ComponentGraph(component_names={'OneHot': component})
    with pytest.raises(ValueError, match='component already instantiated'):
        component_graph.instantiate({'OneHot': {'top_n': 3}})


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


def test_add_node_invalid():
    component_graph = ComponentGraph()
    with pytest.raises(ValueError, match='Cannot add parent that is not yet in the graph'):
        component_graph.add_node('OneHot', OneHotEncoder, parents=['Imputer'])

    component_graph = ComponentGraph(component_names={'Imputer': Imputer})
    with pytest.raises(ValueError, match='Cannot add child that is not yet in the graph'):
        component_graph.add_node('OneHot', OneHotEncoder, children=['Imputer', 'Random Forest'])

    component_graph = ComponentGraph(component_names={'OneHot': OneHotEncoder})
    with pytest.raises(ValueError, match='Cannot add a component that already exists'):
        component_graph.add_node('OneHot', OneHotEncoder)


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
    component_graph_2 = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot_ElasticNet': OneHotEncoder, 'ElasticNet': ElasticNetClassifier},
                                       edges=[('Imputer', 'OneHot_ElasticNet'), ('OneHot_ElasticNet', 'ElasticNet')])

    assert len(component_graph.component_names) == 3
    assert len([comp_name for comp_name, _ in component_graph]) == 3
    component_graph.merge_graph(component_graph_2)
    assert len(component_graph.component_names) == 5
    assert len([comp_name for comp_name, _ in component_graph]) == 5
    order = [comp_name for comp_name, _ in component_graph]
    expected_order = ['Imputer', 'OneHot_ElasticNet', 'ElasticNet', 'OneHot_RandomForest', 'Random Forest']
    assert order == expected_order
    assert component_graph.get_component('Imputer') is component_graph_2.get_component('Imputer')

    parameters = {'OneHot_RandomForest': {'top_n': 3},
                  'OneHot_ElasticNet': {'top_n': 5}}
    component_graph.instantiate(parameters)
    assert component_graph.get_component('OneHot_RandomForest') != component_graph.get_component('OneHot_ElasticNet')


def test_merge_graph_empty():
    component_graph = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot_RandomForest': OneHotEncoder, 'Random Forest': RandomForestClassifier},
                                     edges=[('Imputer', 'OneHot_RandomForest'), ('OneHot_RandomForest', 'Random Forest')])
    component_graph_2 = ComponentGraph()
    component_graph.merge_graph(component_graph_2)
    assert len(component_graph.component_names) == 3
    order = [comp_name for comp_name, _ in component_graph]
    expected_order = ['Imputer', 'OneHot_RandomForest', 'Random Forest']
    assert order == expected_order

    component_graph = ComponentGraph()
    component_graph_2 = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot_RandomForest': OneHotEncoder, 'Random Forest': RandomForestClassifier},
                                       edges=[('Imputer', 'OneHot_RandomForest'), ('OneHot_RandomForest', 'Random Forest')])
    component_graph.merge_graph(component_graph_2)
    assert len(component_graph.component_names) == 3
    order = [comp_name for comp_name, _ in component_graph]
    expected_order = ['Imputer', 'OneHot_RandomForest', 'Random Forest']
    assert order == expected_order


def test_merge_graph_identical():
    component_graph = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot_RandomForest': OneHotEncoder, 'Random Forest': RandomForestClassifier},
                                     edges=[('Imputer', 'OneHot_RandomForest'), ('OneHot_RandomForest', 'Random Forest')])
    component_graph_2 = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot_RandomForest': OneHotEncoder, 'Random Forest': RandomForestClassifier},
                                       edges=[('Imputer', 'OneHot_RandomForest'), ('OneHot_RandomForest', 'Random Forest')])
    component_graph.merge_graph(component_graph_2)
    assert len(component_graph.component_names) == 3
    order = [comp_name for comp_name, _ in component_graph]
    expected_order = ['Imputer', 'OneHot_RandomForest', 'Random Forest']
    assert order == expected_order


def test_merge_graph_post_instantiation():
    component_graph = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot': OneHotEncoder, 'Random Forest': RandomForestClassifier},
                                     edges=[('Imputer', 'OneHot'), ('OneHot', 'Random Forest')])
    component_graph_2 = ComponentGraph(component_names={'Imputer': Imputer, 'OneHot': OneHotEncoder, 'ElasticNet': ElasticNetClassifier},
                                       edges=[('Imputer', 'OneHot'), ('OneHot', 'ElasticNet')])
    component_graph.instantiate({'OneHot': {'top_n': 5}})
    component_graph_2.instantiate({'OneHot': {'top_n': 7}})
    component_graph.merge_graph(component_graph_2)
    assert component_graph.get_component('OneHot').parameters['top_n'] == 7


def test_get_component(example_graph):
    components, edges = example_graph
    component_graph = ComponentGraph(components, edges)

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

    component_graph = ComponentGraph(example_graph[0], example_graph[1])
    assert component_graph.get_estimators() == [RandomForestClassifier, ElasticNetClassifier, LogisticRegressionClassifier]


def test_parents(example_graph):
    components, edges = example_graph
    component_graph = ComponentGraph(components, edges)

    assert list(component_graph.parents('Imputer')) == []
    assert list(component_graph.parents('OneHot_RandomForest')) == ['Imputer']
    assert list(component_graph.parents('OneHot_ElasticNet')) == ['Imputer']
    assert list(component_graph.parents('Random Forest')) == ['OneHot_RandomForest']
    assert list(component_graph.parents('Elastic Net')) == ['OneHot_ElasticNet']
    assert list(component_graph.parents('Logistic Regression')) == ['Random Forest', 'Elastic Net']

    with pytest.raises(ValueError, match='not in the graph'):
        component_graph.parents('Fake component')