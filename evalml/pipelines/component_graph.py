import graphviz
import networkx as nx
from networkx.algorithms.dag import topological_sort

from evalml.exceptions import MissingComponentError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ComponentBase
from evalml.pipelines.components.utils import handle_component_class

class ComponentGraph:
    def  __init__(self, component_names=None, edges=None, random_state=0):
        """ Initializes a component graph for a pipeline as a DAG.
        
        Arguments:
            component_names (dict): Of the form {name: component} pairs, where
                             `name` is a unique string for that component and
                             `component` is either a string component name
                             as recognized by evalml or a direct evalml
                             component class
            edges (list): A list of tuples of the form (from_component, 
                          to_component), referring to the components by the
                          names as represented in the `component_names` dict
        """
        self.component_names = component_names or {}
        self._component_graph = nx.DiGraph()
        if edges:
            self._component_graph.add_edges_from(edges)
        self._compute_order = topological_sort(self._component_graph)
        self.random_state = random_state

    def from_list(self, component_list):
        """Constructs a linear graph from a given list

        Arguments:
            component_list (list): String names or ComponentBase subclasses in 
                                   an order that represents a valid linear graph
        """
        for idx, component in enumerate(component_list):
            component_class = handle_component_class(component)
            component_name = component_class.name

            child = None
            if idx != len(component_list)-1:
                child = [handle_component_class(component_list[idx+1]).name]
            self.add_node(component_name, component_class, children=child)
        return self

    def instantiate(self, parameters):
        """Instantiates all components within the graph using the given parameters

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                               An empty dictionary {} implies using all default values for component parameters.
        """
        for component_name, component_class in self.component_names.items():
            if isinstance(component_class, ComponentBase):
                raise ValueError(f'Cannot instantiate already instantiated component {component_name}')

            try:
                component_class = handle_component_class(component_class)
            except MissingComponentError as e:
                err = "Error recieved when retrieving class for component '{}'".format(component_class)
                raise MissingComponentError(err) from e

            try:
                component_parameters = parameters.get(component_name, {})
                new_component = component_class(**component_parameters, random_state=self.random_state)
            except (ValueError, TypeError) as e:
                err = "Error received when instantiating component {} with the following arguments {}".format(component_name, component_parameters)
                raise ValueError(err) from e

            self.component_names[component_name] = new_component
        return self

    def add_node(self, component_name, component_obj, parents=None, children=None):
        """Add a node to the component graph.

        Arguments:
            component_name (str or int): The name or ID of the component to add
            component_obj (Object or string): The component to add
            parents (list): A list of parents of this new node. Defaults to None.
            children (list): A list of children of this new node. Defaults to  None.
        """
        self.component_names[component_name] = component_obj
        if parents:
            for parent in parents:
                self._component_graph.add_edge(parent, component_name)
        if children:
            for child in children:
                self._component_graph.add_edge(component_name, child)
        self._recompute_order()
        return self

    def add_edge(self, from_component, to_component):
        """Add an edge connecting two nodes of the component graph. Note that both nodes
        must already exist inside the graph.

        Arguments:
            from_component (str or int): The parent node, to be computed first
            to_component (str or int): The child node, to be computed after the parent
        """
        if from_component not in self.component_names.keys() or to_component not in self.component_names.keys():
            raise ValueError("Cannot add an edge for a component not in the graph yet")
        self._component_graph.add_edge(from_component, to_component)
        self._recompute_order()
        return self

    def merge_graph(self, other_graph):
        """Add all components and edges from another `ComponentGraph` object to this one. 
        Components with unique names will be added as unique nodes, even if the component object
        is the same type as another. Components with identical names will be considered the 
        same node, and the object from the incoming graph will be saved as that component.

        Arguments:
            other_graph (ComponentGraph): The other graph to combine with this graph
        """
        for component_name, component in other_graph.component_names.items():
            self.component_names[component_name] = component
        self._component_graph.add_edges_from(other_graph._component_graph.edges)
        self._recompute_order()
        return self

    def get_component(self, component_name):
        """Retrieves a single component object from the graph.

        Arguments:
            component_name (str): Name of the component to retrieve

        Returns:
            ComponentBase object
        """
        try:
            return self.component_names[component_name]
        except KeyError:
            raise ValueError(f'Component {component_name} is not in the graph')

    def get_estimators(self):
        """Gets a list of all the estimator components within this graph

        Returns:
            list: all estimator objects within the graph
        """
        estimators = []
        for component in self.component_names.values():
            if component.model_family is not ModelFamily.NONE:
                estimators.append(component)
        return estimators

    def parents(self, component_name):
        """Finds the names of all parent nodes of the given component

        Arguments:
            component_name (str): Name of the child component to look up

        Returns:
            iterator of parent component names
        """
        if component_name not in self.component_names.keys():
            raise ValueError(f'Component {component_name} is not in the graph')
        return self._component_graph.predecessors(component_name)

    def graph(self, name, format):
        # TODO: Come back and make this pretty
        visual = graphviz.Digraph(name=name, format=format)
        for node in self._component_graph.nodes:
            visual.node(node)
        visual.edges(self._component_graph.edges)
        return visual

    def _recompute_order(self):
        """Regenerated the topologically sorted order of the graph"""
        self._compute_order = topological_sort(self._component_graph)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            component = next(self._compute_order)
            return component, self.component_names[component]
        except StopIteration:
            self._recompute_order() # Reset the generator
            raise StopIteration
