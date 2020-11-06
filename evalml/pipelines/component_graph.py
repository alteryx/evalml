import networkx as nx
import pandas as pd
from networkx.algorithms.dag import topological_sort

from evalml.model_family import ModelFamily
from evalml.pipelines.components import ComponentBase
from evalml.pipelines.components.utils import handle_component_class
from evalml.utils import import_or_raise


class ComponentGraph:
    def __init__(self, component_dict=None, random_state=0):
        """ Initializes a component graph for a pipeline as a DAG.
        """
        self.component_dict = component_dict or {}
        for key, value in self.component_dict.items():
            if not isinstance(value, list):
                self.component_dict[key] = [value]
        self._compute_order = []
        self._recompute_order()
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

            parent = None
            if idx != 0:
                parent = [f"{handle_component_class(component_list[idx - 1]).name}.x"]
            self.add_node(component_name, component_class, parents=parent)
        return self

    def instantiate(self, parameters):
        """Instantiates all uninstantiated components within the graph using the given parameters. An error will be
        raised if a component is already instantiated but the parameters dict contains arguments for that component.

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                               An empty dictionary {} implies using all default values for component parameters.
        """
        for component_name, component_info in self.component_dict.items():
            component_parameters = parameters.get(component_name, {})
            component_class = component_info[0]
            if isinstance(component_class, ComponentBase):
                if component_parameters != {}:
                    raise ValueError(f"Attempting to instantiate component {component_name} with parameters {component_parameters} when component already instantiated")
                else:
                    continue

            component_class = handle_component_class(component_class)

            try:
                new_component = component_class(**component_parameters, random_state=self.random_state)
            except (ValueError, TypeError) as e:
                err = "Error received when instantiating component {} with the following arguments {}".format(component_name, component_parameters)
                raise ValueError(err) from e

            self.component_dict[component_name][0] = new_component
        return self

    def compute_final_features(self, X, y=None, fit=False):
        """Transforms the data by applying all components.

        Arguments:
            X (pd.DataFrame): Input data to the pipeline to transform.
            y (pd.Series): The target training data of length [n_samples]
            fit (bool): Whether to fit the estimators as well as transform it.
                        Defaults to False.

        Returns:
            pd.DataFrame and/or pd.Series - Output(s) of the final component
        """
        output_cache = {}
        final_component = None
        for component_name in list(self._compute_order):
            final_component = component_name
            component_class = self.component_dict[component_name][0]
            x_inputs = []
            y_input = None
            for parent_input in self.parents(component_name):
                if parent_input[-2:] == '.y':
                    y_input = output_cache[parent_input]
                else:
                    x_inputs.append(output_cache[parent_input])
            input_x, input_y = self.merge(x_inputs, y_input, X, y)
            if component_class.model_family == ModelFamily.NONE:  # Transformer
                if fit:
                    output = component_class.fit_transform(input_x, input_y)
                else:
                    output = component_class.transform(input_x, input_y)
                if isinstance(output, tuple):
                    output_x, output_y = output[0], output[1]
                else:
                    output_x = output
                    output_y = None
                output_cache[f"{component_name}.x"] = output_x
                output_cache[f"{component_name}.y"] = output_y
            else:  # Estimator
                if fit:
                    component_class = component_class.fit(input_x, input_y)
                output = component_class.predict(input_x)
                output_cache[component_name] = output
        self._recompute_order()
        if fit:
            return self
        final_component_class = self.component_dict[final_component][0]
        if final_component_class.model_family == ModelFamily.NONE:
            return output_cache[f"{final_component}.x"], output_cache[f"{final_component}.y"]
        else:
            return output_cache[final_component]

    @staticmethod
    def merge(x_inputs, y_input, X, y):
        return_y = y
        if len(x_inputs) == 0:
            return_x = X
        else:
            return_x = pd.DataFrame()
            for x_input in x_inputs:
                return_x = pd.concat([return_x, x_input], axis=1)
        if y_input is not None:
            return_y = y
        return return_x, return_y

    def add_node(self, component_name, component_obj, parents=None):
        """Add a node to the component graph.

        Arguments:
            component_name (str or int): The name or ID of the component to add
            component_obj (Object or string): The component to add
            parents (list): A list of parents of this new node. Defaults to None.
        """
        if component_name in self.component_dict.keys():
            raise ValueError('Cannot add a component that already exists')
        self.component_dict[component_name] = [component_obj]
        valid_parents = self.component_dict.keys()
        if parents:
            for parent in parents:
                if parent[:-2] not in valid_parents and parent not in valid_parents:
                    raise ValueError('Cannot add parent that is not yet in the graph')
                self.component_dict[component_name].append(parent)
        self._recompute_order()
        return self

    def add_edge(self, from_component, to_component):
        """Add an edge connecting two nodes of the component graph. Note that both nodes
        must already exist inside the graph.

        Arguments:
            from_component (str or int): The parent node, to be computed first
            to_component (str or int): The child node, to be computed after the parent
        """
        if from_component not in self.component_dict.keys() or to_component not in self.component_dict.keys():
            raise ValueError("Cannot add an edge for a component not in the graph yet")
        self.component_dict[to_component].append(from_component)
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
            return self.component_dict[component_name][0]
        except KeyError:
            raise ValueError(f'Component {component_name} is not in the graph')

    def get_final_component(self):
        """Retrieves the component that is computed last in the graph, usually the final estimator.

        Returns:
            ComponentBase object
        """
        compute_list = list(self._compute_order)
        self._recompute_order()
        if len(compute_list) == 0:
            if len(self.component_dict) > 0:
                if len(self.component_dict) == 1:
                    return list(self.component_dict.keys())[0]
                raise ValueError("There are no edges in the graph, no final component to return")
            return None
        return compute_list[-1]

    def get_estimators(self):
        """Gets a list of all the estimator components within this graph

        Returns:
            list: all estimator objects within the graph
        """
        estimators = []
        for component_info in self.component_dict.values():
            component = component_info[0]
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
        try:
            component_info = self.component_dict[component_name]
        except KeyError:
            raise ValueError(f"Component {component_name} not in the graph")
        if len(component_info) > 1:
            return component_info[1:]
        return []

    def graph(self, name=None, graph_format=None):
        """Generate an image representing the component graph

        Arguments:
            name (str): Name of the graph. Defaults to None.
            graph_format (str): file format to save the graph in. Defaults to None.

        Returns:
            graphviz.Digraph: Graph object that can be directly displayed in Jupyter notebooks.
        """
        graphviz = import_or_raise('graphviz', error_msg='Please install graphviz to visualize pipelines.')

        # Try rendering a dummy graph to see if a working backend is installed
        try:
            graphviz.Digraph().pipe()
        except graphviz.backend.ExecutableNotFound:
            raise RuntimeError(
                "To graph entity sets, a graphviz backend is required.\n" +
                "Install the backend using one of the following commands:\n" +
                "  Mac OS: brew install graphviz\n" +
                "  Linux (Ubuntu): sudo apt-get install graphviz\n" +
                "  Windows: conda install python-graphviz\n"
            )

        graph = graphviz.Digraph(name=name, format=graph_format,
                                 graph_attr={'splines': 'ortho'})
        graph.attr(rankdir='LR')
        for component_name, component_class in self.component_names.items():
            label = '%s\l' % (component_name)  # noqa: W605
            if isinstance(component_class, ComponentBase):
                parameters = '\l'.join([key + ' : ' + "{:0.2f}".format(val) if (isinstance(val, float))
                                        else key + ' : ' + str(val)
                                        for key, val in component_class.parameters.items()])  # noqa: W605
                label = '%s |%s\l' % (component_name, parameters)  # noqa: W605
            graph.node(component_name, shape='record', label=label)
        graph.edges(self.edges)
        return graph

    def _get_edges(self):
        edges = []
        for component_name, component_info in self.component_dict.items():
            if len(component_info) > 1:
                for parent in component_info[1:]:
                    if parent[-2:] == '.x' or parent[-2:] == '.y':
                        parent = parent[:-2]
                    edges.append((parent, component_name))
        return edges

    def _recompute_order(self):
        """Regenerated the topologically sorted order of the graph"""
        digraph = nx.DiGraph()
        digraph.add_edges_from(self._get_edges())
        self._compute_order = topological_sort(digraph)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            component = next(self._compute_order)
            if component[-2:] == '.x' or component[-2:] == '.y':
                component = component[:-2]
            return component, self.component_dict[component][0]
        except StopIteration:
            self._recompute_order()  # Reset the generator
            raise StopIteration
