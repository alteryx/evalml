import networkx as nx
import pandas as pd
from networkx.algorithms.dag import topological_sort
from networkx.exception import NetworkXUnfeasible

from evalml.pipelines.components import ComponentBase, Estimator, Transformer
from evalml.pipelines.components.utils import handle_component_class
from evalml.utils import get_random_state, import_or_raise


class ComponentGraph:
    def __init__(self, component_dict=None, random_state=0):
        """ Initializes a component graph for a pipeline as a directed acyclic graph (DAG).

        Example:
            >>> component_dict = {'imputer': ['Imputer'], 'ohe': ['One Hot Encoder', 'imputer.x'], 'estimator_1': ['Random Forest Classifier', 'ohe.x'], 'estimator_2': ['Decision Tree Classifier', 'ohe.x'], 'final': ['Logistic Regression Classifier', 'estimator_1', 'estimator_2']}
            >>> component_graph = ComponentGraph(component_dict)
           """
        self.component_dict = component_dict or {}
        self.component_instances = {}
        self._is_instantiated = False
        for component_name, component_info in self.component_dict.items():
            if not isinstance(component_info, list):
                raise ValueError('All component information should be passed in as a list')
            component_class = handle_component_class(component_info[0])
            self.component_instances[component_name] = component_class
        self.compute_order = []
        self._recompute_order()
        self.input_feature_names = {}
        self.random_state = get_random_state(random_state)

    @classmethod
    def from_list(cls, component_list, random_state=0):
        """Constructs a linear ComponentGraph from a given list, where each component in the list feeds its X transformed output to the next component

        Arguments:
            component_list (list): String names or ComponentBase subclasses in
                                   an order that represents a valid linear graph
        """
        component_dict = {}
        previous_component = None
        for idx, component in enumerate(component_list):
            component_class = handle_component_class(component)
            component_name = component_class.name

            if component_name in component_dict.keys():
                component_name = f'{component_name}_{idx}'

            component_dict[component_name] = [component_class]
            if previous_component is not None:
                component_dict[component_name].append(f"{previous_component}.x")
            previous_component = component_name
        return cls(component_dict, random_state=random_state)

    def instantiate(self, parameters):
        """Instantiates all uninstantiated components within the graph using the given parameters. An error will be
        raised if a component is already instantiated but the parameters dict contains arguments for that component.

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                               An empty dictionary {} implies using all default values for component parameters.
        """
        if self._is_instantiated:
            raise ValueError(f"Cannot reinstantiate a component graph that was previously instantiated")

        self._is_instantiated = True
        for component_name, component_class in self.component_instances.items():
            component_parameters = parameters.get(component_name, {})

            try:
                new_component = component_class(**component_parameters, random_state=self.random_state)
            except (ValueError, TypeError) as e:
                err = "Error received when instantiating component {} with the following arguments {}".format(component_name, component_parameters)
                raise ValueError(err) from e

            self.component_instances[component_name] = new_component
        return self

    def fit(self, X, y):
        """Fit each component in the graph

        Arguments:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (pd.Series): The target training data of length [n_samples]
        """
        self._compute_features(self.compute_order, X, y, fit=True)
        return self

    def fit_features(self, X, y):
        """ Fit all components save the final one, usually an estimator. Note that this is
        only useful where the final component has only one parent, otherwise there will be
        information missing.

        Arguments:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (pd.Series): The target training data of length [n_samples]
        """
        return self._compute_features(self.compute_order[:-1], X, y, fit=True)

    def predict(self, X):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame): Data of shape [n_samples, n_features]

        Returns:
            pd.Series: Predicted values.
        """
        return self._compute_features(self.compute_order, X)

    def compute_final_component_features(self, X, y=None):
        """ Transform all components save the final one, and gathers the data from any number of parents
        to get all the information that should be fed to the final component

        Arguments:
            X (pd.DataFrame): Data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame: Transformed values.
        """
        if len(self.compute_order) <= 1:
            return X
        x_inputs = []
        for parent in self.get_parents(self.compute_order[-1]):
            if parent[-2:] == '.x' or parent[-2] == '.y':
                parent = parent[:-2]
            parent_path = self._get_ancestors(parent)
            parent_path.append(parent)
            print('evaluating parent path', parent_path)
            parent_input = self._compute_features(parent_path, X, y=y)
            if isinstance(parent_input, pd.Series):
                parent_input = pd.DataFrame(parent_input, columns=[parent])
            x_inputs.append(parent_input)
        X_t = pd.concat(x_inputs, axis=1)
        return X_t

    def _get_ancestors(self, component):
        ancestors = []
        parents = self.get_parents(component)
        while len(parents) > 0:
            parent = parents.pop()
            if parent[-2:] == '.x' or parent[-2:] == '.y':
                parent = parent[:-2]
            ancestors.append(parent)
            parents.extend(self.get_parents(parent))
        ancestors.reverse()
        return ancestors

    def _compute_features(self, component_list, X, y=None, fit=False):
        """Transforms the data by applying the given components.

        Arguments:
            component_list (list): The list of component names to compute.
            X (pd.DataFrame): Input data to the pipeline to transform.
            y (pd.Series): The target training data of length [n_samples]
            fit (bool): Whether to fit the estimators as well as transform it.
                        Defaults to False.

        Returns:
            pd.DataFrame or pd.Series - Output of the last component
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        output_cache = {}
        final_component = None
        for component_name in component_list:
            final_component = component_name
            component_class = self.get_component(component_name)
            if not isinstance(component_class, ComponentBase):
                raise ValueError('All components must be instantiated before fitting or predicting')
            x_inputs = []
            y_input = None
            for parent_input in self.get_parents(component_name):
                if parent_input[-2:] == '.y':
                    if y_input is not None:
                        raise ValueError(f'Cannot have multiple `y` parents for a single component {component_name}')
                    y_input = output_cache[parent_input]
                else:
                    try:
                        parent_x = output_cache[parent_input]
                        if isinstance(parent_x, pd.Series):
                            parent_x = pd.DataFrame(parent_x, columns=[parent_input])
                        x_inputs.append(parent_x)
                    except KeyError:
                        x_inputs.append(output_cache[f'{parent_input}.x'])
            input_x, input_y = self._consolidate_inputs(x_inputs, y_input, X, y)
            self.input_feature_names.update({component_name: list(input_x.columns)})
            if isinstance(component_class, Transformer):
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
            else:
                if fit:
                    component_class = component_class.fit(input_x, input_y)
                if not (fit and component_class.name == self.get_last_component().name):  # Don't call predict on the final component during fit
                    output = component_class.predict(input_x)
                else:
                    output = None
                output_cache[component_name] = output
        if final_component is None:
            return X
        final_component_class = self.get_component(final_component)
        if isinstance(final_component_class, Transformer):
            return output_cache[f"{final_component}.x"]
        else:
            return output_cache[final_component]

    @staticmethod
    def _consolidate_inputs(x_inputs, y_input, X, y):
        """ Combines any/all X and y inputs for a component, including handling defaults

        Arguments:
            x_inputs (list(pd.DataFrame)): Data to be used as X input for a component
            y_input (pd.Series, None): If present, the Series to use as y input for a component, different from the original y
            X (pd.DataFrame): The original X input, to be used if there is no parent X input
            y (pd.Series): The original y input, to be used if there is no parent y input

        Returns:
            pd.DataFrame, pd.Series: The X and y transformed values to evaluate a component with
        """
        if len(x_inputs) == 0:
            return_x = X
        else:
            return_x = pd.concat(x_inputs, axis=1)
        return_y = y
        if y_input is not None:
            return_y = y_input
        return return_x, return_y

    def get_component(self, component_name):
        """Retrieves a single component object from the graph.

        Arguments:
            component_name (str): Name of the component to retrieve

        Returns:
            ComponentBase object
        """
        try:
            return self.component_instances[component_name]
        except KeyError:
            raise ValueError(f'Component {component_name} is not in the graph')

    def get_last_component(self):
        """Retrieves the component that is computed last in the graph, usually the final estimator.

        Returns:
            ComponentBase object
        """
        if len(self.compute_order) == 0:
            raise ValueError('Cannot get last component from edgeless graph')
        last_component_name = self.compute_order[-1]
        return self.get_component(last_component_name)

    def get_estimators(self):
        """Gets a list of all the estimator components within this graph

        Returns:
            list: all estimator objects within the graph
        """
        if not isinstance(self.get_last_component(), ComponentBase):
            raise ValueError('Cannot get estimators until the component graph is instantiated')
        return [component_class for component_class in self.component_instances.values() if isinstance(component_class, Estimator)]

    def get_parents(self, component_name):
        """Finds the names of all parent nodes of the given component

        Arguments:
            component_name (str): Name of the child component to look up

        Returns:
            list(str): iterator of parent component names
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
                "To visualize component graphs, a graphviz backend is required.\n" +
                "Install the backend using one of the following commands:\n" +
                "  Mac OS: brew install graphviz\n" +
                "  Linux (Ubuntu): sudo apt-get install graphviz\n" +
                "  Windows: conda install python-graphviz\n"
            )

        graph = graphviz.Digraph(name=name, format=graph_format,
                                 graph_attr={'splines': 'ortho'})
        graph.attr(rankdir='LR')
        for component_name, component_class in self.component_instances.items():
            label = '%s\l' % (component_name)  # noqa: W605
            if isinstance(component_class, ComponentBase):
                parameters = '\l'.join([key + ' : ' + "{:0.2f}".format(val) if (isinstance(val, float))
                                        else key + ' : ' + str(val)
                                        for key, val in component_class.parameters.items()])  # noqa: W605
                label = '%s |%s\l' % (component_name, parameters)  # noqa: W605
            graph.node(component_name, shape='record', label=label)
        edges = self._get_edges()
        graph.edges(edges)
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
        edges = self._get_edges()
        if len(self.component_dict) == 1:
            self.compute_order = list(self.component_dict.keys())
            return
        if len(edges) == 0:
            self.compute_order = []
            return
        digraph = nx.DiGraph()
        digraph.add_edges_from(edges)
        if not nx.is_weakly_connected(digraph):
            raise ValueError('The given graph is not completely connected')
        try:
            self.compute_order = list(topological_sort(digraph))
        except NetworkXUnfeasible:
            raise ValueError('The given graph contains a cycle')

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < len(self.compute_order):
            self._i += 1
            return self.get_component(self.compute_order[self._i - 1])
        else:
            raise StopIteration
