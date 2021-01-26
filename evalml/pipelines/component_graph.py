import networkx as nx
import pandas as pd
import woodwork as ww
from networkx.algorithms.dag import topological_sort
from networkx.exception import NetworkXUnfeasible

from evalml.pipelines.components import ComponentBase, Estimator, Transformer
from evalml.pipelines.components.utils import handle_component_class
from evalml.utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    get_random_seed,
    import_or_raise
)


class ComponentGraph:
    def __init__(self, component_dict=None, random_state=0):
        """ Initializes a component graph for a pipeline as a directed acyclic graph (DAG).

        Example:
            >>> component_dict = {'imputer': ['Imputer'], 'ohe': ['One Hot Encoder', 'imputer.x'], 'estimator_1': ['Random Forest Classifier', 'ohe.x'], 'estimator_2': ['Decision Tree Classifier', 'ohe.x'], 'final': ['Logistic Regression Classifier', 'estimator_1', 'estimator_2']}
            >>> component_graph = ComponentGraph(component_dict)
           """
        self.random_state = get_random_seed(random_state)
        self.component_dict = component_dict or {}
        self.component_instances = {}
        self._is_instantiated = False
        for component_name, component_info in self.component_dict.items():
            if not isinstance(component_info, list):
                raise ValueError('All component information should be passed in as a list')
            component_class = handle_component_class(component_info[0])
            self.component_instances[component_name] = component_class
        self.compute_order = self.generate_order(self.component_dict)
        self.input_feature_names = {}
        self._i = 0

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
        component_instances = {}
        for component_name, component_class in self.component_instances.items():
            component_parameters = parameters.get(component_name, {})

            try:
                new_component = component_class(**component_parameters, random_state=self.random_state)
            except (ValueError, TypeError) as e:
                self._is_instantiated = False
                err = "Error received when instantiating component {} with the following arguments {}".format(component_name, component_parameters)
                raise ValueError(err) from e

            component_instances[component_name] = new_component
        self.component_instances = component_instances
        return self

    def fit(self, X, y):
        """Fit each component in the graph

        Arguments:
            X (ww.DataTable, pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series): The target training data of length [n_samples]
        """
        self._compute_features(self.compute_order, X, y, fit=True)
        return self

    def fit_features(self, X, y):
        """Fit all components save the final one, usually an estimator

        Arguments:
            X (ww.DataTable, pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series): The target training data of length [n_samples]

        Returns:
            ww.DataTable: Transformed values.
        """
        return self._fit_transform_features_helper(True, X, y)

    def compute_final_component_features(self, X, y=None):
        """Transform all components save the final one, and gathers the data from any number of parents
        to get all the information that should be fed to the final component

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series): The target training data of length [n_samples]. Defaults to None.

        Returns:
            ww.DataTable: Transformed values.
        """
        return self._fit_transform_features_helper(False, X, y)

    def _fit_transform_features_helper(self, needs_fitting, X, y=None):
        """Helper function that transforms the input data based on the component graph components.

        Arguments:
            needs_fitting (boolean): Determines if components should be fit.
            X (ww.DataTable, pd.DataFrame): Data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series): The target training data of length [n_samples]. Defaults to None.

        Returns:
            ww.DataTable: Transformed values.
        """
        if len(self.compute_order) <= 1:
            return _convert_to_woodwork_structure(X)
        component_outputs = self._compute_features(self.compute_order[:-1], X, y=y, fit=needs_fitting)
        final_component_inputs = []
        for parent in self.get_parents(self.compute_order[-1]):
            parent_output = component_outputs.get(parent, component_outputs.get(f'{parent}.x'))
            if isinstance(parent_output, ww.DataColumn):
                parent_output = parent_output.to_series()
                parent_output = pd.DataFrame(parent_output, columns=[parent])
                parent_output = _convert_to_woodwork_structure(parent_output)
            final_component_inputs.append(parent_output)
        concatted = pd.concat([component_input.to_dataframe() for component_input in final_component_inputs], axis=1)
        return _convert_to_woodwork_structure(concatted)

    def predict(self, X):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data of shape [n_samples, n_features]

        Returns:
            ww.DataColumn: Predicted values.
        """
        if len(self.compute_order) == 0:
            return _convert_to_woodwork_structure(X)
        final_component = self.compute_order[-1]
        outputs = self._compute_features(self.compute_order, X)
        return _convert_to_woodwork_structure(outputs.get(final_component, outputs.get(f'{final_component}.x')))

    def _compute_features(self, component_list, X, y=None, fit=False):
        """Transforms the data by applying the given components.

        Arguments:
            component_list (list): The list of component names to compute.
            X (ww.DataTable, d.DataFrame): Input data to the pipeline to transform.
            y (ww.DataColumn, pd.Series): The target training data of length [n_samples]
            fit (bool): Whether to fit the estimators as well as transform it.
                        Defaults to False.

        Returns:
            dict: Outputs from each component
        """
        X = _convert_to_woodwork_structure(X)
        if len(component_list) == 0:
            return X
        output_cache = {}
        for component_name in component_list:
            component_instance = self.get_component(component_name)
            if not isinstance(component_instance, ComponentBase):
                raise ValueError('All components must be instantiated before fitting or predicting')
            x_inputs = []
            y_input = None
            for parent_input in self.get_parents(component_name):
                if parent_input[-2:] == '.y':
                    if y_input is not None:
                        raise ValueError(f'Cannot have multiple `y` parents for a single component {component_name}')
                    y_input = output_cache[parent_input]
                else:
                    parent_x = output_cache.get(parent_input, output_cache.get(f'{parent_input}.x'))
                    if isinstance(parent_x, ww.DataTable):
                        parent_x = _convert_woodwork_types_wrapper(parent_x.to_dataframe())
                    elif isinstance(parent_x, ww.DataColumn):
                        parent_x = pd.Series(_convert_woodwork_types_wrapper(parent_x.to_series()), name=parent_input)
                    x_inputs.append(parent_x)
            input_x, input_y = self._consolidate_inputs(x_inputs, y_input, X, y)
            col_intersection = set(X.columns.keys()).intersection(set(input_x.columns.keys()))
            for col in col_intersection:
                if (X[col].logical_type != input_x[col].logical_type and
                        "numeric" not in X[col].semantic_tags):  # numeric is special because we may not be able to safely convert (ex: input is int, output is float)
                    try:
                        input_x = input_x.set_types({col: X[col].logical_type})
                    except TypeError:
                        # if there is a column whose type has been converted s.t. it cannot be converted back, keep as is.
                        # example: StandardScaler could convert a boolean column to a float column. This is expected, and we should not
                        # try to convert back to boolean.
                        continue
            self.input_feature_names.update({component_name: list(input_x.columns)})

            if isinstance(component_instance, Transformer):
                if fit:
                    output = component_instance.fit_transform(input_x, input_y)
                else:
                    output = component_instance.transform(input_x, input_y)
                if isinstance(output, tuple):
                    output_x, output_y = output[0], output[1]
                else:
                    output_x = output
                    output_y = None
                output_cache[f"{component_name}.x"] = output_x
                output_cache[f"{component_name}.y"] = output_y
            else:
                if fit:
                    component_instance.fit(input_x, input_y)
                if not (fit and component_name == self.compute_order[-1]):  # Don't call predict on the final component during fit
                    output = component_instance.predict(input_x)
                else:
                    output = None
                output_cache[component_name] = output
        return output_cache

    @staticmethod
    def _consolidate_inputs(x_inputs, y_input, X, y):
        """Combines any/all X and y inputs for a component, including handling defaults

        Arguments:
            x_inputs (list(pd.DataFrame)): Data to be used as X input for a component
            y_input (pd.Series, None): If present, the Series to use as y input for a component, different from the original y
            X (ww.DataTable, pd.DataFrame): The original X input, to be used if there is no parent X input
            y (ww.DataColumn, pd.Series): The original y input, to be used if there is no parent y input

        Returns:
            ww.DataTable, ww.DataColumn: The X and y transformed values to evaluate a component with
        """
        if len(x_inputs) == 0:
            return_x = X
        else:
            return_x = pd.concat(x_inputs, axis=1)
        return_y = y
        if y_input is not None:
            return_y = y_input
        return_x = _convert_to_woodwork_structure(return_x)
        if return_y is not None:
            return_y = _convert_to_woodwork_structure(return_y)
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
            list: All estimator objects within the graph
        """
        if not isinstance(self.get_last_component(), ComponentBase):
            raise ValueError('Cannot get estimators until the component graph is instantiated')
        return [component_class for component_class in self.component_instances.values() if isinstance(component_class, Estimator)]

    def get_parents(self, component_name):
        """Finds the names of all parent nodes of the given component

        Arguments:
            component_name (str): Name of the child component to look up

        Returns:
            list[str]: Iterator of parent component names
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
        edges = self._get_edges(self.component_dict)
        graph.edges(edges)
        return graph

    @staticmethod
    def _get_edges(component_dict):
        edges = []
        for component_name, component_info in component_dict.items():
            if len(component_info) > 1:
                for parent in component_info[1:]:
                    if parent[-2:] == '.x' or parent[-2:] == '.y':
                        parent = parent[:-2]
                    edges.append((parent, component_name))
        return edges

    @classmethod
    def generate_order(cls, component_dict):
        """Regenerated the topologically sorted order of the graph"""
        edges = cls._get_edges(component_dict)
        if len(component_dict) == 1:
            return list(component_dict.keys())
        if len(edges) == 0:
            return []
        digraph = nx.DiGraph()
        digraph.add_edges_from(edges)
        if not nx.is_weakly_connected(digraph):
            raise ValueError('The given graph is not completely connected')
        try:
            compute_order = list(topological_sort(digraph))
        except NetworkXUnfeasible:
            raise ValueError('The given graph contains a cycle')
        end_components = [component for component in compute_order if len(nx.descendants(digraph, component)) == 0]
        if len(end_components) != 1:
            raise ValueError('The given graph has more than one final (childless) component')
        return compute_order

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_component(self.compute_order[index])
        else:
            return self.get_component(index)

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        """Iterator for graphs, retrieves the components in the graph in order

        Returns:
            ComponentBase obj: The next component class or instance in the graph
        """
        if self._i < len(self.compute_order):
            self._i += 1
            return self.get_component(self.compute_order[self._i - 1])
        else:
            self._i = 0
            raise StopIteration
