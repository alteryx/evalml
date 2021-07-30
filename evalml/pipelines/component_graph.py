import warnings

import networkx as nx
import pandas as pd
import woodwork as ww
from networkx.algorithms.dag import topological_sort
from networkx.exception import NetworkXUnfeasible

from evalml.exceptions.exceptions import (
    MissingComponentError,
    ParameterNotUsedWarning,
)
from evalml.pipelines.components import ComponentBase, Estimator, Transformer
from evalml.pipelines.components.transformers.transformer import (
    TargetTransformer,
)
from evalml.pipelines.components.utils import handle_component_class
from evalml.utils import get_logger, import_or_raise, infer_feature_types

logger = get_logger(__file__)


class ComponentGraph:
    """Component graph for a pipeline as a directed acyclic graph (DAG).

    Arguments:
        component_dict (dict): A dictionary which specifies the components and edges between components that should be used to create the component graph. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Example:
        >>> component_dict = {'imputer': ['Imputer'], 'ohe': ['One Hot Encoder', 'imputer.x'], 'estimator_1': ['Random Forest Classifier', 'ohe.x'], 'estimator_2': ['Decision Tree Classifier', 'ohe.x'], 'final': ['Logistic Regression Classifier', 'estimator_1', 'estimator_2']}
        >>> component_graph = ComponentGraph(component_dict)
    """

    def __init__(self, component_dict=None, random_seed=0):
        self.random_seed = random_seed
        self.component_dict = component_dict or {}
        if not isinstance(self.component_dict, dict):
            raise ValueError(
                "component_dict must be a dictionary which specifies the components and edges between components"
            )
        self._validate_component_dict()
        self.component_instances = {}
        self._is_instantiated = False
        for component_name, component_info in self.component_dict.items():
            component_class = handle_component_class(component_info[0])
            self.component_instances[component_name] = component_class
        self.input_feature_names = {}
        self._feature_provenance = {}
        self._i = 0
        self._compute_order = self.generate_order(self.component_dict)

    def _validate_component_dict(self):
        for _, component_inputs in self.component_dict.items():
            if not isinstance(component_inputs, list):
                raise ValueError(
                    "All component information should be passed in as a list"
                )
            component_inputs = component_inputs[1:]
            has_feature_input = any(
                component_input.endswith(".x") or component_input == "X"
                for component_input in component_inputs
            )
            has_target_input = any(
                component_input.endswith(".y") or component_input == "y"
                for component_input in component_inputs
            )
            if not (has_feature_input and has_target_input):
                raise ValueError(
                    "All edges must be specified as either an input feature (.x) or input target (.y)."
                )

    @property
    def compute_order(self):
        """The order that components will be computed or called in."""
        return self._compute_order

    @property
    def default_parameters(self):
        """The default parameter dictionary for this pipeline.

        Returns:
            dict: Dictionary of all component default parameters.
        """
        defaults = {}
        for component in self.component_instances.values():
            if component.default_parameters:
                defaults[component.name] = component.default_parameters
        return defaults

    def instantiate(self, parameters):
        """Instantiates all uninstantiated components within the graph using the given parameters. An error will be
        raised if a component is already instantiated but the parameters dict contains arguments for that component.

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                               An empty dictionary {} or None implies using all default values for component parameters.
        """
        if self._is_instantiated:
            raise ValueError(
                f"Cannot reinstantiate a component graph that was previously instantiated"
            )
        parameters = parameters or {}
        param_set = set(s for s in parameters.keys() if s not in ["pipeline"])
        diff = param_set.difference(set(self.component_instances.keys()))
        if len(diff):
            warnings.warn(ParameterNotUsedWarning(diff))
        self._is_instantiated = True
        component_instances = {}
        for component_name, component_class in self.component_instances.items():
            component_parameters = parameters.get(component_name, {})
            try:
                new_component = component_class(
                    **component_parameters, random_seed=self.random_seed
                )
            except (ValueError, TypeError) as e:
                self._is_instantiated = False
                err = "Error received when instantiating component {} with the following arguments {}".format(
                    component_name, component_parameters
                )
                raise ValueError(err) from e

            component_instances[component_name] = new_component
        self.component_instances = component_instances
        return self

    def fit(self, X, y):
        """Fit each component in the graph

        Arguments:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (pd.Series): The target training data of length [n_samples]
        """
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        self._compute_features(self.compute_order, X, y, fit=True)
        self._feature_provenance = self._get_feature_provenance(X.columns)
        return self

    def fit_features(self, X, y):
        """Fit all components save the final one, usually an estimator

        Arguments:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (pd.Series): The target training data of length [n_samples]

        Returns:
            pd.DataFrame: Transformed values.
        """
        return self._fit_transform_features_helper(True, X, y)

    def compute_final_component_features(self, X, y=None):
        """Transform all components save the final one, and gathers the data from any number of parents
        to get all the information that should be fed to the final component

        Arguments:
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            y (pd.Series): The target training data of length [n_samples]. Defaults to None.

        Returns:
            pd.DataFrame: Transformed values.
        """
        return self._fit_transform_features_helper(False, X, y)

    def _fit_transform_features_helper(self, needs_fitting, X, y=None):
        """Helper function that transforms the input data based on the component graph components.

        Arguments:
            needs_fitting (boolean): Determines if components should be fit.
            X (pd.DataFrame): Data of shape [n_samples, n_features]
            y (pd.Series): The target training data of length [n_samples]. Defaults to None.

        Returns:
            pd.DataFrame: Transformed values.
        """
        if len(self.compute_order) <= 1:
            X = infer_feature_types(X)
            self.input_feature_names.update({self.compute_order[0]: list(X.columns)})
            return X
        component_outputs = self._compute_features(
            self.compute_order[:-1], X, y=y, fit=needs_fitting
        )
        final_component_inputs = []

        parent_inputs = [
            parent_input
            for parent_input in self.get_inputs(self.compute_order[-1])
            if parent_input[-2:] != ".y"
        ]
        for parent in parent_inputs:
            parent_output = component_outputs.get(
                parent, component_outputs.get(f"{parent}.x")
            )
            if isinstance(parent_output, pd.Series):
                parent_output = pd.DataFrame(parent_output, columns=[parent])
                parent_output = infer_feature_types(parent_output)
            if parent_output is not None:
                final_component_inputs.append(parent_output)
        concatted = ww.utils.concat_columns(
            [component_input for component_input in final_component_inputs]
        )
        if needs_fitting:
            self.input_feature_names.update(
                {self.compute_order[-1]: list(concatted.columns)}
            )
        return concatted

    def predict(self, X):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame): Data of shape [n_samples, n_features]

        Returns:
            pd.Series: Predicted values.
        """
        if len(self.compute_order) == 0:
            return infer_feature_types(X)
        final_component = self.compute_order[-1]
        outputs = self._compute_features(self.compute_order, X)
        return infer_feature_types(
            outputs.get(final_component, outputs.get(f"{final_component}.x"))
        )

    def _compute_features(self, component_list, X, y=None, fit=False):
        """Transforms the data by applying the given components.

        Arguments:
            component_list (list): The list of component names to compute.
            X (pd.DataFrame): Input data to the pipeline to transform.
            y (pd.Series): The target training data of length [n_samples]
            fit (bool): Whether to fit the estimators as well as transform it.
                        Defaults to False.

        Returns:
            dict: Outputs from each component
        """
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        most_recent_y = y
        if len(component_list) == 0:
            return X
        output_cache = {}
        for component_name in component_list:
            component_instance = self.get_component(component_name)
            if not isinstance(component_instance, ComponentBase):
                raise ValueError(
                    "All components must be instantiated before fitting or predicting"
                )
            x_inputs = []
            y_input = None

            for parent_input in self.get_inputs(component_name):
                if parent_input[-2:] == ".y" or parent_input == "y":
                    if y_input is not None:
                        raise ValueError(
                            f"Cannot have multiple `y` parents for a single component {component_name}"
                        )
                    y_input = (
                        output_cache[parent_input] if parent_input[-2:] == ".y" else y
                    )

                else:
                    if parent_input == "X":
                        x_inputs.append(X)
                    else:
                        parent_x = output_cache.get(
                            parent_input, output_cache.get(f"{parent_input}.x")
                        )
                        if isinstance(parent_x, pd.Series):
                            parent_x = parent_x.rename(parent_input)
                        x_inputs.append(parent_x)
            input_x, input_y = self._consolidate_inputs(
                x_inputs, y_input, X, most_recent_y
            )
            self.input_feature_names.update({component_name: list(input_x.columns)})
            if isinstance(component_instance, Transformer):
                if fit:
                    output = component_instance.fit_transform(input_x, input_y)
                else:
                    output = component_instance.transform(input_x, input_y)
                if isinstance(output, tuple):
                    output_x, output_y = output[0], output[1]
                    most_recent_y = output_y
                else:
                    output_x = output
                    output_y = None
                output_cache[f"{component_name}.x"] = output_x
                output_cache[f"{component_name}.y"] = output_y
            else:
                if fit:
                    component_instance.fit(input_x, input_y)
                if not (
                    fit and component_name == self.compute_order[-1]
                ):  # Don't call predict on the final component during fit
                    output = component_instance.predict(input_x)
                else:
                    output = None
                output_cache[f"{component_name}.x"] = output
        return output_cache

    def _get_feature_provenance(self, input_feature_names):
        """Get the feature provenance for each feature in the input_feature_names.

        The provenance is a mapping from the original feature names in the dataset to a list of
        features that were created from that original feature.

        For example, after fitting a OHE on a feature called 'cats', with categories 'a' and 'b', the
        provenance would have the following entry: {'cats': ['a', 'b']}.

        If a feature is then calculated from feature 'a', e.g. 'a_squared', then the provenance would instead
        be {'cats': ['a', 'a_squared', 'b']}.

        Arguments:
            input_feature_names (list(str)): Names of the features in the input dataframe.

        Returns:
           dictionary: mapping of feature name to set feature names that were created from that feature.
        """
        if not self.compute_order:
            return {}

        # Every feature comes from one of the original features so
        # each one starts with an empty set
        provenance = {col: set([]) for col in input_feature_names}

        transformers = filter(
            lambda c: isinstance(c, Transformer),
            [self.get_component(c) for c in self.compute_order],
        )
        for component_instance in transformers:
            component_provenance = component_instance._get_feature_provenance()
            for component_input, component_output in component_provenance.items():

                # Case 1: The transformer created features from one of the original features
                if component_input in provenance:
                    provenance[component_input] = provenance[component_input].union(
                        set(component_output)
                    )

                # Case 2: The transformer created features from a feature created from an original feature.
                # Add it to the provenance of the original feature it was created from
                else:
                    for in_feature, out_feature in provenance.items():
                        if component_input in out_feature:
                            provenance[in_feature] = out_feature.union(
                                set(component_output)
                            )

        # Get rid of features that are not in the dataset the final estimator uses
        final_estimator_features = set(
            self.input_feature_names.get(self.compute_order[-1], [])
        )
        for feature in provenance:
            provenance[feature] = provenance[feature].intersection(
                final_estimator_features
            )

        # Delete features that weren't used to create other features
        return {
            feature: children
            for feature, children in provenance.items()
            if len(children)
        }

    @staticmethod
    def _consolidate_inputs(x_inputs, y_input, X, y):
        """Combines any/all X and y inputs for a component, including handling defaults

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
            return_x = ww.concat_columns(x_inputs)
        return_y = y
        if y_input is not None:
            return_y = y_input

        if return_y is not None:
            return_y = infer_feature_types(return_y)
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
            raise ValueError(f"Component {component_name} is not in the graph")

    def get_last_component(self):
        """Retrieves the component that is computed last in the graph, usually the final estimator.

        Returns:
            ComponentBase object
        """
        if len(self.compute_order) == 0:
            raise ValueError("Cannot get last component from edgeless graph")
        last_component_name = self.compute_order[-1]
        return self.get_component(last_component_name)

    def get_estimators(self):
        """Gets a list of all the estimator components within this graph

        Returns:
            list: All estimator objects within the graph
        """
        if not isinstance(self.get_last_component(), ComponentBase):
            raise ValueError(
                "Cannot get estimators until the component graph is instantiated"
            )
        return [
            component_class
            for component_class in self.component_instances.values()
            if isinstance(component_class, Estimator)
        ]

    def get_inputs(self, component_name):
        """Retrieves all inputs for a given component.

        Arguments:
            component_name (str): Name of the component to look up.

        Returns:
            list[str]: List of inputs for the component to use.
        """
        try:
            component_info = self.component_dict[component_name]
        except KeyError:
            raise ValueError(f"Component {component_name} not in the graph")
        if len(component_info) > 1:
            return component_info[1:]
        return []

    def describe(self, return_dict=False):
        """Outputs component graph details including component parameters

        Arguments:
            return_dict (bool): If True, return dictionary of information about component graph. Defaults to False.

        Returns:
            dict: Dictionary of all component parameters if return_dict is True, else None
        """
        components = {}
        for number, component in enumerate(self.component_instances.values(), 1):
            component_string = str(number) + ". " + component.name
            logger.info(component_string)
            components.update(
                {
                    component.name: component.describe(
                        print_name=False, return_dict=return_dict
                    )
                }
            )
        if return_dict:
            return components

    def graph(self, name=None, graph_format=None):
        """Generate an image representing the component graph

        Arguments:
            name (str): Name of the graph. Defaults to None.
            graph_format (str): file format to save the graph in. Defaults to None.

        Returns:
            graphviz.Digraph: Graph object that can be directly displayed in Jupyter notebooks.
        """
        graphviz = import_or_raise(
            "graphviz", error_msg="Please install graphviz to visualize pipelines."
        )

        # Try rendering a dummy graph to see if a working backend is installed
        try:
            graphviz.Digraph().pipe()
        except graphviz.backend.ExecutableNotFound:
            raise RuntimeError(
                "To visualize component graphs, a graphviz backend is required.\n"
                + "Install the backend using one of the following commands:\n"
                + "  Mac OS: brew install graphviz\n"
                + "  Linux (Ubuntu): sudo apt-get install graphviz\n"
                + "  Windows: conda install python-graphviz\n"
            )

        graph = graphviz.Digraph(
            name=name, format=graph_format, graph_attr={"splines": "ortho"}
        )
        graph.attr(rankdir="LR")
        for component_name, component_class in self.component_instances.items():
            label = "%s\l" % (component_name)  # noqa: W605
            if isinstance(component_class, ComponentBase):
                parameters = "\l".join(
                    [
                        key + " : " + "{:0.2f}".format(val)
                        if (isinstance(val, float))
                        else key + " : " + str(val)
                        for key, val in component_class.parameters.items()
                    ]
                )  # noqa: W605
                label = "%s |%s\l" % (component_name, parameters)  # noqa: W605
            graph.node(component_name, shape="record", label=label)
        edges = self._get_edges(self.component_dict)
        graph.edges(edges)
        return graph

    @staticmethod
    def _get_edges(component_dict):
        edges = []
        for component_name, component_info in component_dict.items():
            if len(component_info) > 1:
                for parent in component_info[1:]:
                    if parent == "X" or parent == "y":
                        continue
                    elif parent[-2:] == ".x" or parent[-2:] == ".y":
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
        digraph.add_nodes_from(list(component_dict.keys()))
        digraph.add_edges_from(edges)
        if not nx.is_weakly_connected(digraph):
            raise ValueError("The given graph is not completely connected")
        try:
            compute_order = list(topological_sort(digraph))
        except NetworkXUnfeasible:
            raise ValueError("The given graph contains a cycle")
        end_components = [
            component
            for component in compute_order
            if len(nx.descendants(digraph, component)) == 0
        ]
        if len(end_components) != 1:
            raise ValueError(
                "The given graph has more than one final (childless) component"
            )
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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        random_seed_eq = self.random_seed == other.random_seed
        if not random_seed_eq:
            return False
        attributes_to_check = ["component_dict", "compute_order"]
        for attribute in attributes_to_check:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def __repr__(self):
        component_strs = []
        for (
            component_name,
            component_info,
        ) in self.component_dict.items():
            try:
                component_key = f"'{component_name}': "
                if isinstance(component_info[0], str):
                    component_class = handle_component_class(component_info[0])
                else:
                    component_class = handle_component_class(component_info[0].name)
                component_name = f"'{component_class.name}'"
            except MissingComponentError:
                # Not an EvalML component, use component class name
                component_name = f"{component_info[0].__name__}"

            component_edges_str = ""
            if len(component_info) > 1:
                component_edges_str = ", "
                component_edges_str += ", ".join(
                    [f"'{info}'" for info in component_info[1:]]
                )

            component_str = f"{component_key}[{component_name}{component_edges_str}]"
            component_strs.append(component_str)
        component_dict_str = f"{{{', '.join(component_strs)}}}"
        return component_dict_str

    def _get_parent_y(self, component_name):
        """Helper for inverse_transform method."""
        parents = self.get_inputs(component_name)
        return next(iter(p[:-2] for p in parents if ".y" in p), None)

    def inverse_transform(self, y):
        """Apply component inverse_transform methods to estimator predictions in reverse order.

        Components that implement inverse_transform are PolynomialDetrender, LabelEncoder (tbd).

        Arguments:
            y: (pd.Series): Final component features
        """
        data_to_transform = infer_feature_types(y)
        current_component = self.compute_order[-1]
        has_incoming_y_from_parent = True
        while has_incoming_y_from_parent:
            parent_y = self._get_parent_y(current_component)
            if parent_y:
                component = self.get_component(parent_y)
                if isinstance(component, TargetTransformer):
                    data_to_transform = component.inverse_transform(data_to_transform)
                current_component = parent_y
            else:
                has_incoming_y_from_parent = False

        return data_to_transform
