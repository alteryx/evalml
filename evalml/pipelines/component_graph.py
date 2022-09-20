"""Component graph for a pipeline as a directed acyclic graph (DAG)."""
import inspect
import warnings

import networkx as nx
import pandas as pd
import woodwork as ww
from networkx.algorithms.dag import topological_sort
from networkx.exception import NetworkXUnfeasible

from evalml.exceptions.exceptions import (
    MethodPropertyNotFoundError,
    MissingComponentError,
    ParameterNotUsedWarning,
    PipelineError,
    PipelineErrorCodeEnum,
)
from evalml.pipelines.components import ComponentBase, Estimator, Transformer
from evalml.pipelines.components.utils import handle_component_class
from evalml.utils import (
    _schema_is_equal,
    get_logger,
    import_or_raise,
    infer_feature_types,
)

logger = get_logger(__file__)


class ComponentGraph:
    """Component graph for a pipeline as a directed acyclic graph (DAG).

    Args:
        component_dict (dict): A dictionary which specifies the components and edges between components that should be used to create the component graph. Defaults to None.
        cached_data (dict): A dictionary of nested cached data. If the hashes and components are in this cache, we skip fitting for these components. Expected to be of format
            {hash1: {component_name: trained_component, ...}, hash2: {...}, ...}.
            Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.

    Examples:
        >>> component_dict = {'Imputer': ['Imputer', 'X', 'y'],
        ...                   'Logistic Regression': ['Logistic Regression Classifier', 'Imputer.x', 'y']}
        >>> component_graph = ComponentGraph(component_dict)
        >>> assert component_graph.compute_order == ['Imputer', 'Logistic Regression']
        ...
        ...
        >>> component_dict = {'Imputer': ['Imputer', 'X', 'y'],
        ...                   'OHE': ['One Hot Encoder', 'Imputer.x', 'y'],
        ...                   'estimator_1': ['Random Forest Classifier', 'OHE.x', 'y'],
        ...                   'estimator_2': ['Decision Tree Classifier', 'OHE.x', 'y'],
        ...                   'final': ['Logistic Regression Classifier', 'estimator_1.x', 'estimator_2.x', 'y']}
        >>> component_graph = ComponentGraph(component_dict)

        The default parameters for every component in the component graph.

        >>> assert component_graph.default_parameters == {
        ...     'Imputer': {'categorical_impute_strategy': 'most_frequent',
        ...                 'numeric_impute_strategy': 'mean',
        ...                 'boolean_impute_strategy': 'most_frequent',
        ...                 'categorical_fill_value': None,
        ...                 'numeric_fill_value': None,
        ...                 'boolean_fill_value': None},
        ...     'One Hot Encoder': {'top_n': 10,
        ...                         'features_to_encode': None,
        ...                         'categories': None,
        ...                         'drop': 'if_binary',
        ...                         'handle_unknown': 'ignore',
        ...                         'handle_missing': 'error'},
        ...     'Random Forest Classifier': {'n_estimators': 100,
        ...                                  'max_depth': 6,
        ...                                  'n_jobs': -1},
        ...     'Decision Tree Classifier': {'criterion': 'gini',
        ...                                  'max_features': 'auto',
        ...                                  'max_depth': 6,
        ...                                  'min_samples_split': 2,
        ...                                  'min_weight_fraction_leaf': 0.0},
        ...     'Logistic Regression Classifier': {'penalty': 'l2',
        ...                                        'C': 1.0,
        ...                                        'n_jobs': -1,
        ...                                        'multi_class': 'auto',
        ...                                        'solver': 'lbfgs'}}

    """

    def __init__(self, component_dict=None, cached_data=None, random_seed=0):
        self.random_seed = random_seed
        self.component_dict = component_dict or {}
        if not isinstance(self.component_dict, dict):
            raise ValueError(
                "component_dict must be a dictionary which specifies the components and edges between components",
            )
        self._validate_component_dict()
        self.cached_data = cached_data
        self.component_instances = {}
        self._is_instantiated = False
        for component_name, component_info in self.component_dict.items():
            component_class = handle_component_class(component_info[0])
            self.component_instances[component_name] = component_class

        self._validate_component_dict_edges()

        self.input_feature_names = {}
        self._feature_provenance = {}
        self._feature_logical_types = {}
        self._i = 0
        self._compute_order = self.generate_order(self.component_dict)
        self._input_types = {}

    def _validate_component_dict(self):
        for _, component_inputs in self.component_dict.items():
            if not isinstance(component_inputs, list):
                raise ValueError(
                    "All component information should be passed in as a list",
                )

    def _validate_component_dict_edges(self):
        for _, component_inputs in self.component_dict.items():
            component_inputs = component_inputs[1:]
            has_feature_input = any(
                component_input.endswith(".x") or component_input == "X"
                for component_input in component_inputs
            )
            num_target_inputs = sum(
                component_input.endswith(".y") or component_input == "y"
                for component_input in component_inputs
            )
            if not has_feature_input:
                raise ValueError(
                    "All components must have at least one input feature (.x/X) edge.",
                )
            if num_target_inputs != 1:
                raise ValueError(
                    "All components must have exactly one target (.y/y) edge.",
                )

            def check_all_inputs_have_correct_syntax(edge):
                return not (
                    edge.endswith(".y")
                    or edge == "y"
                    or edge.endswith(".x")
                    or edge == "X"
                )

            if (
                len(
                    list(
                        filter(check_all_inputs_have_correct_syntax, component_inputs),
                    ),
                )
                != 0
            ):
                raise ValueError(
                    "All edges must be specified as either an input feature ('X'/.x) or input target ('y'/.y).",
                )

            target_inputs = [
                component
                for component in component_inputs
                if (component.endswith(".y"))
            ]
            if target_inputs:
                target_component_name = target_inputs[0][:-2]
                target_component_class = self.get_component(target_component_name)
                if not target_component_class.modifies_target:
                    raise ValueError(
                        f"{target_inputs[0]} is not a valid input edge because {target_component_name} does not return a target.",
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

    def instantiate(self, parameters=None):
        """Instantiates all uninstantiated components within the graph using the given parameters. An error will be raised if a component is already instantiated but the parameters dict contains arguments for that component.

        Args:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                An empty dictionary {} or None implies using all default values for component parameters. If a component
                in the component graph is already instantiated, it will not use any of its parameters defined in this dictionary. Defaults to None.

        Returns:
            self

        Raises:
            ValueError: If component graph is already instantiated or if a component errored while instantiating.
        """
        if self._is_instantiated:
            raise ValueError(
                f"Cannot reinstantiate a component graph that was previously instantiated",
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
            if inspect.isclass(component_class):
                try:
                    new_component = component_class(
                        **component_parameters, random_seed=self.random_seed
                    )
                except (ValueError, TypeError) as e:
                    self._is_instantiated = False
                    err = "Error received when instantiating component {} with the following arguments {}".format(
                        component_name,
                        component_parameters,
                    )
                    raise ValueError(err) from e
                component_instances[component_name] = new_component
            elif isinstance(component_class, ComponentBase):
                component_instances[component_name] = component_class
        self.component_instances = component_instances
        return self

    def fit(self, X, y):
        """Fit each component in the graph.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self
        """
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        self._transform_features(self.compute_order, X, y, fit=True)
        self._feature_provenance = self._get_feature_provenance(X.columns)
        return self

    def fit_transform(self, X, y):
        """Fit and transform all components in the component graph, if all components are Transformers.

        Args:
            X (pd.DataFrame): Input features of shape [n_samples, n_features].
            y (pd.Series): The target data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed output.

        Raises:
            ValueError: If final component is an Estimator.
        """
        final_component_instance = self.get_last_component()
        if isinstance(final_component_instance, Estimator):
            raise ValueError(
                "Cannot call fit_transform() on a component graph because the final component is an Estimator. Use fit_and_transform_all_but_final instead.",
            )
        return self.fit(X, y).transform(X, y)

    def fit_and_transform_all_but_final(self, X, y):
        """Fit and transform all components save the final one, usually an estimator.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            Tuple (pd.DataFrame, pd.Series): Transformed features and target.
        """
        return self._fit_transform_features_helper(True, X, y)

    def transform_all_but_final(self, X, y=None):
        """Transform all components save the final one, and gathers the data from any number of parents to get all the information that should be fed to the final component.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples]. Defaults to None.

        Returns:
            pd.DataFrame: Transformed values.
        """
        features, _ = self._fit_transform_features_helper(False, X, y)
        return features

    def _fit_transform_features_helper(self, needs_fitting, X, y=None):
        """Transform (and possibly fit) all components save the final one, and returns the data that should be fed to the final component, usually an estimator.

        Args:
            needs_fitting (boolean): Determines if components should be fit.
            X (pd.DataFrame): Data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples]. Defaults to None.

        Returns:
           Tuple: pd.DataFrame, pd.Series: Transformed features and target.
        """
        if len(self.compute_order) <= 1:
            X = infer_feature_types(X)
            self.input_feature_names.update({self.compute_order[0]: list(X.columns)})
            return X, y
        component_outputs = self._transform_features(
            self.compute_order[:-1],
            X,
            y=y,
            fit=needs_fitting,
            evaluate_training_only_components=needs_fitting,
        )
        x_inputs, y_output = self._consolidate_inputs_for_component(
            component_outputs,
            self.compute_order[-1],
            X,
            y,
        )
        if needs_fitting:
            self.input_feature_names.update(
                {self.compute_order[-1]: list(x_inputs.columns)},
            )
        return x_inputs, y_output

    def _consolidate_inputs_for_component(
        self,
        component_outputs,
        component,
        X,
        y=None,
    ):
        x_inputs = []
        y_input = None
        for parent_input in self.get_inputs(component):
            if parent_input == "y":
                y_input = y
            elif parent_input == "X":
                x_inputs.append(X)
            elif parent_input.endswith(".y"):
                y_input = component_outputs[parent_input]
            elif parent_input.endswith(".x"):
                parent_x = component_outputs[parent_input]
                if isinstance(parent_x, pd.Series):
                    parent_x = parent_x.rename(parent_input)
                x_inputs.append(parent_x)
        x_inputs = ww.concat_columns(x_inputs)
        return x_inputs, y_input

    def transform(self, X, y=None):
        """Transform the input using the component graph.

        Args:
            X (pd.DataFrame): Input features of shape [n_samples, n_features].
            y (pd.Series): The target data of length [n_samples]. Defaults to None.

        Returns:
            pd.DataFrame: Transformed output.

        Raises:
            ValueError: If final component is not a Transformer.
        """
        if len(self.compute_order) == 0:
            return infer_feature_types(X)
        final_component_name = self.compute_order[-1]
        final_component_instance = self.get_last_component()
        if not isinstance(final_component_instance, Transformer):
            raise ValueError(
                "Cannot call transform() on a component graph because the final component is not a Transformer.",
            )

        outputs = self._transform_features(
            self.compute_order,
            X,
            y,
            fit=False,
            evaluate_training_only_components=True,
        )
        output_x = infer_feature_types(outputs.get(f"{final_component_name}.x"))
        output_y = outputs.get(f"{final_component_name}.y", None)
        if output_y is not None:
            return output_x, output_y
        return output_x

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Input features of shape [n_samples, n_features].

        Returns:
            pd.Series: Predicted values.

        Raises:
            ValueError: If final component is not an Estimator.
        """
        if len(self.compute_order) == 0:
            return infer_feature_types(X)
        final_component = self.compute_order[-1]
        final_component_instance = self.get_last_component()
        if not isinstance(final_component_instance, Estimator):
            raise ValueError(
                "Cannot call predict() on a component graph because the final component is not an Estimator.",
            )
        outputs = self._transform_features(
            self.compute_order,
            X,
            evaluate_training_only_components=False,
        )
        return infer_feature_types(outputs.get(f"{final_component}.x"))

    def _return_non_engineered_features(self, X):
        base_features = [
            c
            for c in X.ww.columns
            if X.ww[c].ww.origin == "base" or X.ww[c].ww.origin is None
        ]
        return X.ww[base_features]

    def _transform_features(
        self,
        component_list,
        X,
        y=None,
        fit=False,
        evaluate_training_only_components=False,
    ):
        """Transforms the data by applying the given components.

        Args:
            component_list (list): The list of component names to compute.
            X (pd.DataFrame): Input data to the pipeline to transform.
            y (pd.Series): The target training data of length [n_samples].
            fit (boolean): Whether to fit the estimators as well as transform it. Defaults to False.
            evaluate_training_only_components (boolean): Whether to evaluate training-only components (such as the samplers) during transform or predict. Defaults to False.

        Returns:
            dict: Outputs from each component.

        Raises:
            PipelineError: if input data types are different from the input types the pipeline was fitted on
        """
        X = infer_feature_types(X)
        if not fit:
            X_schema = (
                self._return_non_engineered_features(X).ww.schema
                if "DFS Transformer" in self.compute_order
                else X.ww.schema
            )
            if not _schema_is_equal(X_schema, self._input_types):
                raise PipelineError(
                    "Input X data types are different from the input types the pipeline was fitted on.",
                    code=PipelineErrorCodeEnum.PREDICT_INPUT_SCHEMA_UNEQUAL,
                    details={
                        "input_features_types": X_schema.types,
                        "pipeline_features_types": self._input_types.types,
                    },
                )
        else:
            self._input_types = (
                self._return_non_engineered_features(X).ww.schema
                if "DFS Transformer" in self.compute_order
                else X.ww.schema
            )

        if y is not None:
            y = infer_feature_types(y)

        if len(component_list) == 0:
            return X

        hashes = None
        if self.cached_data is not None:
            hashes = hash(tuple(X.index))

        output_cache = {}
        for component_name in component_list:
            component_instance = self._get_component_from_cache(
                hashes,
                component_name,
                fit,
            )
            if not isinstance(component_instance, ComponentBase):
                raise ValueError(
                    "All components must be instantiated before fitting or predicting",
                )
            x_inputs, y_input = self._consolidate_inputs_for_component(
                output_cache,
                component_name,
                X,
                y,
            )
            self.input_feature_names.update({component_name: list(x_inputs.columns)})
            self._feature_logical_types[component_name] = x_inputs.ww.logical_types
            if isinstance(component_instance, Transformer):
                if fit:
                    if component_instance._is_fitted:
                        output = component_instance.transform(x_inputs, y_input)
                    else:
                        output = component_instance.fit_transform(x_inputs, y_input)
                elif (
                    component_instance.training_only
                    and evaluate_training_only_components is False
                ):
                    output = x_inputs, y_input
                else:
                    output = component_instance.transform(x_inputs, y_input)

                if isinstance(output, tuple):
                    output_x, output_y = output[0], output[1]
                else:
                    output_x = output
                    output_y = None
                output_cache[f"{component_name}.x"] = output_x
                output_cache[f"{component_name}.y"] = output_y
            else:
                if fit and not component_instance._is_fitted:
                    component_instance.fit(x_inputs, y_input)
                if fit and component_name == self.compute_order[-1]:
                    # Don't call predict on the final component during fit
                    output = None
                elif component_name != self.compute_order[-1]:
                    try:
                        output = component_instance.predict_proba(x_inputs)
                        if isinstance(output, pd.DataFrame):
                            if len(output.columns) == 2:
                                # If it is a binary problem, drop the first column since both columns are colinear
                                output = output.ww.drop(output.columns[0])
                            output = output.ww.rename(
                                {
                                    col: f"Col {str(col)} {component_name}.x"
                                    for col in output.columns
                                },
                            )
                    except MethodPropertyNotFoundError:
                        output = component_instance.predict(x_inputs)
                else:
                    output = component_instance.predict(x_inputs)
                output_cache[f"{component_name}.x"] = output
            if self.cached_data is not None and fit:
                self.component_instances[component_name] = component_instance

        return output_cache

    def _get_component_from_cache(self, hashes, component_name, fit):
        """Gets either the stacked ensemble component or the component from component_instances."""
        component_instance = self.get_component(component_name)
        if self.cached_data is not None and fit:
            try:
                component_instance = self.cached_data[hashes][component_name]
            except KeyError:
                pass
        return component_instance

    def _get_feature_provenance(self, input_feature_names):
        """Get the feature provenance for each feature in the input_feature_names.

        The provenance is a mapping from the original feature names in the dataset to a list of
        features that were created from that original feature.

        For example, after fitting a OHE on a feature called 'cats', with categories 'a' and 'b', the
        provenance would have the following entry: {'cats': ['a', 'b']}.

        If a feature is then calculated from feature 'a', e.g. 'a_squared', then the provenance would instead
        be {'cats': ['a', 'a_squared', 'b']}.

        Args:
            input_feature_names (list(str)): Names of the features in the input dataframe.

        Returns:
           dict: Dictionary mapping of feature name to set feature names that were created from that feature.
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
                        set(component_output),
                    )

                # Case 2: The transformer created features from a feature created from an original feature.
                # Add it to the provenance of the original feature it was created from
                else:
                    for in_feature, out_feature in provenance.items():
                        if component_input in out_feature:
                            provenance[in_feature] = out_feature.union(
                                set(component_output),
                            )

        # Get rid of features that are not in the dataset the final estimator uses
        final_estimator_features = set(
            self.input_feature_names.get(self.compute_order[-1], []),
        )
        for feature in provenance:
            provenance[feature] = provenance[feature].intersection(
                final_estimator_features,
            )

        # Delete features that weren't used to create other features
        return {
            feature: children
            for feature, children in provenance.items()
            if len(children)
        }

    def get_component_input_logical_types(self, component_name):
        """Get the logical types that are passed to the given component.

        Args:
            component_name (str): Name of component in the graph

        Returns:
            Dict - Mapping feature name to logical type instance.

        Raises:
            ValueError: If the component is not in the graph.
            ValueError: If the component graph as not been fitted
        """
        if not self._feature_logical_types:
            raise ValueError("Component Graph has not been fit.")
        if component_name not in self._feature_logical_types:
            raise ValueError(f"Component {component_name} is not in the graph")

        return self._feature_logical_types[component_name]

    @property
    def last_component_input_logical_types(self):
        """Get the logical types that are passed to the last component in the pipeline.

        Returns:
            Dict - Mapping feature name to logical type instance.

        Raises:
            ValueError: If the component is not in the graph.
            ValueError: If the component graph as not been fitted
        """
        return self.get_component_input_logical_types(self.compute_order[-1])

    def get_component(self, component_name):
        """Retrieves a single component object from the graph.

        Args:
            component_name (str): Name of the component to retrieve

        Returns:
            ComponentBase object

        Raises:
            ValueError: If the component is not in the graph.
        """
        try:
            return self.component_instances[component_name]
        except KeyError:
            raise ValueError(f"Component {component_name} is not in the graph")

    def get_last_component(self):
        """Retrieves the component that is computed last in the graph, usually the final estimator.

        Returns:
            ComponentBase object

        Raises:
            ValueError: If the component graph has no edges.
        """
        if len(self.compute_order) == 0:
            raise ValueError("Cannot get last component from edgeless graph")
        last_component_name = self.compute_order[-1]
        return self.get_component(last_component_name)

    def get_estimators(self):
        """Gets a list of all the estimator components within this graph.

        Returns:
            list: All estimator objects within the graph.

        Raises:
            ValueError: If the component graph is not yet instantiated.
        """
        if not isinstance(self.get_last_component(), ComponentBase):
            raise ValueError(
                "Cannot get estimators until the component graph is instantiated",
            )
        return [
            component_class
            for component_class in self.component_instances.values()
            if isinstance(component_class, Estimator)
        ]

    def get_inputs(self, component_name):
        """Retrieves all inputs for a given component.

        Args:
            component_name (str): Name of the component to look up.

        Returns:
            list[str]: List of inputs for the component to use.

        Raises:
            ValueError: If the component is not in the graph.
        """
        try:
            component_info = self.component_dict[component_name]
        except KeyError:
            raise ValueError(f"Component {component_name} not in the graph")
        if len(component_info) > 1:
            return component_info[1:]
        return []

    def describe(self, return_dict=False):
        """Outputs component graph details including component parameters.

        Args:
            return_dict (bool): If True, return dictionary of information about component graph. Defaults to False.

        Returns:
            dict: Dictionary of all component parameters if return_dict is True, else None

        Raises:
            ValueError: If the componentgraph is not instantiated
        """
        logger = get_logger(f"{__name__}.describe")
        components = {}
        for number, component in enumerate(self.component_instances.values(), 1):
            component_string = str(number) + ". " + component.name
            logger.info(component_string)
            try:
                components.update(
                    {
                        component.name: component.describe(
                            print_name=False,
                            return_dict=return_dict,
                        ),
                    },
                )
            except TypeError:
                raise ValueError(
                    "You need to instantiate ComponentGraph before calling describe()",
                )
        if return_dict:
            return components

    def graph(self, name=None, graph_format=None):
        """Generate an image representing the component graph.

        Args:
            name (str): Name of the graph. Defaults to None.
            graph_format (str): file format to save the graph in. Defaults to None.

        Returns:
            graphviz.Digraph: Graph object that can be directly displayed in Jupyter notebooks.

        Raises:
            RuntimeError: If graphviz is not installed.
        """
        graphviz = import_or_raise(
            "graphviz",
            error_msg="Please install graphviz to visualize pipelines.",
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
                + "  Windows: conda install python-graphviz\n",
            )

        graph = graphviz.Digraph(
            name=name,
            format=graph_format,
            graph_attr={"splines": "true", "overlap": "scale", "rankdir": "LR"},
        )
        for component_name, component_class in self.component_instances.items():
            label = "%s\l" % (component_name)  # noqa: W605
            if isinstance(component_class, ComponentBase):
                parameters = "\l".join(
                    [
                        key + " : " + "{:0.2f}".format(val)
                        if (isinstance(val, float))
                        else key + " : " + str(val)
                        for key, val in component_class.parameters.items()
                    ],
                )  # noqa: W605
                label = "%s |%s\l" % (component_name, parameters)  # noqa: W605
            graph.node(component_name, shape="record", label=label, nodesep="0.03")

        graph.node("X", shape="circle", label="X")
        graph.node("y", shape="circle", label="y")

        x_edges = self._get_edges(self.component_dict, "features")
        y_edges = self._get_edges(self.component_dict, "target")
        for component_name, component_info in self.component_dict.items():
            for parent in component_info[1:]:
                if parent == "X":
                    x_edges.append(("X", component_name))
                elif parent == "y":
                    y_edges.append(("y", component_name))

        for edge in x_edges:
            graph.edge(edge[0], edge[1], color="black")
        for edge in y_edges:
            graph.edge(edge[0], edge[1], style="dotted")

        return graph

    @staticmethod
    def _get_edges(component_dict, edges_to_return="all"):
        """Gets the edges for a component graph.

        Args:
            component_dict (dict): Component dictionary to get edges from.
            edges_to_return (str): The types of edges to return. Defaults to "all".
                - if "all", returns all types of edges.
                - if "features", returns only feature edges
                - if "target", returns only target edges
        """
        edges = []
        for component_name, component_info in component_dict.items():
            for parent in component_info[1:]:
                feature_edge = parent[-2:] == ".x"
                target_edge = parent[-2:] == ".y"
                return_edge = (
                    (edges_to_return == "features" and feature_edge)
                    or (edges_to_return == "target" and target_edge)
                    or (edges_to_return == "all" and (feature_edge or target_edge))
                )
                if parent == "X" or parent == "y":
                    continue
                elif return_edge:
                    parent = parent[:-2]
                    edges.append((parent, component_name))
        return edges

    @classmethod
    def generate_order(cls, component_dict):
        """Regenerated the topologically sorted order of the graph."""
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
                "The given graph has more than one final (childless) component",
            )
        return compute_order

    def __getitem__(self, index):
        """Get an element in the component graph."""
        if isinstance(index, int):
            return self.get_component(self.compute_order[index])
        else:
            return self.get_component(index)

    def __iter__(self):
        """Iterator for the component graph."""
        self._i = 0
        return self

    def __next__(self):
        """Iterator for graphs, retrieves the components in the graph in order.

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
        """Test for equality."""
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
        """String representation of a component graph."""
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
                    [f"'{info}'" for info in component_info[1:]],
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

        Components that implement inverse_transform are PolynomialDecomposer, LogTransformer, LabelEncoder (tbd).

        Args:
            y: (pd.Series): Final component features.

        Returns:
            pd.Series: The target with inverse transformation applied.
        """
        data_to_transform = infer_feature_types(y)
        current_component = self.compute_order[-1]
        has_incoming_y_from_parent = True
        while has_incoming_y_from_parent:
            parent_y = self._get_parent_y(current_component)
            if parent_y:
                component = self.get_component(parent_y)
                if hasattr(component, "inverse_transform"):
                    data_to_transform = component.inverse_transform(data_to_transform)
                current_component = parent_y
            else:
                has_incoming_y_from_parent = False

        return data_to_transform
