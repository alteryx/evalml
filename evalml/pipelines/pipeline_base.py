"""Base machine learning pipeline class."""
import copy
import inspect
import logging
import os
import sys
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict

import cloudpickle
import numpy as np
import pandas as pd

from evalml.exceptions import ObjectiveCreationError, PipelineScoreError
from evalml.objectives import get_objective
from evalml.pipelines import ComponentGraph
from evalml.pipelines.components import (
    PCA,
    ComponentBase,
    DFSTransformer,
    Estimator,
    LinearDiscriminantAnalysis,
)
from evalml.pipelines.components.utils import all_components, handle_component_class
from evalml.pipelines.pipeline_meta import PipelineBaseMeta
from evalml.problem_types import is_binary
from evalml.utils import (
    import_or_raise,
    infer_feature_types,
    jupyter_check,
    log_subtitle,
    log_title,
    safe_repr,
)
from evalml.utils.logger import get_logger

logger = logging.getLogger(__name__)


class PipelineBase(ABC, metaclass=PipelineBaseMeta):
    """Machine learning pipeline.

    Args:
        component_graph (ComponentGraph, list, dict): ComponentGraph instance, list of components in order, or dictionary of components.
            Accepts strings or ComponentBase subclasses in the list.
            Note that when duplicate components are specified in a list, the duplicate component names will be modified with the
            component's index in the list. For example, the component graph
            [Imputer, One Hot Encoder, Imputer, Logistic Regression Classifier] will have names
            ["Imputer", "One Hot Encoder", "Imputer_2", "Logistic Regression Classifier"].
        parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters. Defaults to None.
        custom_name (str): Custom name for the pipeline. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    problem_type = None
    """None"""

    def __init__(
        self,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        self.random_seed = random_seed

        if isinstance(component_graph, list):  # Backwards compatibility
            for component in component_graph:
                component = handle_component_class(component)
                if not component._supported_by_list_API:
                    raise ValueError(
                        f"{component.name} cannot be defined in a list because edges may be ambiguous. Please use a dictionary to specify the appropriate component graph for this pipeline instead.",
                    )
            self.component_graph = ComponentGraph(
                component_dict=PipelineBase._make_component_dict_from_component_list(
                    component_graph,
                ),
                random_seed=self.random_seed,
            )
        elif isinstance(component_graph, dict):
            self.component_graph = ComponentGraph(
                component_dict=component_graph,
                random_seed=self.random_seed,
            )
        elif isinstance(component_graph, ComponentGraph):
            self.component_graph = ComponentGraph(
                component_dict=component_graph.component_dict,
                cached_data=component_graph.cached_data,
                random_seed=self.random_seed,
            )
        else:
            raise ValueError(
                "component_graph must be a list, dict, or ComponentGraph object",
            )
        self.component_graph.instantiate(parameters)

        self.input_feature_names = {}
        self.input_target_name = None

        self.estimator = None
        if len(self.component_graph.compute_order) > 0:
            final_component = self.component_graph.get_last_component()
            self.estimator = (
                final_component if isinstance(final_component, Estimator) else None
            )
        self._estimator_name = (
            self.component_graph.compute_order[-1]
            if self.estimator is not None
            else None
        )

        self._validate_estimator_problem_type()
        self._is_fitted = False

        self._pipeline_params = None
        if parameters is not None:
            self._pipeline_params = parameters.get("pipeline", {})

        self._custom_name = custom_name

    @property
    def custom_name(self):
        """Custom name of the pipeline."""
        return self._custom_name

    @property
    def name(self):
        """Name of the pipeline."""
        return self.custom_name or self.summary

    @property
    def summary(self):
        """A short summary of the pipeline structure, describing the list of components used.

        Example: Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder

        Returns:
            A string describing the pipeline structure.
        """
        component_graph = [
            type(self.component_graph.component_instances[component])
            for component in copy.copy(self.component_graph.component_instances)
        ]
        if len(component_graph) == 0:
            return "Empty Pipeline"
        summary = "Pipeline"
        component_graph[-1] = component_graph[-1]

        if inspect.isclass(component_graph[-1]) and issubclass(
            component_graph[-1],
            Estimator,
        ):
            estimator_class = component_graph.pop(-1)
            summary = estimator_class.name
        if len(component_graph) == 0:
            return summary
        component_names = [component_class.name for component_class in component_graph]
        return "{} w/ {}".format(summary, " + ".join(component_names))

    @staticmethod
    def _make_component_dict_from_component_list(component_list):
        """Generates a component dictionary from a list of components."""
        components_with_names = []
        seen = set()
        for idx, component in enumerate(component_list):
            component_class = handle_component_class(component)
            component_name = component_class.name
            if component_name in seen:
                component_name = f"{component_name}_{idx}"
            seen.add(component_name)
            components_with_names.append((component_name, component_class))

        component_dict = {}
        most_recent_target = "y"
        most_recent_features = "X"
        for component_name, component_class in components_with_names:
            component_dict[component_name] = [
                component_class,
                most_recent_features,
                most_recent_target,
            ]
            if component_class.modifies_target:
                most_recent_target = f"{component_name}.y"
            if component_class.modifies_features:
                most_recent_features = f"{component_name}.x"
        return component_dict

    def _validate_estimator_problem_type(self):
        """Validates this pipeline's problem_type against that of the estimator from `self.component_graph`."""
        if (
            self.estimator is None
        ):  # Allow for pipelines that do not end with an estimator
            return
        estimator_problem_types = self.estimator.supported_problem_types
        if self.problem_type not in estimator_problem_types:
            raise ValueError(
                "Problem type {} not valid for this component graph. Valid problem types include {}.".format(
                    self.problem_type,
                    estimator_problem_types,
                ),
            )

    def __getitem__(self, index):
        """Get an element in the component graph."""
        if isinstance(index, slice):
            raise NotImplementedError("Slicing pipelines is currently not supported.")
        return self.component_graph[index]

    def __setitem__(self, index, value):
        """Set an element in the component graph."""
        raise NotImplementedError("Setting pipeline components is not supported.")

    def get_component(self, name):
        """Returns component by name.

        Args:
            name (str): Name of component.

        Returns:
            Component: Component to return
        """
        return self.component_graph.get_component(name)

    def describe(self, return_dict=False):
        """Outputs pipeline details including component parameters.

        Args:
            return_dict (bool): If True, return dictionary of information about pipeline. Defaults to False.

        Returns:
            dict: Dictionary of all component parameters if return_dict is True, else None.
        """
        logger = get_logger(f"{__name__}.describe")
        log_title(logger, self.name)
        logger.info("Problem Type: {}".format(self.problem_type))
        logger.info("Model Family: {}".format(str(self.model_family)))

        if self._estimator_name in self.input_feature_names:
            logger.info(
                "Number of features: {}".format(
                    len(self.input_feature_names[self._estimator_name]),
                ),
            )

        # Summary of steps
        log_subtitle(logger, "Pipeline Steps")

        pipeline_dict = {
            "name": self.name,
            "problem_type": self.problem_type,
            "model_family": self.model_family,
            "components": dict(),
        }
        component_dict = self.component_graph.describe(return_dict=return_dict)
        if return_dict:
            pipeline_dict.update({"components": component_dict})
            return pipeline_dict

    def transform_all_but_final(self, X, y=None, X_train=None, y_train=None):
        """Transforms the data by applying all pre-processing components.

        Args:
            X (pd.DataFrame): Input data to the pipeline to transform.
            y (pd.Series or None): Targets corresponding to X. Optional.
            X_train (pd.DataFrame or np.ndarray or None): Training data. Only used for time series.
            y_train (pd.Series or None): Training labels.  Only used for time series.

        Returns:
            pd.DataFrame: New transformed features.
        """
        return self.component_graph.transform_all_but_final(X, y=y)

    def _fit(self, X, y):
        self.input_target_name = y.name
        self.component_graph.fit(X, y)
        self.input_feature_names = self.component_graph.input_feature_names

    @abstractmethod
    def fit(self, X, y):
        """Build a model.

        Args:
            X (pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features].
            y (pd.Series, np.ndarray): The target training data of length [n_samples].

        Returns:
            self
        """

    def transform(self, X, y=None):
        """Transform the input.

        Args:
            X (pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series): The target data of length [n_samples]. Defaults to None.

        Returns:
            pd.DataFrame: Transformed output.
        """
        return self.component_graph.transform(X, y)

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
        return self.component_graph.fit_transform(X, y)

    def predict(self, X, objective=None, X_train=None, y_train=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features].
            objective (Object or string): The objective to use to make predictions.
            X_train (pd.DataFrame or np.ndarray or None): Training data. Ignored. Only used for time series.
            y_train (pd.Series or None): Training labels. Ignored. Only used for time series.

        Returns:
            pd.Series: Predicted values.
        """
        X = infer_feature_types(X)
        predictions = self.component_graph.predict(X)
        predictions.name = self.input_target_name
        return infer_feature_types(predictions)

    @abstractmethod
    def score(self, X, y, objectives, X_train=None, y_train=None):
        """Evaluate model performance on current and additional objectives.

        Args:
            X (pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features].
            y (pd.Series, np.ndarray): True labels of length [n_samples].
            objectives (list): Non-empty list of objectives to score on.
            X_train (pd.DataFrame or np.ndarray or None): Training data. Ignored. Only used for time series.
            y_train (pd.Series or None): Training labels. Ignored. Only used for time series.

        Returns:
            dict: Ordered dictionary of objective scores.
        """

    @staticmethod
    def _score(X, y, predictions, objective):
        return objective.score(y, predictions, X)

    def _score_all_objectives(self, X, y, y_pred, y_pred_proba, objectives):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score.

        Will raise a PipelineScoreError if any objectives fail.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target data.
            y_pred (pd.Series): The pipeline predictions.
            y_pred_proba (pd.Dataframe, pd.Series, None): The predicted probabilities for classification problems.
                Will be a DataFrame for multiclass problems and Series otherwise. Will be None for regression problems.
            objectives (list): List of objectives to score.

        Returns:
            dict: Ordered dictionary with objectives and their scores.
        """
        scored_successfully = OrderedDict()
        exceptions = OrderedDict()
        for objective in objectives:
            try:
                if not objective.is_defined_for_problem_type(self.problem_type):
                    raise ValueError(
                        f"Invalid objective {objective.name} specified for problem type {self.problem_type}",
                    )
                y_pred = self._select_y_pred_for_score(
                    X,
                    y,
                    y_pred,
                    y_pred_proba,
                    objective,
                )
                score = self._score(
                    X,
                    y,
                    y_pred_proba if objective.score_needs_proba else y_pred,
                    objective,
                )
                scored_successfully.update({objective.name: score})
            except Exception as e:
                tb = traceback.format_tb(sys.exc_info()[2])
                exceptions[objective.name] = (e, tb)
        if exceptions:
            # If any objective failed, throw an PipelineScoreError
            raise PipelineScoreError(exceptions, scored_successfully)
        # No objectives failed, return the scores
        return scored_successfully

    def _select_y_pred_for_score(self, X, y, y_pred, y_pred_proba, objective):
        return y_pred

    @property
    def model_family(self):
        """Returns model family of this pipeline."""
        component_graph = copy.copy(self.component_graph)
        if isinstance(component_graph, list):
            return handle_component_class(component_graph[-1]).model_family
        else:
            order = ComponentGraph.generate_order(component_graph.component_dict)
            final_component = order[-1]
            return handle_component_class(
                component_graph[final_component].__class__,
            ).model_family

    @property
    def parameters(self):
        """Parameter dictionary for this pipeline.

        Returns:
            dict: Dictionary of all component parameters.
        """
        components = [
            (component_name, component_class)
            for component_name, component_class in self.component_graph.component_instances.items()
        ]
        component_parameters = {
            c_name: copy.copy(c.parameters) for c_name, c in components if c.parameters
        }
        if self._pipeline_params:
            component_parameters["pipeline"] = self._pipeline_params
        return component_parameters

    @property
    def feature_importance(self):
        """Importance associated with each feature. Features dropped by the feature selection are excluded.

        Returns:
            pd.DataFrame: Feature names and their corresponding importance
        """
        feature_names = self.input_feature_names[self._estimator_name]
        importance = list(
            zip(feature_names, self.estimator.feature_importance),
        )  # note: this only works for binary
        importance.sort(key=lambda x: -abs(x[1]))
        df = pd.DataFrame(importance, columns=["feature", "importance"])
        return df

    def graph_dict(self):
        """Generates a dictionary with nodes consisting of the component names and parameters, and edges detailing component relationships. This dictionary is JSON serializable in most cases.

        x_edges specifies from which component feature data is being passed.
        y_edges specifies from which component target data is being passed.
        This can be used to build graphs across a variety of visualization tools.
        Template:
        {"Nodes": {"component_name": {"Name": class_name, "Parameters": parameters_attributes}, ...}},
        "x_edges": [[from_component_name, to_component_name], [from_component_name, to_component_name], ...],
        "y_edges": [[from_component_name, to_component_name], [from_component_name, to_component_name], ...]}

        Returns:
            dag_dict (dict): A dictionary representing the DAG structure.
        """
        nodes = {}
        for comp_, att_ in self.component_graph.component_instances.items():
            param_dict = {}
            for param, val in att_.parameters.items():
                # Can't JSON serialize components directly, have to split them into name and parameters
                if isinstance(val, ComponentBase):
                    param_dict[f"{param}_name"] = val.name
                    param_dict[f"{param}_parameters"] = val.parameters
                else:
                    if isinstance(val, np.int64):
                        val = int(val)
                    param_dict[param] = val
            nodes[comp_] = {"Parameters": param_dict, "Name": att_.name}

        x_edges_list = self.component_graph._get_edges(
            self.component_graph.component_dict,
            "features",
        )
        y_edges_list = self.component_graph._get_edges(
            self.component_graph.component_dict,
            "target",
        )
        x_edges = [{"from": edge[0], "to": edge[1]} for edge in x_edges_list]
        y_edges = [{"from": edge[0], "to": edge[1]} for edge in y_edges_list]

        for (
            component_name,
            component_info,
        ) in self.component_graph.component_dict.items():
            for parent in component_info[1:]:
                if parent == "X":
                    x_edges.append({"from": "X", "to": component_name})
                elif parent == "y":
                    y_edges.append({"from": "y", "to": component_name})
        nodes["X"] = {"Parameters": {}, "Name": "X"}
        nodes["y"] = {"Parameters": {}, "Name": "y"}
        graph_as_dict = {"Nodes": nodes, "x_edges": x_edges, "y_edges": y_edges}

        for x_edge in graph_as_dict["x_edges"]:
            if x_edge["from"] == "X":
                graph_as_dict["x_edges"].remove(x_edge)
                graph_as_dict["x_edges"].insert(0, x_edge)

        return graph_as_dict

    def graph(self, filepath=None):
        """Generate an image representing the pipeline graph.

        Args:
            filepath (str, optional): Path to where the graph should be saved. If set to None (as by default), the graph will not be saved.

        Returns:
            graphviz.Digraph: Graph object that can be directly displayed in Jupyter notebooks.

        Raises:
            RuntimeError: If graphviz is not installed.
            ValueError: If path is not writeable.
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
                "To graph pipelines, a graphviz backend is required.\n"
                + "Install the backend using one of the following commands:\n"
                + "  Mac OS: brew install graphviz\n"
                + "  Linux (Ubuntu): sudo apt-get install graphviz\n"
                + "  Windows: conda install python-graphviz\n",
            )

        graph_format = None
        path_and_name = None
        if filepath:
            # Explicitly cast to str in case a Path object was passed in
            filepath = str(filepath)
            try:
                f = open(filepath, "w")
                f.close()
            except (IOError, FileNotFoundError):
                raise ValueError(
                    ("Specified filepath is not writeable: {}".format(filepath)),
                )
            path_and_name, graph_format = os.path.splitext(filepath)
            graph_format = graph_format[1:].lower()  # ignore the dot
            supported_filetypes = graphviz.FORMATS
            if graph_format not in supported_filetypes:
                raise ValueError(
                    (
                        "Unknown format '{}'. Make sure your format is one of the "
                        + "following: {}"
                    ).format(graph_format, supported_filetypes),
                )

        graph = self.component_graph.graph(path_and_name, graph_format)

        if filepath:
            graph.render(path_and_name, cleanup=True)

        return graph

    def graph_feature_importance(self, importance_threshold=0):
        """Generate a bar graph of the pipeline's feature importance.

        Args:
            importance_threshold (float, optional): If provided, graph features with a permutation importance whose absolute value is larger than importance_threshold. Defaults to zero.

        Returns:
            plotly.Figure: A bar graph showing features and their corresponding importance.

        Raises:
            ValueError: If importance threshold is not valid.
        """
        go = import_or_raise(
            "plotly.graph_objects",
            error_msg="Cannot find dependency plotly.graph_objects",
        )
        if jupyter_check():
            import_or_raise("ipywidgets", warning=True)

        feat_imp = self.feature_importance
        feat_imp["importance"] = abs(feat_imp["importance"])

        if importance_threshold < 0:
            raise ValueError(
                f"Provided importance threshold of {importance_threshold} must be greater than or equal to 0",
            )

        # Remove features with importance whose absolute value is less than importance threshold
        feat_imp = feat_imp[feat_imp["importance"] >= importance_threshold]

        # List is reversed to go from ascending order to descending order
        feat_imp = feat_imp.iloc[::-1]

        title = "Feature Importance"
        subtitle = "May display fewer features due to feature selection"
        data = [
            go.Bar(x=feat_imp["importance"], y=feat_imp["feature"], orientation="h"),
        ]

        layout = {
            "title": "{0}<br><sub>{1}</sub>".format(title, subtitle),
            "height": 800,
            "xaxis_title": "Feature Importance",
            "yaxis_title": "Feature",
            "yaxis": {"type": "category"},
        }

        fig = go.Figure(data=data, layout=layout)
        return fig

    def save(self, file_path, pickle_protocol=cloudpickle.DEFAULT_PROTOCOL):
        """Saves pipeline at file path.

        Args:
            file_path (str): Location to save file.
            pickle_protocol (int): The pickle data stream format.
        """
        with open(file_path, "wb") as f:
            cloudpickle.dump(self, f, protocol=pickle_protocol)

    @staticmethod
    def load(file_path):
        """Loads pipeline at file path.

        Args:
            file_path (str): Location to load file.

        Returns:
            PipelineBase object
        """
        with open(file_path, "rb") as f:
            return cloudpickle.load(f)

    def clone(self):
        """Constructs a new pipeline with the same components, parameters, and random seed.

        Returns:
            A new instance of this pipeline with identical components, parameters, and random seed.
        """
        clone = self.__class__(
            component_graph=self.component_graph,
            parameters=self.parameters,
            custom_name=self.custom_name,
            random_seed=self.random_seed,
        )
        if is_binary(self.problem_type):
            clone.threshold = self.threshold
        return clone

    def new(self, parameters, random_seed=0):
        """Constructs a new instance of the pipeline with the same component graph but with a different set of parameters. Not to be confused with python's __new__ method.

        Args:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary or None implies using all default values for component parameters. Defaults to None.
            random_seed (int): Seed for the random number generator. Defaults to 0.

        Returns:
            A new instance of this pipeline with identical components.
        """
        return self.__class__(
            self.component_graph,
            parameters=parameters,
            custom_name=self.custom_name,
            random_seed=random_seed,
        )

    def __eq__(self, other):
        """Check for equality."""
        if not isinstance(other, self.__class__):
            return False
        random_seed_eq = self.random_seed == other.random_seed
        if not random_seed_eq:
            return False
        attributes_to_check = [
            "parameters",
            "_is_fitted",
            "component_graph",
            "input_feature_names",
            "input_target_name",
        ]
        for attribute in attributes_to_check:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def __str__(self):
        """String representation of the component graph."""
        return self.name

    def __repr__(self):
        """String representation of the component graph."""

        def repr_component(parameters):
            return ", ".join(
                [f"'{key}': {safe_repr(value)}" for key, value in parameters.items()],
            )

        component_dict_str = repr(self.component_graph)
        parameters_repr = ", ".join(
            [
                f"'{component}':{{{repr_component(parameters)}}}"
                for component, parameters in self.parameters.items()
            ],
        )
        parameters_str = f"parameters={{{parameters_repr}}}"

        custom_name_repr = (
            f"custom_name='{self.custom_name}'" if self.custom_name else None
        )
        random_seed_str = f"random_seed={self.random_seed}"
        additional_args_str = ", ".join(
            [
                arg
                for arg in [
                    parameters_str,
                    custom_name_repr,
                    random_seed_str,
                ]
                if arg is not None
            ],
        )

        return f"pipeline = {(type(self).__name__)}(component_graph={component_dict_str}, {additional_args_str})"

    def __iter__(self):
        """Iterator for the component graph."""
        return self

    def __next__(self):
        """Get the next element in the component graph."""
        return next(self.component_graph)

    def _get_feature_provenance(self):
        return self.component_graph._feature_provenance

    @property
    def _supports_fast_permutation_importance(self):
        has_more_than_one_estimator = (
            sum(isinstance(c, Estimator) for c in self.component_graph) > 1
        )
        _all_components = set(all_components())
        has_custom_components = any(
            c.__class__ not in _all_components for c in self.component_graph
        )
        has_dim_reduction = any(
            isinstance(c, (PCA, LinearDiscriminantAnalysis))
            for c in self.component_graph
        )
        has_dfs = any(isinstance(c, DFSTransformer) for c in self.component_graph)
        return not any(
            [
                has_more_than_one_estimator,
                has_custom_components,
                has_dim_reduction,
                has_dfs,
            ],
        )

    @staticmethod
    def create_objectives(objectives):
        """Create objective instances from a list of strings or objective classes."""
        objective_instances = []
        for objective in objectives:
            try:
                objective_instances.append(
                    get_objective(objective, return_instance=True),
                )
            except ObjectiveCreationError as e:
                msg = f"Cannot pass {objective} as a string in pipeline.score. Instantiate first and then add it to the list of objectives."
                raise ObjectiveCreationError(msg) from e
        return objective_instances

    def can_tune_threshold_with_objective(self, objective):
        """Determine whether the threshold of a binary classification pipeline can be tuned.

        Args:
             objective (ObjectiveBase): Primary AutoMLSearch objective.

        Returns:
             bool: True if the pipeline threshold can be tuned.
        """
        return (
            is_binary(self.problem_type)
            and objective.is_defined_for_problem_type(self.problem_type)
            and objective.can_optimize_threshold
        )

    def inverse_transform(self, y):
        """Apply component inverse_transform methods to estimator predictions in reverse order.

        Components that implement inverse_transform are PolynomialDecomposer, LogTransformer, LabelEncoder (tbd).

        Args:
            y (pd.Series): Final component features.

        Returns:
            pd.Series: The inverse transform of the target.
        """
        return self.component_graph.inverse_transform(y)

    def get_hyperparameter_ranges(self, custom_hyperparameters):
        """Returns hyperparameter ranges from all components as a dictionary.

        Args:
            custom_hyperparameters (dict): Custom hyperparameters for the pipeline.

        Returns:
            dict: Dictionary of hyperparameter ranges for each component in the pipeline.
        """
        hyperparameter_ranges = dict()
        for (
            component_name,
            component_class,
        ) in self.component_graph.component_instances.items():
            component_hyperparameters = copy.copy(component_class.hyperparameter_ranges)
            if custom_hyperparameters and component_name in custom_hyperparameters:
                component_hyperparameters.update(custom_hyperparameters[component_name])
            hyperparameter_ranges[component_name] = component_hyperparameters
        return hyperparameter_ranges
