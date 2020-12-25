import copy
import inspect
import os
import re
import sys
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict

import cloudpickle
import pandas as pd

from .components import Estimator
from .components.utils import handle_component_class

from evalml.exceptions import IllFormattedClassNameError, PipelineScoreError
from evalml.pipelines import ComponentGraph
from evalml.pipelines.pipeline_base_meta import PipelineBaseMeta
from evalml.utils import (
    _convert_to_woodwork_structure,
    check_random_state_equality,
    classproperty,
    get_logger,
    get_random_state,
    import_or_raise,
    jupyter_check,
    log_subtitle,
    log_title,
    safe_repr
)

logger = get_logger(__file__)


class PipelineBase(ABC, metaclass=PipelineBaseMeta):
    """Base class for all pipelines."""

    @property
    @classmethod
    @abstractmethod
    def component_graph(cls):
        """Returns list or dictionary of components representing pipeline graph structure

        Returns:
            list(str / ComponentBase subclass): List of ComponentBase subclasses or strings denotes graph structure of this pipeline
        """

    custom_hyperparameters = None
    custom_name = None
    problem_type = None

    def __init__(self, parameters, random_state=0):
        """Machine learning pipeline made out of transformers and a estimator.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase subclasses in the list

        Arguments:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self.random_state = get_random_state(random_state)
        if isinstance(self.component_graph, list):  # Backwards compatibility
            self._component_graph = ComponentGraph().from_list(self.component_graph, random_state=self.random_state)
        else:
            self._component_graph = ComponentGraph(component_dict=self.component_graph, random_state=self.random_state)
        self._component_graph.instantiate(parameters)

        self.input_feature_names = {}
        final_component = self._component_graph.get_last_component()
        self.estimator = final_component if isinstance(final_component, Estimator) else None
        self._estimator_name = self._component_graph.compute_order[-1] if self.estimator is not None else None

        self._validate_estimator_problem_type()
        self._is_fitted = False
        self._pipeline_params = parameters.get("pipeline", {})

    @classproperty
    def name(cls):
        """Returns a name describing the pipeline.
        By default, this will take the class name and add a space between each capitalized word (class name should be in Pascal Case). If the pipeline has a custom_name attribute, this will be returned instead.
        """
        if cls.custom_name:
            name = cls.custom_name
        else:
            rex = re.compile(r'(?<=[a-z])(?=[A-Z])')
            name = rex.sub(' ', cls.__name__)
            if name == cls.__name__:
                raise IllFormattedClassNameError("Pipeline Class {} needs to follow Pascal Case standards or `custom_name` must be defined.".format(cls.__name__))
        return name

    @classproperty
    def summary(cls):
        """Returns a short summary of the pipeline structure, describing the list of components used.
        Example: Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder
        """
        component_graph = [handle_component_class(component_class) for component_class in copy.copy(cls.linearized_component_graph)]
        if len(component_graph) == 0:
            return "Empty Pipeline"
        summary = "Pipeline"
        component_graph[-1] = component_graph[-1]

        if inspect.isclass(component_graph[-1]) and issubclass(component_graph[-1], Estimator):
            estimator_class = component_graph.pop(-1)
            summary = estimator_class.name
        if len(component_graph) == 0:
            return summary
        component_names = [component_class.name for component_class in component_graph]
        return '{} w/ {}'.format(summary, ' + '.join(component_names))

    @classproperty
    def linearized_component_graph(cls):
        """Returns a component graph in list form. Note: this is not guaranteed to be in proper component computation order
        """
        if isinstance(cls.component_graph, list):
            return cls.component_graph
        else:
            return [component_info[0] for component_info in cls.component_graph.values()]

    def _validate_estimator_problem_type(self):
        """Validates this pipeline's problem_type against that of the estimator from `self.component_graph`"""
        if self.estimator is None:  # Allow for pipelines that do not end with an estimator
            return
        estimator_problem_types = self.estimator.supported_problem_types
        if self.problem_type not in estimator_problem_types:
            raise ValueError("Problem type {} not valid for this component graph. Valid problem types include {}."
                             .format(self.problem_type, estimator_problem_types))

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise NotImplementedError('Slicing pipelines is currently not supported.')
        elif isinstance(index, int):
            component_name = self.component_graph[index]
            if not isinstance(component_name, str):
                component_name = component_name.name
            return self.get_component(component_name)
        else:
            return self.get_component(index)

    def __setitem__(self, index, value):
        raise NotImplementedError('Setting pipeline components is not supported.')

    def get_component(self, name):
        """Returns component by name

        Arguments:
            name (str): Name of component

        Returns:
            Component: Component to return

        """
        return self._component_graph.get_component(name)

    def describe(self):
        """Outputs pipeline details including component parameters

        Arguments:
            return_dict (bool): If True, return dictionary of information about pipeline. Defaults to false

        Returns:
            dict: Dictionary of all component parameters if return_dict is True, else None
        """
        log_title(logger, self.name)
        logger.info("Problem Type: {}".format(self.problem_type))
        logger.info("Model Family: {}".format(str(self.model_family)))

        if self._estimator_name in self.input_feature_names:
            logger.info("Number of features: {}".format(len(self.input_feature_names[self._estimator_name])))

        # Summary of steps
        log_subtitle(logger, "Pipeline Steps")
        for number, component in enumerate(self._component_graph, 1):
            component_string = str(number) + ". " + component.name
            logger.info(component_string)
            component.describe(print_name=False)

    def compute_estimator_features(self, X, y=None):
        """Transforms the data by applying all pre-processing components.

        Arguments:
            X (pd.DataFrame): Input data to the pipeline to transform.

        Returns:
            pd.DataFrame - New transformed features.
        """
        X_t = self._component_graph.compute_final_component_features(X, y=y)
        return X_t

    def _compute_features_during_fit(self, X, y):
        X_t = self._component_graph.fit_features(X, y)
        self.input_feature_names = self._component_graph.input_feature_names
        return X_t

    def _fit(self, X, y):
        self._component_graph.fit(X, y)
        self.input_feature_names = self._component_graph.input_feature_names

    @abstractmethod
    def fit(self, X, y):
        """Build a model

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (ww.DataColumn, pd.Series, np.ndarray): The target training data of length [n_samples]

        Returns:
            self

        """

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]
            objective (Object or string): The objective to use to make predictions

        Returns:
            pd.Series: Predicted values.
        """
        X = _convert_to_woodwork_structure(X)
        return self._component_graph.predict(X)

    @abstractmethod
    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Arguments:
            X (ww.DataTable, pd.DataFrame or np.ndarray): Data of shape [n_samples, n_features]
            y (pd.Series, ww.DataColumn, or np.ndarray): True labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: Ordered dictionary of objective scores
        """

    @staticmethod
    def _score(X, y, predictions, objective):
        return objective.score(y, predictions, X)

    def _score_all_objectives(self, X, y, y_pred, y_pred_proba, objectives):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score.

        Will raise a PipelineScoreError if any objectives fail.
        Arguments:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target data.
            y_pred (pd.Series): The pipeline predictions.
            y_pred_proba (pd.Dataframe, pd.Series, None): The predicted probabilities for classification problems.
                Will be a DataFrame for multiclass problems and Series otherwise. Will be None for regression problems.
            objectives (list): List of objectives to score.
        """
        scored_successfully = OrderedDict()
        exceptions = OrderedDict()
        for objective in objectives:
            try:
                if not objective.is_defined_for_problem_type(self.problem_type):
                    raise ValueError(f'Invalid objective {objective.name} specified for problem type {self.problem_type}')
                score = self._score(X, y, y_pred_proba if objective.score_needs_proba else y_pred, objective)
                scored_successfully.update({objective.name: score})
            except Exception as e:
                tb = traceback.format_tb(sys.exc_info()[2])
                exceptions[objective.name] = (e, tb)
        if exceptions:
            # If any objective failed, throw an PipelineScoreError
            raise PipelineScoreError(exceptions, scored_successfully)
        # No objectives failed, return the scores
        return scored_successfully

    @classproperty
    def model_family(cls):
        "Returns model family of this pipeline template"""
        component_graph = copy.copy(cls.component_graph)
        if isinstance(component_graph, list):
            return handle_component_class(component_graph[-1]).model_family
        else:
            order = ComponentGraph.generate_order(component_graph)
            final_component = order[-1]
            return handle_component_class(component_graph[final_component][0]).model_family

    @classproperty
    def hyperparameters(cls):
        "Returns hyperparameter ranges from all components as a dictionary"
        hyperparameter_ranges = dict()
        component_graph = copy.copy(cls.component_graph)
        if isinstance(component_graph, list):
            for component_class in component_graph:
                component_class = handle_component_class(component_class)
                component_hyperparameters = copy.copy(component_class.hyperparameter_ranges)
                if cls.custom_hyperparameters and component_class.name in cls.custom_hyperparameters:
                    component_hyperparameters.update(cls.custom_hyperparameters.get(component_class.name, {}))
                hyperparameter_ranges[component_class.name] = component_hyperparameters
        else:
            for component_name, component_info in component_graph.items():
                component_class = handle_component_class(component_info[0])
                component_hyperparameters = copy.copy(component_class.hyperparameter_ranges)
                if cls.custom_hyperparameters and component_name in cls.custom_hyperparameters:
                    component_hyperparameters.update(cls.custom_hyperparameters.get(component_name, {}))
                hyperparameter_ranges[component_name] = component_hyperparameters
        return hyperparameter_ranges

    @property
    def parameters(self):
        """Returns parameter dictionary for this pipeline

        Returns:
            dict: Dictionary of all component parameters
        """
        components = [(component_name, component_class) for component_name, component_class in self._component_graph.component_instances.items()]
        component_parameters = {c_name: copy.copy(c.parameters) for c_name, c in components if c.parameters}
        if self._pipeline_params:
            component_parameters['pipeline'] = self._pipeline_params
        return component_parameters

    @classproperty
    def default_parameters(cls):
        """Returns the default parameter dictionary for this pipeline.

        Returns:
            dict: Dictionary of all component default parameters.
        """
        defaults = {}
        for c in cls.component_graph:
            component = handle_component_class(c)
            if component.default_parameters:
                defaults[component.name] = component.default_parameters
        return defaults

    @property
    def feature_importance(self):
        """Return importance associated with each feature. Features dropped by the feature selection are excluded.

        Returns:
            pd.DataFrame including feature names and their corresponding importance
        """
        feature_names = self.input_feature_names[self._estimator_name]
        importance = list(zip(feature_names, self.estimator.feature_importance))  # note: this only works for binary
        importance.sort(key=lambda x: -abs(x[1]))
        df = pd.DataFrame(importance, columns=["feature", "importance"])
        return df

    def graph(self, filepath=None):
        """Generate an image representing the pipeline graph

        Arguments:
            filepath (str, optional): Path to where the graph should be saved. If set to None (as by default), the graph will not be saved.

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

        graph_format = None
        path_and_name = None
        if filepath:
            # Explicitly cast to str in case a Path object was passed in
            filepath = str(filepath)
            try:
                f = open(filepath, 'w')
                f.close()
            except (IOError, FileNotFoundError):
                raise ValueError(('Specified filepath is not writeable: {}'.format(filepath)))
            path_and_name, graph_format = os.path.splitext(filepath)
            graph_format = graph_format[1:].lower()  # ignore the dot
            supported_filetypes = graphviz.backend.FORMATS
            if graph_format not in supported_filetypes:
                raise ValueError(("Unknown format '{}'. Make sure your format is one of the " +
                                  "following: {}").format(graph_format, supported_filetypes))

        graph = self._component_graph.graph(path_and_name, graph_format)

        if filepath:
            graph.render(path_and_name, cleanup=True)

        return graph

    def graph_feature_importance(self, importance_threshold=0):
        """Generate a bar graph of the pipeline's feature importance

        Arguments:
            importance_threshold (float, optional): If provided, graph features with a permutation importance whose absolute value is larger than importance_threshold. Defaults to zero.

        Returns:
            plotly.Figure, a bar graph showing features and their corresponding importance
        """
        go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")
        if jupyter_check():
            import_or_raise("ipywidgets", warning=True)

        feat_imp = self.feature_importance
        feat_imp['importance'] = abs(feat_imp['importance'])

        if importance_threshold < 0:
            raise ValueError(f'Provided importance threshold of {importance_threshold} must be greater than or equal to 0')

        # Remove features with importance whose absolute value is less than importance threshold
        feat_imp = feat_imp[feat_imp['importance'] >= importance_threshold]

        # List is reversed to go from ascending order to descending order
        feat_imp = feat_imp.iloc[::-1]

        title = 'Feature Importance'
        subtitle = 'May display fewer features due to feature selection'
        data = [go.Bar(
            x=feat_imp['importance'],
            y=feat_imp['feature'],
            orientation='h'
        )]

        layout = {
            'title': '{0}<br><sub>{1}</sub>'.format(title, subtitle),
            'height': 800,
            'xaxis_title': 'Feature Importance',
            'yaxis_title': 'Feature',
            'yaxis': {
                'type': 'category'
            }
        }

        fig = go.Figure(data=data, layout=layout)
        return fig

    def save(self, file_path, pickle_protocol=cloudpickle.DEFAULT_PROTOCOL):
        """Saves pipeline at file path

        Arguments:
            file_path (str): location to save file
            pickle_protocol (int): the pickle data stream format.

        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            cloudpickle.dump(self, f, protocol=pickle_protocol)

    @staticmethod
    def load(file_path):
        """Loads pipeline at file path

        Arguments:
            file_path (str): location to load file

        Returns:
            PipelineBase object
        """
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f)

    def clone(self, random_state=0):
        """Constructs a new pipeline with the same parameters and components.

        Arguments:
            random_state (int): the value to seed the random state with. Can also be a RandomState instance. Defaults to 0.

        Returns:
            A new instance of this pipeline with identical parameters and components
        """
        return self.__class__(self.parameters, random_state=random_state)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        random_state_eq = check_random_state_equality(self.random_state, other.random_state)
        if not random_state_eq:
            return False
        attributes_to_check = ['parameters', '_is_fitted', 'component_graph', 'input_feature_names']
        for attribute in attributes_to_check:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def __str__(self):
        return self.name

    def __repr__(self):

        def repr_component(parameters):
            return ', '.join([f"'{key}': {safe_repr(value)}" for key, value in parameters.items()])

        parameters_repr = ' '.join([f"'{component}':{{{repr_component(parameters)}}}," for component, parameters in self.parameters.items()])
        return f'{(type(self).__name__)}(parameters={{{parameters_repr}}})'
