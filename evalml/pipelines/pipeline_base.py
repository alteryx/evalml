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

from evalml.exceptions import (
    IllFormattedClassNameError,
    MissingComponentError,
    PipelineScoreError
)
from evalml.pipelines.pipeline_base_meta import PipelineBaseMeta
from evalml.utils import (
    check_random_state_equality,
    classproperty,
    get_logger,
    get_random_state,
    import_or_raise,
    jupyter_check,
    log_subtitle,
    log_title
)

logger = get_logger(__file__)


class PipelineBase(ABC, metaclass=PipelineBaseMeta):
    """Base class for all pipelines."""

    @property
    @classmethod
    @abstractmethod
    def component_graph(cls):
        """Returns list of components representing pipeline graph structure

        Returns:
            list(str / ComponentBase subclass): list of ComponentBase subclasses or strings denotes graph structure of this pipeline
        """

    custom_hyperparameters = None
    custom_name = None
    problem_type = None

    def __init__(self, parameters, random_state=0):
        """Machine learning pipeline made out of transformers and a estimator.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase subclasses in the list

        Arguments:
            parameters (dict): dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self.random_state = get_random_state(random_state)
        self.component_graph = [self._instantiate_component(component_class, parameters) for component_class in self.component_graph]
        self.input_feature_names = {}
        self.estimator = self.component_graph[-1] if isinstance(self.component_graph[-1], Estimator) else None
        if self.estimator is None:
            raise ValueError("A pipeline must have an Estimator as the last component in component_graph.")

        self._validate_estimator_problem_type()
        self._is_fitted = False

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
        component_graph = [handle_component_class(component_class) for component_class in copy.copy(cls.component_graph)]
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

    def _validate_estimator_problem_type(self):
        """Validates this pipeline's problem_type against that of the estimator from `self.component_graph`"""
        estimator_problem_types = self.estimator.supported_problem_types
        if self.problem_type not in estimator_problem_types:
            raise ValueError("Problem type {} not valid for this component graph. Valid problem types include {}."
                             .format(self.problem_type, estimator_problem_types))

    def _instantiate_component(self, component_class, parameters):
        """Instantiates components with parameters in `parameters`"""
        try:
            component_class = handle_component_class(component_class)
        except MissingComponentError as e:
            err = "Error recieved when retrieving class for component '{}'".format(component_class)
            raise MissingComponentError(err) from e

        component_name = component_class.name
        try:
            component_parameters = parameters.get(component_name, {})
            new_component = component_class(**component_parameters, random_state=self.random_state)
        except (ValueError, TypeError) as e:
            err = "Error received when instantiating component {} with the following arguments {}".format(component_name, component_parameters)
            raise ValueError(err) from e
        return new_component

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise NotImplementedError('Slicing pipelines is currently not supported.')
        elif isinstance(index, int):
            return self.component_graph[index]
        else:
            return self.get_component(index)

    def __setitem__(self, index, value):
        raise NotImplementedError('Setting pipeline components is not supported.')

    def get_component(self, name):
        """Returns component by name

        Arguments:
            name (str): name of component

        Returns:
            Component: component to return

        """
        return next((component for component in self.component_graph if component.name == name), None)

    def describe(self):
        """Outputs pipeline details including component parameters

        Arguments:
            return_dict (bool): If True, return dictionary of information about pipeline. Defaults to false

        Returns:
            dict: dictionary of all component parameters if return_dict is True, else None
        """
        log_title(logger, self.name)
        logger.info("Problem Type: {}".format(self.problem_type))
        logger.info("Model Family: {}".format(str(self.model_family)))

        if self.estimator.name in self.input_feature_names:
            logger.info("Number of features: {}".format(len(self.input_feature_names[self.estimator.name])))

        # Summary of steps
        log_subtitle(logger, "Pipeline Steps")
        for number, component in enumerate(self.component_graph, 1):
            component_string = str(number) + ". " + component.name
            logger.info(component_string)
            component.describe(print_name=False)

    def _transform(self, X):
        """Transforms the data by applying all pre-processing components.

        Arguments:
            X (pd.DataFrame): Input data to the pipeline to transform.

        Returns:
            pd.DataFrame - New dataframe.
        """
        X_t = X
        for component in self.component_graph[:-1]:
            X_t = component.transform(X_t)
        return X_t

    def _fit(self, X, y):
        X_t = X
        y_t = y
        for component in self.component_graph[:-1]:
            self.input_feature_names.update({component.name: list(pd.DataFrame(X_t))})
            X_t = component.fit_transform(X_t, y_t)

        self.input_feature_names.update({self.estimator.name: list(pd.DataFrame(X_t))})
        self.estimator.fit(X_t, y_t)

    @abstractmethod
    def fit(self, X, y):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self

        """

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Arguments:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]
            objective (Object or string): the objective to use to make predictions

        Returns:
            pd.Series: estimated labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_t = self._transform(X)
        return self.estimator.predict(X_t)

    @abstractmethod
    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Arguments:
            X (pd.DataFrame or np.array): data of shape [n_samples, n_features]
            y (pd.Series): true labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: ordered dictionary of objective scores
        """

    @staticmethod
    def _score(X, y, predictions, objective):
        return objective.score(y, predictions, X)

    def _score_all_objectives(self, X, y, y_pred, y_pred_proba, objectives):
        """Given data, model predictions or predicted probabilities computed on the data, and an objective, evaluate and return the objective score.

        Will raise a PipelineScoreError if any objectives fail.
        Arguments:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The labels.
            y_pred (pd.Series): The pipeline predictions.
            y_pred_proba (pd.Dataframe, pd.Series, None): The predicted probabilities for classification problems.
                Will be a DataFrame for multiclass problems and Series otherwise. Will be None for regression problems.
            objectives (list): List of objectives to score.
        """
        scored_successfully = OrderedDict()
        exceptions = OrderedDict()
        for objective in objectives:
            try:
                if self.problem_type != objective.problem_type:
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
        return handle_component_class(component_graph[-1]).model_family

    @classproperty
    def hyperparameters(cls):
        "Returns hyperparameter ranges from all components as a dictionary"
        hyperparameter_ranges = dict()
        component_graph = copy.copy(cls.component_graph)
        for component_class in component_graph:
            component_class = handle_component_class(component_class)
            component_hyperparameters = copy.copy(component_class.hyperparameter_ranges)
            if cls.custom_hyperparameters and component_class.name in cls.custom_hyperparameters:
                component_hyperparameters.update(cls.custom_hyperparameters.get(component_class.name, {}))
            hyperparameter_ranges[component_class.name] = component_hyperparameters
        return hyperparameter_ranges

    @property
    def parameters(self):
        """Returns parameter dictionary for this pipeline

        Returns:
            dict: dictionary of all component parameters
        """
        return {c.name: copy.copy(c.parameters) for c in self.component_graph if c.parameters}

    @classproperty
    def default_parameters(cls):
        """Returns the default parameter dictionary for this pipeline.

        Returns:
            dict: dictionary of all component default parameters.
        """
        defaults = {}
        for c in cls.component_graph:
            component = handle_component_class(c)
            if component.default_parameters:
                defaults[component.name] = component.default_parameters
        return defaults

    @property
    def feature_importance(self):
        """Return importance associated with each feature. Features dropped by feature selection are excluded"""
        feature_names = self.input_feature_names[self.estimator.name]
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

        # Initialize a new directed graph
        graph = graphviz.Digraph(name=self.name, format=graph_format,
                                 graph_attr={'splines': 'ortho'})
        graph.attr(rankdir='LR')

        # Draw components
        for component in self.component_graph:
            label = '%s\l' % (component.name)  # noqa: W605
            if len(component.parameters) > 0:
                parameters = '\l'.join([key + ' : ' + "{:0.2f}".format(val) if (isinstance(val, float))
                                        else key + ' : ' + str(val)
                                        for key, val in component.parameters.items()])  # noqa: W605
                label = '%s |%s\l' % (component.name, parameters)  # noqa: W605
            graph.node(component.name, shape='record', label=label)

        # Draw edges
        for i in range(len(self.component_graph[:-1])):
            graph.edge(self.component_graph[i].name, self.component_graph[i + 1].name)

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
