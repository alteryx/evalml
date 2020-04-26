import copy
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict

import cloudpickle
import pandas as pd

from .components import Estimator, handle_component

from evalml.exceptions import IllFormattedClassNameError
from evalml.objectives import get_objective
from evalml.utils import (
    Logger,
    classproperty,
    get_random_state,
    import_or_raise
)

logger = Logger()


class PipelineBase(ABC):
    """Base class for all pipelines."""

    @property
    @classmethod
    @abstractmethod
    def component_graph(cls):
        """Returns list of components representing pipeline graph structure

        Returns:
            list(str/ComponentBase): list of ComponentBase objects or strings denotes graph structure of this pipeline
        """
        return NotImplementedError("This pipeline must have `component_graph` as a class variable.")

    custom_hyperparameters = None
    custom_name = None
    problem_type = None

    def __init__(self, parameters, random_state=0):
        """Machine learning pipeline made out of transformers and a estimator.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase objects in the list

        Arguments:
            parameters (dict): dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self.random_state = get_random_state(random_state)
        self.component_graph = [self._instantiate_component(c, parameters) for c in self.component_graph]
        self.input_feature_names = {}
        self.results = {}
        self.estimator = self.component_graph[-1] if isinstance(self.component_graph[-1], Estimator) else None
        if self.estimator is None:
            raise ValueError("A pipeline must have an Estimator as the last component in component_graph.")

        self._validate_estimator_problem_type()

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
        component_graph = [handle_component(component) for component in copy.copy(cls.component_graph)]
        if len(component_graph) == 0:
            return "Empty Pipeline"
        summary = "Pipeline"
        component_graph[-1] = component_graph[-1]

        if isinstance(component_graph[-1], Estimator):
            estimator = component_graph.pop()
            summary = estimator.name
        if len(component_graph) == 0:
            return summary
        component_names = [component.name for component in component_graph]
        return '{} w/ {}'.format(summary, ' + '.join(component_names))

    def _validate_estimator_problem_type(self):
        """Validates this pipeline's problem_type against that of the estimator from `self.component_graph`"""
        estimator_problem_types = self.estimator.supported_problem_types
        if self.problem_type not in estimator_problem_types:
            raise ValueError("Problem type {} not valid for this component graph. Valid problem types include {}."
                             .format(self.problem_type, estimator_problem_types))

    def _instantiate_component(self, component, parameters):
        """Instantiates components with parameters in `parameters`"""
        component = handle_component(component)
        component_class = component.__class__
        component_name = component.name
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
        logger.log_title(self.name)
        logger.log("Problem Type: {}".format(self.problem_type))
        logger.log("Model Family: {}".format(str(self.model_family)))

        if self.estimator.name in self.input_feature_names:
            logger.log("Number of features: {}".format(len(self.input_feature_names[self.estimator.name])))

        # Summary of steps
        logger.log_subtitle("Pipeline Steps")
        for number, component in enumerate(self.component_graph, 1):
            component_string = str(number) + ". " + component.name
            logger.log(component_string)
            component.describe(print_name=False)

    def _transform(self, X):
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

    def fit(self, X, y):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self._fit(X, y)
        return self

    def predict(self, X, objective=None):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            objective (Object or string): the objective to use to make predictions

        Returns:
            pd.Series : estimated labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_t = self._transform(X)
        return self.estimator.predict(X_t)

    def score(self, X, y, objectives):
        """Evaluate model performance on current and additional objectives

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            y (pd.Series) : true labels of length [n_samples]
            objectives (list): Non-empty list of objectives to score on

        Returns:
            dict: ordered dictionary of objective scores
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        objectives = [get_objective(o) for o in objectives]
        y_predicted = None
        scores = OrderedDict()
        for objective in objectives:
            if objective.score_needs_proba:
                raise ValueError("Objective `{}` does not support score_needs_proba".format(objective.name))
            else:
                if y_predicted is None:
                    y_predicted = self.predict(X)
                scores.update({objective.name: objective.score(y, y_predicted, X)})
        return scores

    @classproperty
    def model_family(cls):
        "Returns model family of this pipeline template"""
        component_graph = copy.copy(cls.component_graph)
        return handle_component(component_graph[-1]).model_family

    @classproperty
    def hyperparameters(cls):
        "Returns hyperparameter ranges as a flat dictionary from all components "
        hyperparameter_ranges = dict()
        component_graph = copy.copy(cls.component_graph)
        for component in component_graph:
            component = handle_component(component)
            hyperparameter_ranges.update(component.hyperparameter_ranges)

        if cls.custom_hyperparameters:
            hyperparameter_ranges.update(cls.custom_hyperparameters)
        return hyperparameter_ranges

    @property
    def parameters(self):
        """Returns parameter dictionary for this pipeline

        Returns:
            dict: dictionary of all component parameters
        """
        return {c.name: copy.copy(c.parameters) for c in self.component_graph if c.parameters}

    @property
    def feature_importances(self):
        """Return feature importances. Features dropped by feature selection are excluded"""
        feature_names = self.input_feature_names[self.estimator.name]
        importances = list(zip(feature_names, self.estimator.feature_importances))  # note: this only works for binary
        importances.sort(key=lambda x: -abs(x[1]))
        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df

    def graph(self, filepath=None):
        """Generate an image representing the pipeline graph

        Arguments:
            filepath (str, optional) : Path to where the graph should be saved. If set to None (as by default), the graph will not be saved.

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

    def graph_feature_importance(self, show_all_features=False):
        """Generate a bar graph of the pipeline's feature importances

        Arguments:
            show_all_features (bool, optional) : If true, graph features with an importance value of zero. Defaults to false.

        Returns:
            plotly.Figure, a bar graph showing features and their importances
        """
        go = import_or_raise("plotly.graph_objects", error_msg="Cannot find dependency plotly.graph_objects")

        feat_imp = self.feature_importances
        feat_imp['importance'] = abs(feat_imp['importance'])

        if not show_all_features:
            # Remove features with zero importance
            feat_imp = feat_imp[feat_imp['importance'] != 0]

        # List is reversed to go from ascending order to descending order
        feat_imp = feat_imp.iloc[::-1]

        title = 'Feature Importances'
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

    def save(self, file_path):
        """Saves pipeline at file path

        Args:
            file_path (str) : location to save file

        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """Loads pipeline at file path

        Args:
            file_path (str) : location to load file

        Returns:
            PipelineBase obj
        """
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f)
