import copy
import re
from abc import ABC, abstractmethod
from collections import OrderedDict

import cloudpickle
import pandas as pd

from .components import Estimator, handle_component
from .graphs import make_feature_importance_graph, make_pipeline_graph

from evalml.exceptions import IllFormattedClassNameError
from evalml.objectives import get_objective
from evalml.problem_types import handle_problem_types
from evalml.utils import Logger, classproperty, get_random_state

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

    @property
    @classmethod
    @abstractmethod
    def supported_problem_types(cls):
        """Returns a list of ProblemTypes that this pipeline supports

        Returns:
            list(str/ProblemType): list of ProblemType objects or strings that this pipeline supports
        """
        return NotImplementedError("This pipeline must have `supported_problem_types` as a class variable.")

    custom_hyperparameters = None

    def __init__(self, parameters, random_state=0):
        """Machine learning pipeline made out of transformers and a estimator.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase objects in the list

            supported_problem_types (list): List of problem types for this pipeline. Accepts strings or ProbemType enum in the list.

        Arguments:
            parameters (dict): dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary {} implies using all default values for component parameters.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
        """
        self.random_state = get_random_state(random_state)
        self.component_graph = [self._instantiate_component(c, parameters) for c in self.component_graph]
        self.input_feature_names = {}
        self.results = {}
        self.supported_problem_types = [handle_problem_types(problem_type) for problem_type in self.supported_problem_types]
        self.estimator = self.component_graph[-1] if isinstance(self.component_graph[-1], Estimator) else None
        if self.estimator is None:
            raise ValueError("A pipeline must have an Estimator as the last component in component_graph.")

        self._validate_problem_types(self.supported_problem_types)

    @classproperty
    def name(cls):
        """Returns a name describing the pipeline.
        By default, this will take the class name and add a space between each capitalized word. If the pipeline has a _name attribute, this will be returned instead.
        """
        try:
            name = cls._name
        except AttributeError:
            rex = re.compile(r'(?<=[a-z])(?=[A-Z])')
            name = rex.sub(' ', cls.__name__)
            if name == cls.__name__:
                raise IllFormattedClassNameError("Pipeline Class {} needs to follow pascall case standards or `_name` must be defined.".format(cls.__name__))
        return name

    @classproperty
    def summary(cls):
        """Returns a short summary of the pipeline structure, describing the list of components used.
        Example: Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder
        """
        def _generate_summary(component_graph):
            component_graph[-1] = handle_component(component_graph[-1])
            estimator = component_graph[-1] if isinstance(component_graph[-1], Estimator) else None
            if estimator is not None:
                summary = "{}".format(estimator.name)
            else:
                summary = "Pipeline"
            for index, component in enumerate(component_graph[:-1]):
                component = handle_component(component)
                if index == 0:
                    summary += " w/ {}".format(component.name)
                else:
                    summary += " + {}".format(component.name)
            return summary

        return _generate_summary(cls.component_graph)

    def _validate_problem_types(self, problem_types):
        """Validates provided `problem_types` against the estimator in `self.component_graph`

        Arguments:
            problem_types (list): list of ProblemTypes
        """
        estimator_problem_types = self.estimator.supported_problem_types
        for problem_type in self.supported_problem_types:
            if problem_type not in estimator_problem_types:
                raise ValueError("Problem type {} not valid for this component graph. Valid problem types include {}.".format(problem_type, estimator_problem_types))

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
        logger.log("Supported Problem Types: {}".format(', '.join([str(problem_type) for problem_type in self.supported_problem_types])))
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
                y_predictions = y_predicted

            scores.update({objective.name: objective.score(y_predictions, y, X)})

        return scores

    def graph(self, filepath=None):
        """Generate an image representing the pipeline graph

        Arguments:
            filepath (str, optional) : Path to where the graph should be saved. If set to None (as by default), the graph will not be saved.

        Returns:
            graphviz.Digraph: Graph object that can be directly displayed in Jupyter notebooks.
        """
        return make_pipeline_graph(self.component_graph, self.name, filepath=filepath)

    @classproperty
    def model_family(cls):
        "Returns model family of this pipeline template"""
        return handle_component(cls.component_graph[-1]).model_family

    @classproperty
    def hyperparameters(cls):
        "Returns hyperparameter ranges as a flat dictionary from all components "
        hyperparameter_ranges = dict()
        for component in cls.component_graph:
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

    def feature_importance_graph(self, show_all_features=False):
        """Generate a bar graph of the pipeline's feature importances

        Arguments:
            show_all_features (bool, optional) : If true, graph features with an importance value of zero. Defaults to false.

        Returns:
            plotly.Figure, a bar graph showing features and their importances
        """
        return make_feature_importance_graph(self.feature_importances, show_all_features=show_all_features)

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
