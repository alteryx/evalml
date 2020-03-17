import re
from abc import ABC, abstractmethod
from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from .components import Estimator, handle_component
from .graphs import make_feature_importance_graph, make_pipeline_graph

from evalml.exceptions import IllFormattedClassNameError
from evalml.objectives import get_objective
from evalml.problem_types import handle_problem_types
from evalml.utils import Logger


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class PipelineBase(ABC):
    """Base class for all pipelines."""

    @property
    @classmethod
    @abstractmethod
    def component_graph(cls):
        return NotImplementedError("This pipeline must have `component_graph` as a class variable.")

    @property
    @classmethod
    @abstractmethod
    def problem_types(cls):
        return NotImplementedError("This pipeline must have `problem_types` as a class variable.")

    def __init__(self, parameters, objective):
        """Machine learning pipeline made out of transformers and a estimator.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase objects in the list
            problem_types (list): List of problem types for this pipeline. Accepts strings or ProbemType enum in the list.

        Arguments:
            objective (ObjectiveBase): the objective to optimize

            parameters (dict): dictionary with component names as keys and dictionary of that component's parameters as values.
                If `random_state`, `n_jobs`, or 'number_features' are provided as component parameters they will override the corresponding
                value provided as arguments to the pipeline. An empty dictionary {} implies using all default values for component parameters.
        """
        self.component_graph = [self._instantiate_component(c, parameters) for c in self.component_graph]
        self.problem_types = [handle_problem_types(problem_type) for problem_type in self.problem_types]
        self.logger = Logger()
        self.objective = get_objective(objective)
        self.input_feature_names = {}
        self.results = {}
        self.parameters = parameters

        self.estimator = self.component_graph[-1] if isinstance(self.component_graph[-1], Estimator) else None
        if self.estimator is None:
            raise ValueError("A pipeline must have an Estimator as the last component in component_graph.")

        self._validate_problem_types(self.problem_types)

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
    def summary(self):
        """Returns a short summary of the pipeline structure, describing the list of components used.
        Example: Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder
        """
        return self._generate_summary()

    def _generate_summary(self):
        if self.estimator is not None:
            summary = "{}".format(self.estimator.name)
        else:
            summary = "Pipeline"
        for index, component in enumerate(self.component_graph[:-1]):
            if index == 0:
                summary += " w/ {}".format(component.name)
            else:
                summary += " + {}".format(component.name)

        return summary

    def _validate_problem_types(self, problem_types):
        """Validates provided `problem_types` against the estimator in `self.component_graph`

        Arguments:
            problem_types (list): list of ProblemTypes
        """
        estimator_problem_types = self.estimator.problem_types
        for problem_type in self.problem_types:
            if problem_type not in estimator_problem_types:
                raise ValueError("Problem type {} not valid for this component graph. Valid problem types include {}.".format(problem_type, estimator_problem_types))

    def _instantiate_component(self, component, parameters):
        """Instantiates components with parameters in `parameters`"""
        component = handle_component(component)
        component_class = component.__class__
        component_name = component.name
        try:
            component_parameters = parameters.get(component_name, {})
            new_component = component_class(**component_parameters)
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

    def describe(self, return_dict=False):
        """Outputs pipeline details including component parameters

        Arguments:
            return_dict (bool): If True, return dictionary of information about pipeline. Defaults to false

        Returns:
            dict: dictionary of all component parameters if return_dict is True, else None
        """
        self.logger.log_title(self.name)
        self.logger.log("Problem Types: {}".format(', '.join([str(problem_type) for problem_type in self.problem_types])))
        self.logger.log("Model Type: {}".format(str(self.model_type)))
        better_string = "lower is better"
        if self.objective.greater_is_better:
            better_string = "greater is better"
        objective_string = "Objective to Optimize: {} ({})".format(self.objective.name, better_string)
        self.logger.log(objective_string)

        if self.estimator.name in self.input_feature_names:
            self.logger.log("Number of features: {}".format(len(self.input_feature_names[self.estimator.name])))

        # Summary of steps
        self.logger.log_subtitle("Pipeline Steps")
        for number, component in enumerate(self.component_graph, 1):
            component_string = str(number) + ". " + component.name
            self.logger.log(component_string)
            component.describe(print_name=False)

        if return_dict:
            return self.parameters

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

    def fit(self, X, y, objective_fit_size=.2):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

        Returns:

            self

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if self.objective.needs_fitting:
            X, X_objective, y, y_objective = train_test_split(X, y, test_size=objective_fit_size, random_state=self.estimator.random_state)

        self._fit(X, y)

        if self.objective.needs_fitting:
            y_predicted = self.predict_proba(X_objective)

            if self.objective.uses_extra_columns:
                self.objective.fit(y_predicted, y_objective, X_objective)
            else:
                self.objective.fit(y_predicted, y_objective)
        return self

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]

        Returns:
            pd.Series : estimated labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_t = self._transform(X)

        if self.objective and self.objective.needs_fitting:
            y_predicted = self.predict_proba(X)

            if self.objective.uses_extra_columns:
                return self.objective.predict(y_predicted, X)

            return self.objective.predict(y_predicted)

        return self.estimator.predict(X_t)

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]

        Returns:
            pd.DataFrame : probability estimates
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = self._transform(X)
        proba = self.estimator.predict_proba(X)

        if proba.shape[1] <= 2:
            return proba[:, 1]
        else:
            return proba

    def score(self, X, y, other_objectives=None):
        """Evaluate model performance on current and additional objectives

        Args:
            X (pd.DataFrame or np.array) : data of shape [n_samples, n_features]
            y (pd.Series) : true labels of length [n_samples]
            other_objectives (list): list of other objectives to score

        Returns:
            float, dict:  score, ordered dictionary of other objective scores
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        other_objectives = other_objectives or []
        other_objectives = [get_objective(o) for o in other_objectives]
        y_predicted = None
        y_predicted_proba = None

        scores = []
        for objective in [self.objective] + other_objectives:
            if objective.score_needs_proba:
                if y_predicted_proba is None:
                    y_predicted_proba = self.predict_proba(X)
                y_predictions = y_predicted_proba
            else:
                if y_predicted is None:
                    y_predicted = self.predict(X)
                y_predictions = y_predicted

            if objective.uses_extra_columns:
                scores.append(objective.score(y_predictions, y, X))
            else:
                scores.append(objective.score(y_predictions, y))
        if not other_objectives:
            return scores[0], {}

        other_scores = OrderedDict(zip([n.name for n in other_objectives], scores[1:]))

        return scores[0], other_scores

    def graph(self, filepath=None):
        """Generate an image representing the pipeline graph

        Arguments:
            filepath (str, optional) : Path to where the graph should be saved. If set to None (as by default), the graph will not be saved.

        Returns:
            graphviz.Digraph: Graph object that can be directly displayed in Jupyter notebooks.
        """
        return make_pipeline_graph(self.component_graph, self.name, filepath=filepath)

    @property
    def model_type(self):
        """Returns model family of this pipeline template"""
        return self.estimator.model_type

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
