import copy
import inspect
import re
from abc import ABC, abstractmethod
from collections import OrderedDict

import pandas as pd
from sklearn.model_selection import train_test_split

from .components import Estimator, handle_component
from .pipeline_plots import PipelinePlots

from evalml.objectives import get_objective
from evalml.problem_types import handle_problem_types
from evalml.utils import Logger


class classproperty:
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class PipelineBase(ABC):

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelinePlots

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

    def __init__(self, parameters, objective, random_state=0, n_jobs=-1, number_features=None):
        """Machine learning pipeline made out of transformers and a estimator.

        Required Class Variables:
            component_graph (list): List of components in order. Accepts strings or ComponentBase objects in the list
            problem_types (list): List of problem types for this pipeline. Accepts strings or ProbemType enum in the list.

        Arguments:
            objective (ObjectiveBase): the objective to optimize

            parameters (dict): dictionary with component names as keys and dictionary of that component's parameters as values.
                If `random_state`, `n_jobs`, or 'number_features' are provided as component parameters they will override the corresponding
                value provided as arguments to the pipeline. An empty dictionary {} implies using all default values for component parameters.

            random_state (int): random seed/state. Defaults to 0. `random_state` can also be provided directly to components
                using the parameters dictionary argument.

            n_jobs (int): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
                `n_jobs` can also be provided directly to components using the parameters dictionary argument.

            number_features (int): Number of features in dataset. Defaults to None. `number_features` can also be provided directly to components
                using the parameters dictionary argument.
        """
        self.component_graph = [handle_component(component) for component in self.component_graph]
        self.problem_types = [handle_problem_types(problem_type) for problem_type in self.problem_types]
        self.logger = Logger()
        self.objective = get_objective(objective)
        self.input_feature_names = {}
        self.results = {}
        self.parameters = parameters
        self.plot = PipelinePlots(self)
        self.random_state = random_state
        self.number_features = number_features

        self.n_jobs = n_jobs
        if not isinstance(n_jobs, (int, type(None))) or n_jobs == 0:
            raise ValueError('n_jobs must be an non-zero integer or None. n_jobs is set to `{}`.'.format(n_jobs))

        self._instantiate_components()
        self.estimator = self.component_graph[-1] if isinstance(self.component_graph[-1], Estimator) else None

        # check if one and only estimator in pipeline is the last element in component_graph
        if not isinstance(self.component_graph[-1], Estimator):
            raise ValueError("A pipeline must have an Estimator as the last component in component_graph.")

        self._validate_problem_types(self.problem_types)

    @classproperty
    def name(cls):
        "Returns either `_name` defined on pipeline class or the pipeline class name"
        try:
            name = cls._name
        except AttributeError:
            rex = re.compile(r'(?<=[a-z])(?=[A-Z])')
            name = rex.sub(' ', cls.__name__)
        return name

    @property
    def summary(self):
        "Returns string of pipeline structure: `Logistic Regression Classifier w/ ... + ..."
        return self._generate_summary()

    def _generate_summary(self):
        if self.estimator is not None:
            name = "{}".format(self.estimator.name)
        else:
            name = "Pipeline"
        for index, component in enumerate(self.component_graph[:-1]):
            if index == 0:
                name += " w/ {}".format(component.name)
            else:
                name += " + {}".format(component.name)

        return name

    def _validate_problem_types(self, problem_types):
        estimator_problem_types = self.estimator.problem_types
        for problem_type in self.problem_types:
            if problem_type not in estimator_problem_types:
                raise ValueError("Problem type {} not valid for this component graph. Valid problem types include {}.".format(problem_type, estimator_problem_types))

    def _instantiate_components(self):
        for index, component in enumerate(self.component_graph):
            component_class = component.__class__
            component_name = component.name
            if component_class.hyperparameter_ranges == {}:
                new_component = component_class()
            elif component_name not in self.parameters:
                try:
                    component_parameters = self._check_arguments_and_add(dict(), component_class)
                    new_component = component_class(**component_parameters)
                except TypeError as e:
                    raise ValueError("\nPlease provide the required parameters for {} in the `parameters` dictionary argument.".format(component_name)) from e
            else:
                try:
                    component_parameters = copy.deepcopy(self.parameters[component_name])
                    self._validate_component_parameters(component_class, self.parameters[component_name])

                    # Add random_state, n_jobs and number_features into component parameters if doesn't exist
                    component_parameters = self._check_arguments_and_add(component_parameters, component_class)
                    new_component = component_class(**component_parameters)
                except ValueError as e:
                    raise ValueError("Error received when instantiating component {} with the following arguments {}".format(component_name, self.parameters[component_name])) from e
            self.component_graph[index] = new_component

    def _check_arguments_and_add(self, component_parameters, component_class):
        if 'random_state' in inspect.signature(component_class.__init__).parameters and 'random_state' not in component_parameters:
            component_parameters['random_state'] = self.random_state
        if 'n_jobs' in inspect.signature(component_class.__init__).parameters and 'n_jobs' not in component_parameters:
            component_parameters['n_jobs'] = self.n_jobs
        if 'number_features' in inspect.signature(component_class.__init__).parameters and 'number_features' not in component_parameters:
            component_parameters['number_features'] = self.number_features

        return component_parameters

    def _validate_component_parameters(self, component_class, parameters):
        for parameter, parameter_value in parameters.items():
            if parameter not in inspect.signature(component_class.__init__).parameters:
                raise ValueError("{} is not a hyperparameter of {}".format(parameter, component_class.name))
            if parameter in component_class.hyperparameter_ranges and parameter_value not in component_class.hyperparameter_ranges[parameter]:
                raise ValueError("{} = {} not in hyperparameter range of {}".format(parameter, parameter_value, component_class.name))

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
        self.logger.log_title(type(self).name)
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
            X, X_objective, y, y_objective = train_test_split(X, y, test_size=objective_fit_size, random_state=self.random_state)

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

    @property
    def model_type(self):
        """Returns model family of this pipeline template"""

        # TODO: Refactor to model_family
        # In future there potentially could be multiple estimators

        return self.estimator.model_type

    @property
    def feature_importances(self):
        """Return feature importances. Features dropped by feature selection are excluded"""
        feature_names = self.input_feature_names[self.estimator.name]
        importances = list(zip(feature_names, self.estimator.feature_importances))  # note: this only works for binary
        importances.sort(key=lambda x: -abs(x[1]))
        df = pd.DataFrame(importances, columns=["feature", "importance"])
        return df
