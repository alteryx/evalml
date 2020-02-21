from .components import Estimator, handle_component

from evalml.utils import Logger
from evalml.problem_types import handle_problem_types

class PipelineTemplate:
    def __init__(self, component_graph, supported_problem_types):
        self.component_graph = [handle_component(component) for component in component_graph]
        self.estimator = self.component_graph[-1] if isinstance(self.component_graph[-1], Estimator) else None
        self.supported_problem_types = [handle_problem_types(problem_type) for problem_type in supported_problem_types]
        self.name = self._generate_name()
        self.logger = Logger()

        self._validate_problem_types(self.supported_problem_types)
        
    def _generate_name(self):
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

    def _validate_problem_types(self, supported_problem_types):
        estimator_problem_types = self.estimator.problem_types
        for problem_type in self.supported_problem_types:
            if problem_type not in estimator_problem_types:
                raise ValueError("Supported problem type {} not valid for this component graph. Valid problem types include {}.".format(problem_type, estimator_problem_types))

    @property
    def model_family(self):
        """Returns model family of this pipeline template"""

        # TODO: Refactor to model_family
        # In future there potentially could be multiple estimators

        return self.estimator.model_type

    def describe(self, return_dict):
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
        for number, component in enumerate(self.component_list, 1):
            component_string = str(number) + ". " + component.name
            self.logger.log(component_string)
            component.describe(print_name=False)

        if return_dict:
            return self.parameters