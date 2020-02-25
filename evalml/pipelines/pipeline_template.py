from .components import Estimator, handle_component

from evalml.problem_types import handle_problem_types
from evalml.utils import Logger


class PipelineTemplate:
    def __init__(self, component_graph, supported_problem_types):
        self.component_graph = [handle_component(component) for component in component_graph]
        self.estimator = self.component_graph[-1] if isinstance(self.component_graph[-1], Estimator) else None
        self.supported_problem_types = [handle_problem_types(problem_type) for problem_type in supported_problem_types]
        self.name = self._generate_name()
        self.logger = Logger()

        # check if one and only estimator in pipeline is the last element in component_list
        if not isinstance(self.component_graph[-1], Estimator):
            raise ValueError("A pipeline must have an Estimator as the last component in component_list.")

        self._validate_problem_types(self.supported_problem_types)

    def _generate_name(cls):
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

    @property
    def description(self):
        """Outputs pipeline details including component parameters"""
        self.logger.log_title(self.name)
        self.logger.log("Supported Problem Types: {}".format(', '.join([str(problem_type) for problem_type in self.supported_problem_types])))
        self.logger.log("Model Family: {}".format(str(self.model_family)))

        # Summary of steps
        self.logger.log_subtitle("Pipeline Steps")
        for number, component in enumerate(self.component_graph, 1):
            component_string = str(number) + ". " + component.name
            self.logger.log(component_string)
            component.describe(print_name=False, print_parameters=False)
