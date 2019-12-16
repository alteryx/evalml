
class PipelineTemplate:
    

    def __init__(self, component_list):
        self.component_list = component_list
        self.estimator = self.component_list[-1]
        self.name = self._generate_name()
        self.problem_types = self.estimator.problem_types
        self.model_type = self.estimator.model_type

        # todo: also problem type, etc.
        # go through list and find estimator to get all the goodies

    def get_hyperparameters(self):
        hyperparameter_ranges = {}
        for component in self.component_list:
            hyperparameter_ranges.update(component.hyperparameter_ranges)
        return hyperparameter_ranges

    def _generate_name(self):
        if self.estimator is not None:
            name = "{}".format(self.estimator.name)
        else:
            name = "Pipeline"
        for index, component in enumerate(self.component_list[:-1]):
            if index == 0:
                name += " w/ {}".format(component.name)
            else:
                name += " + {}".format(component.name)

        return name   