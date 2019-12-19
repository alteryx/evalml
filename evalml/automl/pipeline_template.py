from evalml.pipelines import PipelineBase


class PipelineTemplate:

    def __init__(self, component_list):
        self.component_list = component_list
        self.estimator = self.component_list[-1]
        self.name = self._generate_name()
        self.problem_types = self.estimator.problem_types
        self.model_type = self.estimator.model_type

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

    def get_hyperparameters(self):
        """
        Gets hyperparameter space
        """
        hyperparameter_ranges = {}
        for component in self.component_list:
            hyperparameter_ranges.update(component.hyperparameter_ranges)
        return hyperparameter_ranges

    def get_comp_to_hyperparameters_names(self):
        hyperparameter_ranges = {}
        for component in self.component_list:
            hyperparameter_ranges.update({component.name: list(component.hyperparameter_ranges.keys())})
        return hyperparameter_ranges

    def generate_pipeline_with_params(self, objective, parameters, random_state):
        """
        Generate pipeline with default or specified parameters

        Arguments:
            parameters (dict)
        """
        component_objs = []
        comp_to_hyperparams = self.get_comp_to_hyperparameters_names()
        for c in self.component_list:
            component_params = comp_to_hyperparams[c.name]
            relevant_params = {}
            for (param_name, param_value) in parameters:
                if param_name in component_params:
                    relevant_params.update({param_name: param_value})
            obj = c(**dict(relevant_params))
            component_objs.append(obj)

        pipeline = PipelineBase(objective=objective,
                                n_jobs=-1,
                                component_list=component_objs,
                                random_state=random_state)

        return pipeline
