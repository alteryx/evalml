from evalml.pipelines import PipelineBase


class PipelineTemplate:

    def __init__(self, component_list, parameters=None):
        self.component_list = component_list
        self.estimator = self.component_list[-1]
        self.name = self._generate_name()
        self.problem_types = self.estimator.problem_types
        self.model_type = self.estimator.model_type
        if parameters:
            self.parameters = parameters

    def get_hyperparameters(self):
        hyperparameter_ranges = {}
        for component in self.component_list:
            hyperparameter_ranges.update(component.hyperparameter_ranges)
        return hyperparameter_ranges

    def get_params_to_hyperparameters_names(self):
        hyperparameter_ranges = {}
        for component in self.component_list:
            hyperparameter_ranges.update({component.name: list(component.hyperparameter_ranges.keys())})
        return hyperparameter_ranges

    def generate_pipeline_with_params(self, objective, parameters, random_state):
        component_objs = []
        for c in self.component_list:
            params = self.get_params_to_hyperparameters_names()[c.name]
            # print ("mapping:", c.name, params, parameters)
            relevant_params = {}
            for p in parameters:
                if p[0] in params:
                    relevant_params.update({p})
            obj = c(**dict(relevant_params))
            component_objs.append(obj)

        pipeline = PipelineBase(objective=objective,
                                n_jobs=-1,
                                component_list=component_objs,
                                random_state=random_state)

        return pipeline

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
