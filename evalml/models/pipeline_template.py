
class PipelineTemplate:
    

    def __init__(self, component_list):
        self.component_list = component_list
        # todo: also problem type, etc.
        # go through list and find estimator to get all the goodies

    def get_hyperparameters(self):
        hyperparameter_ranges = {}
        for component in self.component_list:
            hyperparameter_ranges.update(component.hyperparameter_ranges)
        return hyperparameter_ranges


    # def generate_pipeline():
    #     # tbd... go through component list + generate? or do this in autobase?
    #     pass