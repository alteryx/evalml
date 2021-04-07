from evalml.pipelines import MulticlassClassificationPipeline


class BaselineMulticlassPipeline(MulticlassClassificationPipeline):
    """Baseline Pipeline for multiclass classification."""
    custom_name = "Baseline Multiclass Classification Pipeline"
    component_graph = ["Baseline Classifier"]

    def __init__(self, parameters):
        return super().__init__(self.component_graph, None, parameters)

    def new(self, parameters, random_seed):
        return self.__class__(self.parameters)

    def clone(self):
        return self.__class__(self.parameters)


class ModeBaselineMulticlassPipeline(MulticlassClassificationPipeline):
    """Mode Baseline Pipeline for multiclass classification."""
    custom_name = "Mode Baseline Multiclass Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = {"strategy": ["mode"]}

    def __init__(self, parameters):
        return super().__init__(self.component_graph, None, parameters)

    def new(self, parameters, random_seed):
        return self.__class__(self.parameters)

    def clone(self):
        return self.__class__(self.parameters)
