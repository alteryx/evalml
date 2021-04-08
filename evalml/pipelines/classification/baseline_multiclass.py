from evalml.pipelines import MulticlassClassificationPipeline


class BaselineMulticlassPipeline(MulticlassClassificationPipeline):
    """Baseline Pipeline for multiclass classification."""
    custom_name = "Baseline Multiclass Classification Pipeline"
    component_graph = ["Baseline Classifier"]

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph, self.custom_name, parameters, custom_hyperparameters=self.custom_hyperparameters, random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(self.parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters)


class ModeBaselineMulticlassPipeline(MulticlassClassificationPipeline):
    """Mode Baseline Pipeline for multiclass classification."""
    custom_name = "Mode Baseline Multiclass Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = {"strategy": ["mode"]}

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph, self.custom_name, parameters, custom_hyperparameters=self.custom_hyperparameters, random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(self.parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters)
