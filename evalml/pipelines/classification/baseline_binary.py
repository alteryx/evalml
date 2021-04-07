from evalml.pipelines import BinaryClassificationPipeline


class BaselineBinaryPipeline(BinaryClassificationPipeline):
    """Baseline Pipeline for binary classification."""
    custom_name = "Baseline Classification Pipeline"
    name = "Baseline Classification Pipeline"
    component_graph = ["Baseline Classifier"]

    def __init__(self, parameters):
        return super().__init__(self.component_graph, None, parameters)

    def new(self, parameters, random_seed):
        return self.__class__(self.parameters)

    def clone(self):
        return self.__class__(self.parameters)


class ModeBaselineBinaryPipeline(BinaryClassificationPipeline):
    """Mode Baseline Pipeline for binary classification."""
    name = "Mode Baseline Binary Classification Pipeline"
    custom_name = "Mode Baseline Binary Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = {"strategy": ["mode"]}

    def __init__(self, parameters):
        return super().__init__(self.component_graph, None, parameters)

    def new(self, parameters, random_seed):
        return self.__class__(self.parameters)

    def clone(self):
        return self.__class__(self.parameters)
