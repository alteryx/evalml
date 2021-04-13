from evalml.pipelines import BinaryClassificationPipeline


class BaselineBinaryPipeline(BinaryClassificationPipeline):
    """Baseline Pipeline for binary classification."""
    custom_name = "Baseline Classification Pipeline"
    component_graph = ["Baseline Classifier"]

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph,
                                custom_name=self.custom_name,
                                parameters=parameters,
                                custom_hyperparameters=self.custom_hyperparameters,
                                random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)


class ModeBaselineBinaryPipeline(BinaryClassificationPipeline):
    """Mode Baseline Pipeline for binary classification."""
    custom_name = "Mode Baseline Binary Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = {"strategy": ["mode"]}

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph,
                                custom_name=self.custom_name,
                                parameters=parameters,
                                custom_hyperparameters=self.custom_hyperparameters,
                                random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)
