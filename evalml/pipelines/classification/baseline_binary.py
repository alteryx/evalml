from evalml.pipelines import BinaryClassificationPipeline


class BaselineBinaryPipeline(BinaryClassificationPipeline):
    """Baseline Pipeline for binary classification."""
    custom_name = "Baseline Binary Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = None

    def __init__(self, parameters, random_seed=0):
        super().__init__(self.component_graph,
                         custom_name=self.custom_name,
                         parameters=parameters,
                         custom_hyperparameters=self.custom_hyperparameters,
                         random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)


class ModeBaselineBinaryPipeline(BaselineBinaryPipeline):
    """Mode Baseline Pipeline for binary classification."""
    custom_name = "Mode Baseline Binary Classification Pipeline"
    custom_hyperparameters = {"strategy": ["mode"]}
