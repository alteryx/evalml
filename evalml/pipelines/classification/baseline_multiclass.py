from evalml.pipelines import MulticlassClassificationPipeline


class BaselineMulticlassPipeline(MulticlassClassificationPipeline):
    """Baseline Pipeline for multiclass classification."""
    custom_name = "Baseline Multiclass Classification Pipeline"
    component_graph = ["Baseline Classifier"]

    def __init__(self, parameters):
        return super().__init__(self.component_graph, self.custom_name, {})


class ModeBaselineMulticlassPipeline(MulticlassClassificationPipeline):
    """Mode Baseline Pipeline for multiclass classification."""
    custom_name = "Mode Baseline Multiclass Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = {"strategy": ["mode"]}

    def __init__(self, parameters):
        return super().__init__(self.component_graph, self.custom_name, {})
