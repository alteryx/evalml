from evalml.pipelines import MulticlassClassificationPipeline


class ModeBaselineMulticlassPipeline(MulticlassClassificationPipeline):
    """"Mode Baseline Pipeline for multiclass classification"""
    custom_name = "Mode Baseline Multiclass Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = {"strategy": ["mode"]}
