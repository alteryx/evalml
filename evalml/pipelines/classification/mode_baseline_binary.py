from evalml.pipelines import BinaryClassificationPipeline


class ModeBaselineBinaryPipeline(BinaryClassificationPipeline):
    """"Mode Baseline Pipeline for binary classification"""
    custom_name = "Mode Baseline Classification Pipeline"
    component_graph = ["Baseline Classifier"]
    custom_hyperparameters = {"strategy": ["mode"]}
