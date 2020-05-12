from evalml.pipelines import BinaryClassificationPipeline


class BaselineBinaryPipeline(BinaryClassificationPipeline):
    """"Baseline Pipeline for binary classification"""
    custom_name = "Baseline Classification Pipeline"
    component_graph = ["Baseline Classifier"]
