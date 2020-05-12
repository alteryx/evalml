from evalml.pipelines import BinaryClassificationPipeline


class ZeroRBinaryPipeline(BinaryClassificationPipeline):
    """"ZeroR Pipeline for binary classification"""
    custom_name = "ZeroR Classification Pipeline"
    component_graph = ["ZeroR Classifier"]
