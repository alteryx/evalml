from evalml.pipelines import MulticlassClassificationPipeline


class ZeroRMulticlassPipeline(MulticlassClassificationPipeline):
    """"ZeroR Pipeline for multiclass classification"""
    custom_name = "ZeroR Multiclass Classification Pipeline"
    component_graph = ["ZeroR Classifier"]
