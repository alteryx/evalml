from evalml.pipelines import MulticlassClassificationPipeline


class ETMulticlassClassificationPipeline(MulticlassClassificationPipeline):
    """Extra Trees Pipeline for multiclass classification"""
    custom_name = "Extra Trees Multiclass Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Extra Trees Classifier']
