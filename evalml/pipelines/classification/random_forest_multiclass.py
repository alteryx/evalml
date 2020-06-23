from evalml.pipelines import MulticlassClassificationPipeline


class RFMulticlassClassificationPipeline(MulticlassClassificationPipeline):
    """Random Forest Pipeline for multiclass classification."""
    custom_name = "Random Forest Multiclass Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Random Forest Classifier']
