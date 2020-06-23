from evalml.pipelines import MulticlassClassificationPipeline


class XGBoostMulticlassPipeline(MulticlassClassificationPipeline):
    """XGBoost Pipeline for multiclass classification."""
    custom_name = "XGBoost Multiclass Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'XGBoost Classifier']
