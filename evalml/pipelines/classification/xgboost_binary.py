from evalml.pipelines import BinaryClassificationPipeline


class XGBoostBinaryPipeline(BinaryClassificationPipeline):
    """XGBoost Pipeline for both binary and multiclass classification"""
    custom_name = "XGBoost Binary Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'XGBoost Classifier']
