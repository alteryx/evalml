from evalml.pipelines import BinaryClassificationPipeline


class XGBoostBinaryPipeline(BinaryClassificationPipeline):
    """XGBoost Pipeline for binary classification"""
    custom_name = "XGBoost Binary Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'XGBoost Classifier']
