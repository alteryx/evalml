from evalml.pipelines import BinaryClassificationPipeline


class ETBinaryClassificationPipeline(BinaryClassificationPipeline):
    """Extra Trees Pipeline for binary classification"""
    custom_name = "Extra Trees Binary Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Extra Trees Classifier']
