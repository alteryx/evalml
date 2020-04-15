from evalml.pipelines import BinaryClassificationPipeline


class XGBoostBinaryPipeline(BinaryClassificationPipeline):
    """XGBoost Pipeline for both binary and multiclass classification"""
    custom_name = "XGBoost Binary Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'XGBoost Classifier']

    def __init__(self, parameters, random_state=0):
        super().__init__(parameters=parameters,
                         random_state=random_state)
