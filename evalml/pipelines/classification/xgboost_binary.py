from evalml.pipelines import BinaryClassificationPipeline


class XGBoostBinaryPipeline(BinaryClassificationPipeline):
    """XGBoost Pipeline for both binary and multiclass classification"""
    _name = "XGBoost Binary Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'XGBoost Classifier']
    supported_problem_types = ['binary']

    def __init__(self, parameters):
        super().__init__(parameters=parameters)
