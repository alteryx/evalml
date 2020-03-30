from evalml.pipelines import BinaryClassificationPipeline


class RFBinaryClassificationPipeline(BinaryClassificationPipeline):
    """Random Forest Pipeline for binary classification"""
    _name = "Random Forest Binary Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Random Forest Classifier']
    supported_problem_types = ['binary']
