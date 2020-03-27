from evalml.pipelines import PipelineBase


class RFClassificationPipeline(PipelineBase):
    """Random Forest Pipeline for both binary and multiclass classification"""
    _name = "Random Forest Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Random Forest Classifier']
    supported_problem_types = ['binary', 'multiclass']
