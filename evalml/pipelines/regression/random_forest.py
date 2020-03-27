from evalml.pipelines import PipelineBase


class RFRegressionPipeline(PipelineBase):
    """Random Forest Pipeline for regression problems"""
    _name = "Random Forest Regression Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Regressor Select From Model', 'Random Forest Regressor']
    supported_problem_types = ['regression']
