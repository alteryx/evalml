from evalml.pipelines import RegressionPipeline


class RFRegressionPipeline(RegressionPipeline):
    """Random Forest Pipeline for regression problems"""
    _name = "Random Forest Regression Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Regressor Select From Model', 'Random Forest Regressor']
    supported_problem_types = ['regression']
