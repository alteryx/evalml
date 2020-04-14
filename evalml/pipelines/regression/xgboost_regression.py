from evalml.pipelines import RegressionPipeline


class XGBoostRegressionPipeline(RegressionPipeline):
    """XGBoost Pipeline for regression problems"""
    _name = "XGBoost Regression Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Regressor Select From Model', 'XGBoost Regressor']

    def __init__(self, parameters, random_state=0):
        super().__init__(parameters=parameters,
                         random_state=random_state)
