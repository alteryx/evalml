from evalml.pipelines import RegressionPipeline


class ENRegressionPipeline(RegressionPipeline):
    """Elastic Net Pipeline for regression problems"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Regressor Select From Model', 'Elastic Net Regressor']
