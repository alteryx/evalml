from evalml.pipelines import RegressionPipeline


class ENRegressionPipeline(RegressionPipeline):
    """Elastic Net Pipeline for regression problems"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Elastic Net Regressor']
