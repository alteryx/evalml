from evalml.pipelines import RegressionPipeline


class ETRegressionPipeline(RegressionPipeline):
    """Extra Trees Pipeline for regression problems."""
    custom_name = "Extra Trees Regression Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Extra Trees Regressor']
