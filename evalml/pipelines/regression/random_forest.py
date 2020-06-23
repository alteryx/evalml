from evalml.pipelines import RegressionPipeline


class RFRegressionPipeline(RegressionPipeline):
    """Random Forest Pipeline for regression problems."""
    custom_name = "Random Forest Regression Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Random Forest Regressor']
