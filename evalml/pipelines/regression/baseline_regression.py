from evalml.pipelines import RegressionPipeline


class BaselineRegressionPipeline(RegressionPipeline):
    """Baseline Pipeline for regression problems"""
    _name = "Baseline Regression Pipeline"
    component_graph = ["Baseline Regressor"]
