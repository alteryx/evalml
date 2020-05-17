from evalml.pipelines import RegressionPipeline


class MeanBaselineRegressionPipeline(RegressionPipeline):
    """Baseline Pipeline for regression problems"""
    _name = "Mean Baseline Regression Pipeline"
    component_graph = ["Baseline Regressor"]
    custom_hyperparameters = {"strategy": ["mean"]}
