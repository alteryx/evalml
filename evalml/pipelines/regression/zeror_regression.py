from evalml.pipelines import RegressionPipeline


class ZeroRRegressionPipeline(RegressionPipeline):
    """ZeroR Pipeline for regression problems"""
    _name = "ZeroR Regression Pipeline"
    component_graph = ["ZeroR Regressor"]
