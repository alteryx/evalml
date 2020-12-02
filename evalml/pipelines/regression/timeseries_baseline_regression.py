from evalml.pipelines import TimeSeriesRegressionPipeline


class TimeSeriesBaselineRegressionPipeline(TimeSeriesRegressionPipeline):
    """Baseline Pipeline for time series regression problems."""
    _name = "Time Series Baseline Regression Pipeline"
    component_graph = ["Delayed Feature Transformer", "Time Series Baseline Regressor"]
