from evalml.pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline
)


class TimeSeriesBaselineRegressionPipeline(TimeSeriesRegressionPipeline):
    """Baseline Pipeline for time series regression problems."""
    _name = "Time Series Baseline Regression Pipeline"
    component_graph = ["Time Series Baseline Estimator"]


class TimeSeriesBaselineBinaryPipeline(TimeSeriesBinaryClassificationPipeline):
    """Baseline Pipeline for time series binary classification problems."""
    _name = "Time Series Baseline Binary Pipeline"
    component_graph = ["Time Series Baseline Estimator"]


class TimeSeriesBaselineMulticlassPipeline(TimeSeriesMulticlassClassificationPipeline):
    """Baseline Pipeline for time series multiclass classification problems."""
    _name = "Time Series Baseline Multiclass Pipeline"
    component_graph = ["Time Series Baseline Estimator"]
