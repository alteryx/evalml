from .binary_classification_pipeline import BinaryClassificationPipeline
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline
)
from .regression_pipeline import RegressionPipeline
from .time_series_classification_pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline
)
from .time_series_regression_pipeline import TimeSeriesRegressionPipeline


class GeneratedPipelineBinary(BinaryClassificationPipeline):
    """Generated Pipeline class for Binary Classification Pipelines
    """
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineMulticlass(MulticlassClassificationPipeline):
    """Generated Pipeline class for Multiclass Classification Pipelines
    """
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineRegression(RegressionPipeline):
    """Generated Pipeline class for Regression Pipelines
    """
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineTimeSeriesBinary(TimeSeriesBinaryClassificationPipeline):
    """Generated Pipeline class for Time Series Binary Classification Pipelines
    """
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineTimeSeriesMulticlass(TimeSeriesMulticlassClassificationPipeline):
    """Generated Pipeline class for Time Series Multiclass Classification Pipelines
    """
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineTimeSeriesRegression(TimeSeriesRegressionPipeline):
    """Generated Pipeline class for Time Series Regression Pipelines
    """
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None
