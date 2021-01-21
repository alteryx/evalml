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
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineMulticlass(MulticlassClassificationPipeline):
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineRegression(RegressionPipeline):
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineTimeSeriesBinary(TimeSeriesBinaryClassificationPipeline):
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineTimeSeriesMulticlass(TimeSeriesMulticlassClassificationPipeline):
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None


class GeneratedPipelineTimeSeriesRegression(TimeSeriesRegressionPipeline):
    custom_name = ''
    component_graph = []
    custom_hyperparameters = None
