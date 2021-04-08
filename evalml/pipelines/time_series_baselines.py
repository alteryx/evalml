from evalml.pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline
)


class TimeSeriesBaselineRegressionPipeline(TimeSeriesRegressionPipeline):
    """Baseline Pipeline for time series regression problems."""
    custom_name = "Time Series Baseline Regression Pipeline"
    component_graph = ["Time Series Baseline Estimator"]

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph, self.custom_name, parameters, custom_hyperparameters=None, random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(self.parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters)


class TimeSeriesBaselineBinaryPipeline(TimeSeriesBinaryClassificationPipeline):
    """Baseline Pipeline for time series binary classification problems."""
    custom_name = "Time Series Baseline Binary Pipeline"
    component_graph = ["Time Series Baseline Estimator"]

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph, self.custom_name, parameters, custom_hyperparameters=None, random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(self.parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters)


class TimeSeriesBaselineMulticlassPipeline(TimeSeriesMulticlassClassificationPipeline):
    """Baseline Pipeline for time series multiclass classification problems."""
    custom_name = "Time Series Baseline Multiclass Pipeline"
    component_graph = ["Time Series Baseline Estimator"]

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph, self.custom_name, parameters, custom_hyperparameters=None, random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(self.parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters)
