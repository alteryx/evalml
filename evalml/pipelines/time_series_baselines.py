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
        super().__init__(self.component_graph,
                         custom_name=self.custom_name,
                         parameters=parameters,
                         custom_hyperparameters=None,
                         random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)


class TimeSeriesBaselineBinaryPipeline(TimeSeriesBinaryClassificationPipeline):
    """Baseline Pipeline for time series binary classification problems."""
    custom_name = "Time Series Baseline Binary Pipeline"
    component_graph = ["Time Series Baseline Estimator"]

    def __init__(self, parameters, random_seed=0):
        super().__init__(self.component_graph,
                         custom_name=self.custom_name,
                         parameters=parameters,
                         custom_hyperparameters=None,
                         random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)


class TimeSeriesBaselineMulticlassPipeline(TimeSeriesMulticlassClassificationPipeline):
    """Baseline Pipeline for time series multiclass classification problems."""
    custom_name = "Time Series Baseline Multiclass Pipeline"
    component_graph = ["Time Series Baseline Estimator"]

    def __init__(self, parameters, random_seed=0):
        super().__init__(self.component_graph,
                         custom_name=self.custom_name,
                         parameters=parameters,
                         custom_hyperparameters=None,
                         random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)
