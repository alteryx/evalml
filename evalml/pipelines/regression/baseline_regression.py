from evalml.pipelines import RegressionPipeline


class BaselineRegressionPipeline(RegressionPipeline):
    """Baseline Pipeline for regression problems."""
    custom_name = "Baseline Regression Pipeline"
    component_graph = ["Baseline Regressor"]

    def __init__(self, parameters):
        return super().__init__(self.component_graph, None, parameters)

    def new(self, parameters, random_seed):
        return self.__class__(self.parameters)

    def clone(self):
        return self.__class__(self.parameters)


class MeanBaselineRegressionPipeline(RegressionPipeline):
    """Baseline Pipeline for regression problems."""
    custom_name = "Mean Baseline Regression Pipeline"
    component_graph = ["Baseline Regressor"]
    custom_hyperparameters = {"strategy": ["mean"]}

    def __init__(self, parameters):
        return super().__init__(self.component_graph, None, parameters)

    def new(self, parameters, random_seed):
        return self.__class__(self.parameters)

    def clone(self):
        return self.__class__(self.parameters)
