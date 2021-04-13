from evalml.pipelines import RegressionPipeline


class BaselineRegressionPipeline(RegressionPipeline):
    """Baseline Pipeline for regression problems."""
    custom_name = "Baseline Regression Pipeline"
    component_graph = ["Baseline Regressor"]

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph,
                                custom_name=self.custom_name,
                                parameters=parameters,
                                custom_hyperparameters=None,
                                random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)


class MeanBaselineRegressionPipeline(RegressionPipeline):
    """Baseline Pipeline for regression problems."""
    custom_name = "Mean Baseline Regression Pipeline"
    component_graph = ["Baseline Regressor"]
    custom_hyperparameters = {"strategy": ["mean"]}

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph,
                                custom_name=self.custom_name,
                                parameters=parameters,
                                custom_hyperparameters=self.custom_hyperparameters,
                                random_seed=random_seed)

    def new(self, parameters, random_seed=0):
        return self.__class__(parameters, random_seed=random_seed)

    def clone(self):
        return self.__class__(self.parameters, random_seed=self.random_seed)
