from evalml.pipelines import RegressionPipeline


class BaselineRegressionPipeline(RegressionPipeline):
    """Baseline Pipeline for regression problems."""
    _name = "Baseline Regression Pipeline"
    component_graph = ["Baseline Regressor"]

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph, self._name, {})

class MeanBaselineRegressionPipeline(RegressionPipeline):
    """Baseline Pipeline for regression problems."""
    _name = "Mean Baseline Regression Pipeline"
    component_graph = ["Baseline Regressor"]
    custom_hyperparameters = {"strategy": ["mean"]}

    def __init__(self, parameters, random_seed=0):
        return super().__init__(self.component_graph, self._name, {})
