"""Ensemble pipeline mix-in class."""
from evalml.pipelines.pipeline_base import PipelineBase
from evalml.pipelines.pipeline_meta import PipelineBaseMeta


class EnsemblePipelineBase(PipelineBase, metaclass=PipelineBaseMeta):
    def __init__(
        self,
        input_pipelines,
        component_graph,
        parameters=None,
        custom_name=None,
        random_seed=0,
    ):
        self.input_pipelines = input_pipelines
        self._is_stacked_ensemble = True

        super().__init__(
            component_graph,
            custom_name=custom_name,
            parameters=parameters,
            random_seed=random_seed,
        )

    @property
    def _all_input_pipelines_fitted(self):
        for pipeline in self.input_pipelines:
            if not pipeline._is_fitted:
                return False
        return True

    def _fit_input_pipelines(self, X, y, force_retrain=False):
        fitted_pipelines = []
        for pipeline in self.input_pipelines:
            if pipeline._is_fitted and not force_retrain:
                fitted_pipelines.append(pipeline)
            else:
                if force_retrain:
                    new_pl = pipeline.clone()
                else:
                    new_pl = pipeline
                fitted_pipelines.append(new_pl.fit(X, y))
        self.input_pipelines = fitted_pipelines

    def clone(self):
        """Constructs a new pipeline with the same components, parameters, and random seed.

        Returns:
            A new instance of this pipeline with identical components, parameters, and random seed.
        """
        clone = self.__class__(
            input_pipelines=self.input_pipelines,
            component_graph=self.component_graph,
            parameters=self.parameters,
            custom_name=self.custom_name,
            random_seed=self.random_seed,
        )
        return clone

    def new(self, parameters, random_seed=0):
        """Constructs a new instance of the pipeline with the same component graph but with a different set of parameters. Not to be confused with python's __new__ method.

        Args:
            parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
                 An empty dictionary or None implies using all default values for component parameters. Defaults to None.
            random_seed (int): Seed for the random number generator. Defaults to 0.

        Returns:
            A new instance of this pipeline with identical components.
        """
        return self.__class__(
            self.input_pipelines,
            self.component_graph,
            parameters=parameters,
            custom_name=self.custom_name,
            random_seed=random_seed,
        )
