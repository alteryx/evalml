import inspect

import numpy as np

from .automl_algorithm import AutoMLAlgorithm, AutoMLAlgorithmException

from evalml.pipelines.components import handle_component


class IterativeAlgorithm(AutoMLAlgorithm):
    """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance."""

    def __init__(self,
                 objective,
                 max_pipelines=None,
                 allowed_model_families=None,
                 allowed_pipelines=None,
                 tuner_class=None,
                 random_state=0,
                 samples_per_batch=5,
                 n_jobs=-1,  # TODO remove
                 number_features=None):  # TODO remove
        super().__init__(objective=objective,
                         max_pipelines=max_pipelines,
                         allowed_model_families=allowed_model_families,
                         allowed_pipelines=allowed_pipelines,
                         tuner_class=tuner_class,
                         random_state=random_state)
        self.samples_per_batch = samples_per_batch
        self.n_jobs = n_jobs
        self.number_features = number_features
        self._first_batch_results = []

    def can_continue(self):
        max_pipeline_check = self._pipeline_number < (self.max_pipelines or np.inf)
        return max_pipeline_check and self._batch_number <= len(self.allowed_pipelines)

    def next_batch(self):
        if not self.can_continue():
            raise StopIteration('No more batches available')
        next_batch = []
        if self._batch_number == 0:
            next_batch = [self._init_pipeline(cls) for cls in self.allowed_pipelines]
        else:
            _, pipeline_class = self._pop_best_in_batch()
            if pipeline_class is None:
                raise AutoMLAlgorithmException('Some results are needed before the next automl batch can be computed.')
            next_batch = [self._init_pipeline(pipeline_class) for i in range(self.samples_per_batch)]
        self._pipeline_number += len(next_batch)
        self._batch_number += 1
        return next_batch

    def add_result(self, score, pipeline):
        super().add_result(score, pipeline)
        if self.batch_number <= 1:
            # use score_to_minimize so we can use one comparator in _get_best_in_batch
            score_to_minimize = -score if self.objective.greater_is_better else score
            self._first_batch_results.append((score_to_minimize, pipeline.__class__))

    def _pop_best_in_batch(self):
        if len(self._first_batch_results) == 0:
            return None, None
        # argmax by score
        best_idx = 0
        for idx, (score, _) in enumerate(self._first_batch_results):
            if score < self._first_batch_results[best_idx][0]:
                best_idx = idx
        return self._first_batch_results.pop(best_idx)

    def _init_pipeline(self, pipeline_class):
        proposed_parameters = self._tuners[pipeline_class.name].propose()
        parameters = self._transform_parameters(pipeline_class, proposed_parameters)
        return pipeline_class(parameters=parameters)

    def _transform_parameters(self, pipeline_class, proposed_parameters):
        parameters = {}
        component_graph = [handle_component(c) for c in pipeline_class.component_graph]
        for component in component_graph:
            component_parameters = proposed_parameters[component.name]
            init_params = inspect.signature(component.__class__.__init__).parameters

            # Inspects each component and adds the following parameters when needed
            if 'n_jobs' in init_params:
                component_parameters['n_jobs'] = self.n_jobs
            if 'number_features' in init_params:
                component_parameters['number_features'] = self.number_features
            parameters[component.name] = component_parameters
        return parameters
