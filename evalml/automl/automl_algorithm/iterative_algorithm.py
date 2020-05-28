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
                 pipelines_per_batch=5,
                 n_jobs=-1,  # TODO remove
                 number_features=None):  # TODO remove
        """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance.

        Arguments:
            objective (ObjectiveBase): An objective which defines the problem type and whether larger or smaller scores are more optimal
            max_pipelines (int): The maximum number of pipelines to be evaluated.
            allowed_model_families (list(str, ModelFamily)): The model families enabled in the search. The default value of None indicates all model families are allowed.
            allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search. The default of None indicates all pipelines for this problem type are allowed.
            tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
            pipelines_per_batch (int): the number of pipelines to be evaluated in each batch, after the first batch.
            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            number_features (int): The number of columns in the input features.
        """
        super().__init__(objective=objective,
                         max_pipelines=max_pipelines,
                         allowed_model_families=allowed_model_families,
                         allowed_pipelines=allowed_pipelines,
                         tuner_class=tuner_class,
                         random_state=random_state)
        self.pipelines_per_batch = pipelines_per_batch
        self.n_jobs = n_jobs
        self.number_features = number_features
        self._first_batch_results = []

    def _can_continue(self):
        """Are there more pipelines to evaluate?"""
        max_pipeline_check = self._pipeline_number < (self.max_pipelines or np.inf)
        return max_pipeline_check and self._batch_number <= len(self.allowed_pipelines)

    def next_batch(self):
        """Get the next batch of pipelines to evaluate

        Returns:
            list(PipelineBase): a list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """
        if not self._can_continue():
            raise StopIteration('No more batches available.')
        next_batch = []
        if self._batch_number == 0:
            next_batch = [pipeline_class(parameters=self._transform_parameters(pipeline_class, {}))
                          for pipeline_class in self.allowed_pipelines]
        else:
            _, pipeline_class = self._pop_best_in_batch()
            if pipeline_class is None:
                raise AutoMLAlgorithmException('Some results are needed before the next automl batch can be computed.')
            for i in range(self.pipelines_per_batch):
                proposed_parameters = self._tuners[pipeline_class.name].propose()
                next_batch.append(pipeline_class(parameters=self._transform_parameters(pipeline_class, proposed_parameters)))
        self._pipeline_number += len(next_batch)
        self._batch_number += 1
        return next_batch

    def add_result(self, score, pipeline):
        """Register results from evaluating a pipeline

        Arguments:
            score (float): The score obtained by this pipeline on the primary objective.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
        """
        super().add_result(score, pipeline)
        if self.batch_number <= 1:
            # use score_to_minimize so we can use one comparator in _get_best_in_batch
            score_to_minimize = -score if self.objective.greater_is_better else score
            self._first_batch_results.append((score_to_minimize, pipeline.__class__))

    def _pop_best_in_batch(self):
        """Pop the item remaining in the results from the first batch with the best primary objective score."""
        if len(self._first_batch_results) == 0:
            return None, None
        # argmax by score
        best_idx = 0
        for idx, (score, _) in enumerate(self._first_batch_results):
            if score < self._first_batch_results[best_idx][0]:
                best_idx = idx
        return self._first_batch_results.pop(best_idx)

    def _transform_parameters(self, pipeline_class, proposed_parameters):
        """Given a pipeline parameters dict, make sure n_jobs and number_features are set."""
        parameters = {}
        component_graph = [handle_component(c) for c in pipeline_class.component_graph]
        for component in component_graph:
            component_parameters = proposed_parameters.get(component.name, {})
            init_params = inspect.signature(component.__class__.__init__).parameters

            # Inspects each component and adds the following parameters when needed
            if 'n_jobs' in init_params:
                component_parameters['n_jobs'] = self.n_jobs
            if 'number_features' in init_params:
                component_parameters['number_features'] = self.number_features
            parameters[component.name] = component_parameters
        return parameters
