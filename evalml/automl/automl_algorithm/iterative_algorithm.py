import inspect
from operator import itemgetter

from .automl_algorithm import AutoMLAlgorithm, AutoMLAlgorithmException

from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    StackedEnsembleClassifier,
    StackedEnsembleRegressor
)
from evalml.pipelines.components.utils import handle_component_class
from evalml.pipelines.utils import make_stacked_ensemble_pipeline
from evalml.problem_types import ProblemTypes


class IterativeAlgorithm(AutoMLAlgorithm):
    """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance."""

    def __init__(self,
                 allowed_pipelines=None,
                 max_iterations=None,
                 tuner_class=None,
                 random_state=0,
                 pipelines_per_batch=5,
                 n_jobs=-1,  # TODO remove
                 number_features=None):  # TODO remove
        """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance.

        Arguments:
            allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search. The default of None indicates all pipelines for this problem type are allowed.
            max_iterations (int): The maximum number of iterations to be evaluated.
            tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
            pipelines_per_batch (int): the number of pipelines to be evaluated in each batch, after the first batch.
            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            number_features (int): The number of columns in the input features.
        """
        super().__init__(allowed_pipelines=allowed_pipelines,
                         max_iterations=max_iterations,
                         tuner_class=tuner_class,
                         random_state=random_state)
        self.pipelines_per_batch = pipelines_per_batch
        self.n_jobs = n_jobs
        self.number_features = number_features
        self._first_batch_results = []

    def next_batch(self):
        """Get the next batch of pipelines to evaluate

        Returns:
            list(PipelineBase): a list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """
        if self._batch_number == 1:
            if len(self._first_batch_results) == 0:
                raise AutoMLAlgorithmException('No results were reported from the first batch')
            self._first_batch_results = sorted(self._first_batch_results, key=itemgetter(0))

        next_batch = []
        
        # for i in range(len(self._first_batch_results)):
        #     pipeline_class = self._first_batch_results[i][1]
        #     if pipeline_class.model_family != ModelFamily.NONE and pipeline_class.model_famiy != ModelFamily.BASELINE
        # has_stackable = [self._first_batch_results[i][1].model_family for i in self._first_batch_results if ]
        # print ("mod math:", self.batch_number, (self._batch_number) % (len(self._first_batch_results) + 1) )
        if self._batch_number == 0:
            next_batch = [pipeline_class(parameters=self._transform_parameters(pipeline_class, {}))
                          for pipeline_class in self.allowed_pipelines]

        # One after training all pipelines one round
        elif len(self._first_batch_results) > 1 and self._batch_number != 1 and (self._batch_number) % (len(self._first_batch_results) + 1) == 1:
            input_pipelines = []
            for i in range(len(self._first_batch_results)):
                pipeline_class = self._first_batch_results[i][1]
                proposed_parameters = self._tuners[pipeline_class.name].propose()
                input_pipelines.append(pipeline_class(parameters=self._transform_parameters(pipeline_class, proposed_parameters)))
            ensemble = make_stacked_ensemble_pipeline(input_pipelines, input_pipelines[0].problem_type)
            next_batch.append(ensemble)
        else:
            idx = (self._batch_number - 1) % len(self._first_batch_results)
            pipeline_class = self._first_batch_results[idx][1]
            for i in range(self.pipelines_per_batch):
                proposed_parameters = self._tuners[pipeline_class.name].propose()
                next_batch.append(pipeline_class(parameters=self._transform_parameters(pipeline_class, proposed_parameters)))
        self._pipeline_number += len(next_batch)
        self._batch_number += 1
        return next_batch

    def add_result(self, score_to_minimize, pipeline):
        """Register results from evaluating a pipeline

        Arguments:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
        """
        if pipeline.model_family != ModelFamily.ENSEMBLE:
            super().add_result(score_to_minimize, pipeline)
        if self.batch_number == 1:
            self._first_batch_results.append((score_to_minimize, pipeline.__class__))

    def _transform_parameters(self, pipeline_class, proposed_parameters):
        """Given a pipeline parameters dict, make sure n_jobs and number_features are set."""
        parameters = {}
        component_graph = [handle_component_class(c) for c in pipeline_class.component_graph]
        for component_class in component_graph:
            component_parameters = proposed_parameters.get(component_class.name, {})
            init_params = inspect.signature(component_class.__init__).parameters

            # Inspects each component and adds the following parameters when needed
            if 'n_jobs' in init_params:
                component_parameters['n_jobs'] = self.n_jobs
            if 'number_features' in init_params:
                component_parameters['number_features'] = self.number_features
            parameters[component_class.name] = component_parameters
        return parameters
