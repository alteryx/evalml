import inspect
from operator import itemgetter

import numpy as np

from .automl_algorithm import AutoMLAlgorithm, AutoMLAlgorithmException

from evalml.model_family import ModelFamily
from evalml.pipelines.components.utils import handle_component_class
from evalml.pipelines.utils import _make_stacked_ensemble_pipeline


class IterativeAlgorithm(AutoMLAlgorithm):
    """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance."""

    def __init__(self,
                 allowed_pipelines=None,
                 max_iterations=None,
                 tuner_class=None,
                 text_columns=None,
                 random_state=0,
                 pipelines_per_batch=5,
                 n_jobs=-1,  # TODO remove
                 number_features=None,  # TODO remove
                 ensembling=False,
                 pipeline_params=None):
        """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance.

        Arguments:
            allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search. The default of None indicates all pipelines for this problem type are allowed.
            max_iterations (int): The maximum number of iterations to be evaluated.
            tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.
            pipelines_per_batch (int): the number of pipelines to be evaluated in each batch, after the first batch.
            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            number_features (int): The number of columns in the input features.
            ensembling (boolean): If True, runs ensembling in a separate batch after every allowed pipeline class has been iterated over. Defaults to False.
            pipeline_params (dict or None): Pipeline-level parameters that should be passed to the proposed pipelines.
        """
        super().__init__(allowed_pipelines=allowed_pipelines,
                         max_iterations=max_iterations,
                         tuner_class=tuner_class,
                         random_state=random_state)
        self.pipelines_per_batch = pipelines_per_batch
        self.n_jobs = n_jobs
        self.number_features = number_features
        self._text_columns = text_columns
        self._first_batch_results = []
        self._best_pipeline_info = {}
        self.ensembling = ensembling and len(self.allowed_pipelines) > 1
        self._pipeline_params = pipeline_params or {}

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
        if self._batch_number == 0:
            next_batch = [pipeline_class(parameters=self._transform_parameters(pipeline_class, {}), random_state=self.random_state)
                          for pipeline_class in self.allowed_pipelines]

        # One after training all pipelines one round
        elif (self.ensembling and
              self._batch_number != 1 and
              (self._batch_number) % (len(self._first_batch_results) + 1) == 0):
            input_pipelines = []
            for pipeline_dict in self._best_pipeline_info.values():
                pipeline_class = pipeline_dict['pipeline_class']
                pipeline_params = pipeline_dict['parameters']
                input_pipelines.append(pipeline_class(parameters=self._transform_parameters(pipeline_class, pipeline_params),
                                                      random_state=self.random_state))
            ensemble = _make_stacked_ensemble_pipeline(input_pipelines, input_pipelines[0].problem_type,
                                                       random_state=self.random_state)
            next_batch.append(ensemble)
        else:
            num_pipeline_classes = (len(self._first_batch_results) + 1) if self.ensembling else len(self._first_batch_results)
            idx = (self._batch_number - 1) % num_pipeline_classes
            pipeline_class = self._first_batch_results[idx][1]
            for i in range(self.pipelines_per_batch):
                proposed_parameters = self._tuners[pipeline_class.name].propose()
                pl_parameters = self._transform_parameters(pipeline_class, proposed_parameters)
                next_batch.append(pipeline_class(parameters=pl_parameters, random_state=self.random_state))
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
            if self.batch_number == 1:
                try:
                    super().add_result(score_to_minimize, pipeline)
                except ValueError as e:
                    if 'is not within the bounds of the space' in str(e):
                        raise ValueError("Default parameters for components in pipeline {} not in the hyperparameter ranges: {}".format(pipeline.name, e))
                    else:
                        raise(e)
            else:
                super().add_result(score_to_minimize, pipeline)
        if self.batch_number == 1:
            self._first_batch_results.append((score_to_minimize, pipeline.__class__))

        current_best_score = self._best_pipeline_info.get(pipeline.model_family, {}).get('score', np.inf)
        if score_to_minimize is not None and score_to_minimize < current_best_score:
            self._best_pipeline_info.update({pipeline.model_family: {'score': score_to_minimize,
                                                                     'pipeline_class': pipeline.__class__,
                                                                     'parameters': pipeline.parameters}
                                             })

    def _transform_parameters(self, pipeline_class, proposed_parameters):
        """Given a pipeline parameters dict, make sure n_jobs and number_features are set."""
        parameters = {}
        if self._pipeline_params:
            parameters['pipeline'] = self._pipeline_params
        component_graph = [handle_component_class(c) for c in pipeline_class.linearized_component_graph]
        for component_class in component_graph:
            component_parameters = proposed_parameters.get(component_class.name, {})
            init_params = inspect.signature(component_class.__init__).parameters

            # Add the text columns parameter if the component is a TextFeaturizer
            if component_class.name == "Text Featurization Component":
                component_parameters['text_columns'] = self._text_columns

            # Inspects each component and adds the following parameters when needed
            if 'n_jobs' in init_params:
                component_parameters['n_jobs'] = self.n_jobs
            if 'number_features' in init_params:
                component_parameters['number_features'] = self.number_features
            # Pass the pipeline params to the components that need them
            for param_name, value in self._pipeline_params.items():
                if param_name in init_params:
                    component_parameters[param_name] = value
            parameters[component_class.name] = component_parameters
        return parameters
