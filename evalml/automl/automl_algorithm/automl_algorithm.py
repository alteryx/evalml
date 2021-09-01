"""Base class for the AutoML algorithms which power EvalML."""
from abc import ABC, abstractmethod

from evalml.exceptions import PipelineNotFoundError
from evalml.pipelines.components.utils import handle_component_class
from evalml.pipelines.utils import _make_stacked_ensemble_pipeline
from evalml.tuners import SKOptTuner


class AutoMLAlgorithmException(Exception):
    """Exception raised when an error is encountered during the computation of the automl algorithm."""

    pass


class AutoMLAlgorithm(ABC):
    """Base class for the AutoML algorithms which power EvalML.

    This class represents an automated machine learning (AutoML) algorithm. It encapsulates the decision-making logic behind an automl search, by both deciding which pipelines to evaluate next and by deciding what set of parameters to configure the pipeline with.

    To use this interface, you must define a next_batch method which returns the next group of pipelines to evaluate on the training data. That method may access state and results recorded from the previous batches, although that information is not tracked in a general way in this base class. Overriding add_result is a convenient way to record pipeline evaluation info if necessary.

    Args:
        allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search. The default of None indicates all pipelines for this problem type are allowed.
        custom_hyperparameters (dict): Custom hyperparameter ranges specified for pipelines to iterate over.
        max_iterations (int): The maximum number of iterations to be evaluated.
        tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def __init__(
        self,
        allowed_pipelines=None,
        custom_hyperparameters=None,
        max_iterations=None,
        tuner_class=None,
        random_seed=0,
    ):
        self.random_seed = random_seed
        self.allowed_pipelines = allowed_pipelines or []
        self.max_iterations = max_iterations
        self._tuner_class = tuner_class or SKOptTuner
        self._tuners = {}
        self._best_pipeline_info = {}
        self.text_in_ensembling = False
        self.n_jobs = -1
        for pipeline in self.allowed_pipelines:
            pipeline_hyperparameters = pipeline.get_hyperparameter_ranges(
                custom_hyperparameters
            )
            self._tuners[pipeline.name] = self._tuner_class(
                pipeline_hyperparameters, random_seed=self.random_seed
            )
        self._pipeline_number = 0
        self._batch_number = 0

    @abstractmethod
    def next_batch(self):
        """Get the next batch of pipelines to evaluate.

        Returns:
            list[PipelineBase]: A list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """

    def add_result(self, score_to_minimize, pipeline, trained_pipeline_results):
        """Register results from evaluating a pipeline.

        Args:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
            trained_pipeline_results (dict): Results from training a pipeline.

        Raises:
            PipelineNotFoundError: If pipeline is not allowed in search.
        """
        if pipeline.name not in self._tuners:
            raise PipelineNotFoundError(
                f"No such pipeline allowed in this AutoML search: {pipeline.name}"
            )
        self._tuners[pipeline.name].add(pipeline.parameters, score_to_minimize)

    @property
    def pipeline_number(self):
        """Returns the number of pipelines which have been recommended so far."""
        return self._pipeline_number

    @property
    def batch_number(self):
        """Returns the number of batches which have been recommended so far."""
        return self._batch_number

    def _create_ensemble(self):
        next_batch = []

        # Custom Stacked Pipelines
        ensembler_component_graph = {}
        final_components = []
        problem_type = None
        n_jobs_ensemble = 1 if self.text_in_ensembling else self.n_jobs

        for model_type, best_info in self._best_pipeline_info.items():

            def _make_new_component_name(component_name):
                return str(model_type) + " Pipeline - " + component_name

            pipeline = best_info["pipeline"]
            if problem_type is None:
                problem_type = pipeline.problem_type
            final_component = None
            ensemble_y = "y"
            for name, component_list in pipeline.component_graph.component_dict.items():
                new_component_list = []
                new_component_name = _make_new_component_name(name)
                for i, item in enumerate(component_list):
                    if i == 0:
                        fitted_comp = handle_component_class(item)
                        new_component_list.append(fitted_comp)
                    elif isinstance(item, str) and item not in ["X", "y"]:
                        new_component_list.append(_make_new_component_name(item))
                    else:
                        new_component_list.append(item)
                    if i != 0 and item.endswith(".y"):
                        ensemble_y = _make_new_component_name(item)
                ensembler_component_graph[new_component_name] = new_component_list
                final_component = new_component_name
            final_components.append(final_component)

        ensemble = _make_stacked_ensemble_pipeline(
            problem_type,
            component_graph=ensembler_component_graph,
            final_components=final_components,
            random_seed=self.random_seed,
            n_jobs=n_jobs_ensemble,
            ensemble_y=ensemble_y,
        )
        next_batch.append(ensemble)

        # Sklearn Stacked Pipelines
        input_pipelines = []
        for pipeline_dict in self._best_pipeline_info.values():
            pipeline = pipeline_dict["pipeline"]
            pipeline_params = pipeline_dict["parameters"]
            if hasattr(self, "_transform_parameters"):
                parameters = self._transform_parameters(pipeline, pipeline_params)
            if hasattr(self, "_selected_cols"):
                if (
                    "Select Columns Transformer"
                    in pipeline.component_graph.component_instances
                ):
                    parameters.update(
                        {"Select Columns Transformer": {"columns": self._selected_cols}}
                    )
            input_pipelines.append(
                pipeline.new(parameters=parameters, random_seed=self.random_seed)
            )
        ensemble = _make_stacked_ensemble_pipeline(
            problem_type,
            input_pipelines=input_pipelines,
            random_seed=self.random_seed,
            n_jobs=n_jobs_ensemble,
            use_sklearn=True,
        )
        next_batch.append(ensemble)
        return next_batch
