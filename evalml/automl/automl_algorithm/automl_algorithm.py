"""Base class for the AutoML algorithms which power EvalML."""
from abc import ABC, abstractmethod

from evalml.exceptions import PipelineNotFoundError
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
        tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
        text_in_ensembling (boolean): If True and ensembling is True, then n_jobs will be set to 1 to avoid downstream sklearn stacking issues related to nltk. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def __init__(
        self,
        allowed_pipelines=None,
        custom_hyperparameters=None,
        tuner_class=None,
        text_in_ensembling=False,
        random_seed=0,
        n_jobs=-1,
    ):
        self.random_seed = random_seed
        self.allowed_pipelines = allowed_pipelines or []
        self._tuner_class = tuner_class or SKOptTuner
        self._tuners = {}
        self._best_pipeline_info = {}
        self.text_in_ensembling = text_in_ensembling
        self.n_jobs = n_jobs
        self._selected_cols = None
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

    @abstractmethod
    def _transform_parameters(self, pipeline, proposed_parameters):
        """Given a pipeline parameters dict, make sure pipeline_params, custom_hyperparameters, n_jobs are set properly.

        Arguments:
            pipeline (PipelineBase): The pipeline object to update the parameters.
            proposed_parameters (dict): Parameters to use when updating the pipeline.
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
        best_pipelines = list(self._best_pipeline_info.values())
        problem_type = best_pipelines[0]["pipeline"].problem_type
        n_jobs_ensemble = 1 if self.text_in_ensembling else self.n_jobs
        input_pipelines = []
        for pipeline_dict in best_pipelines:
            pipeline = pipeline_dict["pipeline"]
            pipeline_params = self._transform_parameters(
                pipeline, pipeline_dict["parameters"]
            )
            if (
                "Numeric Pipeline - Select Columns Transformer"
                in pipeline.component_graph.component_instances
            ):
                pipeline_params.update(self._create_split_select_parameters())
            elif (
                "Select Columns Transformer"
                in pipeline.component_graph.component_instances
            ):
                pipeline_params.update(
                    {"Select Columns Transformer": {"columns": self._selected_cols}}
                )
            input_pipelines.append(
                pipeline.new(parameters=pipeline_params, random_seed=self.random_seed)
            )

        ensemble = _make_stacked_ensemble_pipeline(
            input_pipelines,
            problem_type,
            random_seed=self.random_seed,
            n_jobs=n_jobs_ensemble,
        )
        next_batch.append(ensemble)
        return next_batch

    def _set_additional_pipeline_params(self):
        drop_columns = (
            self._pipeline_params["Drop Columns Transformer"]["columns"]
            if "Drop Columns Transformer" in self._pipeline_params
            else None
        )
        index_and_unknown_columns = list(
            self.X.ww.select(["index", "unknown"], return_schema=True).columns
        )
        unknown_columns = list(self.X.ww.select("unknown", return_schema=True).columns)
        if len(index_and_unknown_columns) > 0 and drop_columns is None:
            self._pipeline_params["Drop Columns Transformer"] = {
                "columns": index_and_unknown_columns
            }
            if len(unknown_columns):
                self.logger.info(
                    f"Removing columns {unknown_columns} because they are of 'Unknown' type"
                )
