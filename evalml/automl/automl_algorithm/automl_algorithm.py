"""Base class for the AutoML algorithms which power EvalML."""
import inspect
from abc import ABC, abstractmethod

from skopt.space import Categorical, Integer, Real

from evalml.exceptions import PipelineNotFoundError
from evalml.pipelines.utils import _make_stacked_ensemble_pipeline
from evalml.problem_types import is_multiclass
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
        search_parameters (dict): Search parameter ranges specified for pipelines to iterate over.
        tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
        text_in_ensembling (boolean): If True and ensembling is True, then n_jobs will be set to 1 to avoid downstream sklearn stacking issues related to nltk. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def __init__(
        self,
        allowed_pipelines=None,
        search_parameters=None,
        tuner_class=None,
        text_in_ensembling=False,
        random_seed=0,
        n_jobs=-1,
    ):
        self.random_seed = random_seed
        self._tuner_class = tuner_class or SKOptTuner
        self._tuners = {}
        self._best_pipeline_info = {}
        self.text_in_ensembling = text_in_ensembling
        self.n_jobs = n_jobs
        self._selected_cols = None
        self.search_parameters = search_parameters or {}
        self._hyperparameters = {}
        self._pipeline_parameters = {}
        self.allowed_pipelines = []
        if allowed_pipelines is not None:
            self._set_allowed_pipelines(allowed_pipelines)
        self._pipeline_number = 0
        self._batch_number = 0
        self._default_max_batches = 1

    @abstractmethod
    def next_batch(self):
        """Get the next batch of pipelines to evaluate.

        Returns:
            list[PipelineBase]: A list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """

    @abstractmethod
    def num_pipelines_per_batch(self, batch_number):
        """Return the number of pipelines in the nth batch.

        Args:
            batch_number (int): which batch to calculate the number of pipelines for.

        Returns:
            int: number of pipelines in the given batch.
        """

    def _set_allowed_pipelines(self, allowed_pipelines):
        """Sets the allowed parameters and creates the tuners for the input pipelines."""
        self.allowed_pipelines = allowed_pipelines
        for pipeline in self.allowed_pipelines:
            self._create_tuner(pipeline)

    def _create_tuner(self, pipeline):
        """Creates a tuner given the input pipeline."""
        pipeline_hyperparameters = pipeline.get_hyperparameter_ranges(
            self._hyperparameters,
        )
        self._tuners[pipeline.name] = self._tuner_class(
            pipeline_hyperparameters,
            random_seed=self.random_seed,
        )

    def _separate_hyperparameters_from_parameters(self):
        """Seperate out the parameter and hyperparameter values from the search parameters dict."""
        for key, value in self.search_parameters.items():
            hyperparam = {}
            param = {}
            for name, parameters in value.items():
                if isinstance(parameters, (Integer, Categorical, Real)):
                    hyperparam[name] = parameters
                else:
                    param[name] = parameters
            if hyperparam:
                self._hyperparameters[key] = hyperparam
            if param:
                self._pipeline_parameters[key] = param

    def _transform_parameters(self, pipeline, proposed_parameters):
        """Given a pipeline parameters dict, make sure pipeline_parameters, custom_hyperparameters, n_jobs are set properly.

        Arguments:
            pipeline (PipelineBase): The pipeline object to update the parameters.
            proposed_parameters (dict): Parameters to use when updating the pipeline.
        """
        parameters = {}
        if "pipeline" in self._pipeline_parameters:
            parameters["pipeline"] = self._pipeline_parameters["pipeline"]
        for (
            name,
            component_instance,
        ) in pipeline.component_graph.component_instances.items():
            component_class = type(component_instance)
            component_parameters = proposed_parameters.get(name, {})
            init_params = inspect.signature(component_class.__init__).parameters
            # Only overwrite the parameters that were passed in on pipeline parameters
            # if they don't exist in the propsed parameters
            if name in self._pipeline_parameters and name not in component_parameters:
                for param_name, value in self._pipeline_parameters[name].items():
                    component_parameters[param_name] = value
            # Inspects each component and adds the following parameters when needed
            if "n_jobs" in init_params:
                component_parameters["n_jobs"] = self.n_jobs
            if "number_features" in init_params and hasattr(self, "number_features"):
                component_parameters["number_features"] = self.number_features
            if "pipeline" in self.search_parameters:
                for param_name, value in self.search_parameters["pipeline"].items():
                    if param_name in init_params:
                        component_parameters[param_name] = value
            parameters[name] = component_parameters
        return parameters

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
                f"No such pipeline allowed in this AutoML search: {pipeline.name}",
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

    @property
    def default_max_batches(self):
        """Returns the number of max batches AutoMLSearch should run by default."""
        return 1

    def _create_ensemble(self, label_encoder_params=None):
        next_batch = []
        best_pipelines = list(self._best_pipeline_info.values())
        problem_type = best_pipelines[0]["pipeline"].problem_type
        n_jobs_ensemble = 1 if self.text_in_ensembling else self.n_jobs
        input_pipelines = []
        cached_data = {
            model_family: x["cached_data"]
            for model_family, x in self._best_pipeline_info.items()
        }
        for pipeline_dict in best_pipelines:
            pipeline = pipeline_dict["pipeline"]
            input_pipelines.append(pipeline)

        if label_encoder_params is not None:
            label_encoder_params = {"Label Encoder": label_encoder_params}
        else:
            label_encoder_params = {}

        ensemble = _make_stacked_ensemble_pipeline(
            input_pipelines,
            problem_type,
            random_seed=self.random_seed,
            n_jobs=n_jobs_ensemble,
            cached_data=cached_data,
            label_encoder_params=label_encoder_params,
        )
        next_batch.append(ensemble)
        return next_batch

    def _set_additional_pipeline_params(self):
        drop_columns = (
            self.search_parameters["Drop Columns Transformer"]["columns"]
            if "Drop Columns Transformer" in self.search_parameters
            else None
        )
        index_and_unknown_columns = list(
            self.X.ww.select(["index", "unknown"], return_schema=True).columns,
        )
        unknown_columns = list(self.X.ww.select("unknown", return_schema=True).columns)
        if len(index_and_unknown_columns) > 0 and drop_columns is None:
            self.search_parameters["Drop Columns Transformer"] = {
                "columns": index_and_unknown_columns,
            }
            if len(unknown_columns):
                self.logger.info(
                    f"Removing columns {unknown_columns} because they are of 'Unknown' type",
                )
        kina_columns = self.search_parameters.get("pipeline", {}).get(
            "known_in_advance",
            [],
        )
        if kina_columns:
            no_kin_columns = [c for c in self.X.columns if c not in kina_columns]
            kin_name = "Known In Advance Pipeline - Select Columns Transformer"
            no_kin_name = "Not Known In Advance Pipeline - Select Columns Transformer"
            self.search_parameters[kin_name] = {"columns": kina_columns}
            self.search_parameters[no_kin_name] = {"columns": no_kin_columns}

    def _filter_estimators(
        self,
        estimators,
        problem_type,
        allow_long_running_models,
        allowed_model_families,
        y_unique,
        logger,
    ):
        """Function to remove computationally expensive and long-running estimators from datasets with large numbers of unique classes. Thresholds were determined empirically."""
        estimators_to_drop = []
        if (
            not is_multiclass(problem_type)
            or allow_long_running_models
            or allowed_model_families is not None
        ):
            return estimators
        if y_unique > 75:
            estimators_to_drop.extend(["Elastic Net Classifier", "XGBoost Classifier"])
        if y_unique > 150:
            estimators_to_drop.append("CatBoost Classifier")

        dropped_estimators = [e for e in estimators if e.name in estimators_to_drop]
        if len(dropped_estimators):
            logger.info(
                "Dropping estimators {} because the number of unique targets is {} and `allow_long_running_models` is set to {}".format(
                    ", ".join(sorted([e.name for e in dropped_estimators])),
                    y_unique,
                    allow_long_running_models,
                ),
            )
        estimators = [e for e in estimators if e not in dropped_estimators]
        return estimators
