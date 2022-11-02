"""An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance."""
import logging
import warnings
from operator import itemgetter

import numpy as np

from evalml.automl.automl_algorithm.automl_algorithm import (
    AutoMLAlgorithm,
    AutoMLAlgorithmException,
)
from evalml.automl.utils import get_pipelines_from_component_graphs
from evalml.exceptions import ParameterNotUsedWarning
from evalml.model_family import ModelFamily
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.utils import infer_feature_types
from evalml.utils.logger import get_logger

_ESTIMATOR_FAMILY_ORDER = [
    ModelFamily.ARIMA,
    ModelFamily.PROPHET,
    ModelFamily.EXPONENTIAL_SMOOTHING,
    ModelFamily.LINEAR_MODEL,
    ModelFamily.XGBOOST,
    ModelFamily.LIGHTGBM,
    ModelFamily.CATBOOST,
    ModelFamily.RANDOM_FOREST,
    ModelFamily.DECISION_TREE,
    ModelFamily.EXTRA_TREES,
]


class IterativeAlgorithm(AutoMLAlgorithm):
    """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance.

    Args:
        X (pd.DataFrame): Training data.
        y (pd.Series): Target data.
        problem_type (ProblemType): Problem type associated with training data.
        sampler_name (BaseSampler): Sampler to use for preprocessing. Defaults to None.
        allowed_model_families (list(str, ModelFamily)): The model families to search. The default of None searches over all
            model families. Run evalml.pipelines.components.utils.allowed_model_families("binary") to see options. Change `binary`
            to `multiclass` or `regression` depending on the problem type. Note that if allowed_pipelines is provided,
            this parameter will be ignored.
        allowed_component_graphs (dict): A dictionary of lists or ComponentGraphs indicating the component graphs allowed in the search.
            The format should follow { "Name_0": [list_of_components], "Name_1": [ComponentGraph(...)] }

            The default of None indicates all pipeline component graphs for this problem type are allowed. Setting this field will cause
            allowed_model_families to be ignored.

            e.g. allowed_component_graphs = { "My_Graph": ["Imputer", "One Hot Encoder", "Random Forest Classifier"] }
        max_batches (int): The maximum number of batches to be evaluated. Used to determine ensembling. Defaults to None.
        max_iterations (int): The maximum number of iterations to be evaluated. Used to determine ensembling. Defaults to None.
        tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        pipelines_per_batch (int): The number of pipelines to be evaluated in each batch, after the first batch. Defaults to 5.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to None.
        number_features (int): The number of columns in the input features. Defaults to None.
        ensembling (boolean): If True, runs ensembling in a separate batch after every allowed pipeline class has been iterated over. Defaults to False.
        text_in_ensembling (boolean): If True and ensembling is True, then n_jobs will be set to 1 to avoid downstream sklearn stacking issues related to nltk. Defaults to False.
        search_parameters (dict or None): Pipeline-level parameters and custom hyperparameter ranges specified for pipelines to iterate over. Hyperparameter ranges
            must be passed in as skopt.space objects. Defaults to None.
        _estimator_family_order (list(ModelFamily) or None): specify the sort order for the first batch. Defaults to None, which uses _ESTIMATOR_FAMILY_ORDER.
        allow_long_running_models (bool): Whether or not to allow longer-running models for large multiclass problems. If False and no pipelines, component graphs, or model families are provided,
            AutoMLSearch will not use Elastic Net or XGBoost when there are more than 75 multiclass targets and will not use CatBoost when there are more than 150 multiclass targets. Defaults to False.
        features (list)[FeatureBase]: List of features to run DFS on in AutoML pipelines. Defaults to None. Features will only be computed if the columns used by the feature exist in the input and if the feature itself is not in input.
        verbose (boolean): Whether or not to display logging information regarding pipeline building. Defaults to False.
        exclude_featurizers (list[str]): A list of featurizer components to exclude from the pipelines built by IterativeAlgorithm.
            Valid options are "DatetimeFeaturizer", "EmailFeaturizer", "URLFeaturizer", "NaturalLanguageFeaturizer", "TimeSeriesFeaturizer"
    """

    def __init__(
        self,
        X,
        y,
        problem_type,
        sampler_name=None,
        allowed_model_families=None,
        allowed_component_graphs=None,
        max_batches=None,
        max_iterations=None,
        tuner_class=None,
        random_seed=0,
        pipelines_per_batch=5,
        n_jobs=-1,  # TODO remove
        number_features=None,  # TODO remove
        ensembling=False,
        text_in_ensembling=False,
        search_parameters=None,
        _estimator_family_order=None,
        allow_long_running_models=False,
        features=None,
        verbose=False,
        exclude_featurizers=None,
    ):
        self.X = infer_feature_types(X)
        self.y = infer_feature_types(y)
        self.problem_type = problem_type
        self.random_seed = random_seed
        self.sampler_name = sampler_name
        self.allowed_model_families = allowed_model_families
        self.pipelines_per_batch = pipelines_per_batch
        self.n_jobs = n_jobs
        self.number_features = number_features
        self._first_batch_results = []
        self._best_pipeline_info = {}
        self.ensembling = ensembling
        self.search_parameters = search_parameters or {}
        self.text_in_ensembling = text_in_ensembling
        self.max_batches = max_batches
        self.max_iterations = max_iterations
        self.allow_long_running_models = allow_long_running_models
        if verbose:
            self.logger = get_logger(f"{__name__}.verbose")
        else:
            self.logger = logging.getLogger(__name__)

        self._estimator_family_order = (
            _estimator_family_order or _ESTIMATOR_FAMILY_ORDER
        )
        if search_parameters and not isinstance(search_parameters, dict):
            raise ValueError(
                f"If search_parameters provided, must be of type dict. Received {type(search_parameters)}",
            )
        self.allowed_pipelines = []

        self.features = features
        self.allowed_component_graphs = allowed_component_graphs
        self._set_additional_pipeline_params()
        self.exclude_featurizers = exclude_featurizers

        super().__init__(
            allowed_pipelines=self.allowed_pipelines,
            search_parameters=self.search_parameters,
            tuner_class=tuner_class,
            text_in_ensembling=self.text_in_ensembling,
            random_seed=random_seed,
            n_jobs=self.n_jobs,
        )
        self._separate_hyperparameters_from_parameters()
        self._create_pipelines()
        self._set_allowed_pipelines(self.allowed_pipelines)

    def _create_pipelines(self):
        indices = []
        pipelines_to_sort = []
        pipelines_end = []
        self.allowed_pipelines = []
        if self.allowed_component_graphs is None:
            self.logger.info("Generating pipelines to search over...")
            allowed_estimators = get_estimators(
                self.problem_type,
                self.allowed_model_families,
            )
            allowed_estimators = self._filter_estimators(
                allowed_estimators,
                self.problem_type,
                self.allow_long_running_models,
                self.allowed_model_families,
                self.y.nunique(),
                self.logger,
            )
            self.logger.debug(
                f"allowed_estimators set to {[estimator.name for estimator in allowed_estimators]}",
            )
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always", category=ParameterNotUsedWarning)
                self.allowed_pipelines = [
                    make_pipeline(
                        self.X,
                        self.y,
                        estimator,
                        self.problem_type,
                        parameters=self._pipeline_parameters,
                        sampler_name=self.sampler_name,
                        known_in_advance=self._pipeline_parameters.get(
                            "pipeline",
                            {},
                        ).get("known_in_advance", None),
                        features=self.features,
                        exclude_featurizers=self.exclude_featurizers,
                    )
                    for estimator in allowed_estimators
                ]
                if (
                    len(self.allowed_pipelines)
                    and "STL Decomposer"
                    in self.allowed_pipelines[-1].component_graph.compute_order
                ):
                    without_pipelines = [
                        make_pipeline(
                            self.X,
                            self.y,
                            estimator,
                            self.problem_type,
                            parameters=self._pipeline_parameters,
                            sampler_name=self.sampler_name,
                            known_in_advance=self._pipeline_parameters.get(
                                "pipeline",
                                {},
                            ).get("known_in_advance", None),
                            features=self.features,
                            exclude_featurizers=self.exclude_featurizers,
                            include_decomposer=False,
                        )
                        for estimator in allowed_estimators
                    ]
                    self.allowed_pipelines = self.allowed_pipelines + without_pipelines
            self._catch_warnings(w)
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always", category=ParameterNotUsedWarning)
                self.allowed_pipelines = get_pipelines_from_component_graphs(
                    self.allowed_component_graphs,
                    self.problem_type,
                    self._pipeline_parameters,
                    self.random_seed,
                )
            self._catch_warnings(w)
        if self.allowed_pipelines == []:
            raise ValueError("No allowed pipelines to search")

        if self.ensembling and len(self.allowed_pipelines) == 1:
            self.logger.warning(
                "Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run.",
            )
            self.ensembling = False

        if self.ensembling and self.max_iterations is not None:
            # Baseline + first batch + each pipeline iteration + 1
            first_ensembling_iteration = (
                1
                + len(self.allowed_pipelines)
                + len(self.allowed_pipelines) * self.pipelines_per_batch
                + 1
            )
            if self.max_iterations < first_ensembling_iteration:
                self.ensembling = False
                self.logger.warning(
                    f"Ensembling is set to True, but max_iterations is too small, so ensembling will not run. Set max_iterations >= {first_ensembling_iteration} to run ensembling.",
                )
            else:
                self.logger.info(
                    f"Ensembling will run at the {first_ensembling_iteration} iteration and every {len(self.allowed_pipelines) * self.pipelines_per_batch} iterations after that.",
                )

        if self.max_batches and self.max_iterations is None:
            self.show_batch_output = True
            if self.ensembling:
                ensemble_nth_batch = len(self.allowed_pipelines) + 1
                num_ensemble_batches = (self.max_batches - 1) // ensemble_nth_batch
                if num_ensemble_batches == 0:
                    self.ensembling = False
                    self.logger.warning(
                        f"Ensembling is set to True, but max_batches is too small, so ensembling will not run. Set max_batches >= {ensemble_nth_batch + 1} to run ensembling.",
                    )
                else:
                    self.logger.info(
                        f"Ensembling will run every {ensemble_nth_batch} batches.",
                    )

                self.max_iterations = (
                    1
                    + len(self.allowed_pipelines)
                    + self.pipelines_per_batch
                    * (self.max_batches - 1 - num_ensemble_batches)
                    + num_ensemble_batches
                )
            else:
                self.max_iterations = (
                    1
                    + len(self.allowed_pipelines)
                    + (self.pipelines_per_batch * (self.max_batches - 1))
                )

        for pipeline in self.allowed_pipelines or []:
            if pipeline.model_family in self._estimator_family_order:
                indices.append(
                    self._estimator_family_order.index(pipeline.model_family),
                )
                pipelines_to_sort.append(pipeline)
            else:
                pipelines_end.append(pipeline)
        pipelines_start = [
            pipeline
            for _, pipeline in (
                sorted(zip(indices, pipelines_to_sort), key=lambda pair: pair[0]) or []
            )
        ]
        self.allowed_pipelines = pipelines_start + pipelines_end

        self.logger.debug(
            f"allowed_pipelines set to {[pipeline.name for pipeline in self.allowed_pipelines]}",
        )
        self.logger.debug(
            f"allowed_model_families set to {self.allowed_model_families}",
        )
        self.logger.info(f"{len(self.allowed_pipelines)} pipelines ready for search.")

    def next_batch(self):
        """Get the next batch of pipelines to evaluate.

        Returns:
            list[PipelineBase]: A list of instances of PipelineBase subclasses, ready to be trained and evaluated.

        Raises:
            AutoMLAlgorithmException: If no results were reported from the first batch.
        """
        if self._batch_number == 1:
            if len(self._first_batch_results) == 0:
                raise AutoMLAlgorithmException(
                    "No results were reported from the first batch",
                )
            self._first_batch_results = sorted(
                self._first_batch_results,
                key=itemgetter(0),
            )

        next_batch = []
        if self._batch_number == 0:
            for pipeline in self.allowed_pipelines:
                starting_parameters = self._tuners[
                    pipeline.name
                ].get_starting_parameters(self._hyperparameters, self.random_seed)
                parameters = self._transform_parameters(pipeline, starting_parameters)
                next_batch.append(
                    pipeline.new(parameters=parameters, random_seed=self.random_seed),
                )

        # One after training all pipelines one round
        elif (
            self.ensembling
            and self._batch_number != 1
            and (self._batch_number) % (len(self._first_batch_results) + 1) == 0
        ):
            next_batch = self._create_ensemble(
                self._pipeline_parameters.get("Label Encoder", {}),
            )
        else:
            num_pipelines = (
                (len(self._first_batch_results) + 1)
                if self.ensembling
                else len(self._first_batch_results)
            )
            idx = (self._batch_number - 1) % num_pipelines
            pipeline = self._first_batch_results[idx][1]
            for i in range(self.pipelines_per_batch):
                proposed_parameters = self._tuners[pipeline.name].propose()
                parameters = self._transform_parameters(pipeline, proposed_parameters)
                next_batch.append(
                    pipeline.new(parameters=parameters, random_seed=self.random_seed),
                )
        self._pipeline_number += len(next_batch)
        self._batch_number += 1
        return next_batch

    def num_pipelines_per_batch(self, batch_number):
        """Return the number of pipelines in the nth batch.

        Args:
            batch_number (int): which batch to calculate the number of pipelines for.

        Returns:
            int: number of pipelines in the given batch.
        """
        if batch_number == 0:
            return len(self.allowed_pipelines)
        return self.pipelines_per_batch

    def add_result(
        self,
        score_to_minimize,
        pipeline,
        trained_pipeline_results,
        cached_data=None,
    ):
        """Register results from evaluating a pipeline.

        Args:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
            trained_pipeline_results (dict): Results from training a pipeline.
            cached_data (dict): A dictionary of cached data, where the keys are the model family. Expected to be of format
                {model_family: {hash1: trained_component_graph, hash2: trained_component_graph...}...}.
                Defaults to None.

        Raises:
            ValueError: If default parameters are not in the acceptable hyperparameter ranges.
        """
        cached_data = cached_data or {}
        if pipeline.model_family != ModelFamily.ENSEMBLE:
            if self.batch_number == 1:
                try:
                    super().add_result(
                        score_to_minimize,
                        pipeline,
                        trained_pipeline_results,
                    )
                except ValueError as e:
                    if "is not within the bounds of the space" in str(e):
                        raise ValueError(
                            "Default parameters for components in pipeline {} not in the hyperparameter ranges: {}".format(
                                pipeline.name,
                                e,
                            ),
                        )
                    else:
                        raise (e)
            else:
                super().add_result(
                    score_to_minimize,
                    pipeline,
                    trained_pipeline_results,
                )
        if self.batch_number == 1:
            self._first_batch_results.append((score_to_minimize, pipeline))
        current_best_score = self._best_pipeline_info.get(
            pipeline.model_family,
            {},
        ).get("mean_cv_score", np.inf)
        if (
            score_to_minimize is not None
            and score_to_minimize < current_best_score
            and pipeline.model_family != ModelFamily.ENSEMBLE
        ):
            self._best_pipeline_info.update(
                {
                    pipeline.model_family: {
                        "mean_cv_score": score_to_minimize,
                        "pipeline": pipeline,
                        "parameters": pipeline.parameters,
                        "id": trained_pipeline_results["id"],
                        "cached_data": cached_data,
                    },
                },
            )

    def _catch_warnings(self, warning_list):
        parameter_not_used_warnings = []
        raised_messages = []
        for msg in warning_list:
            if isinstance(msg.message, ParameterNotUsedWarning):
                parameter_not_used_warnings.append(msg.message)
            # Raise non-PNU warnings immediately, but only once per warning
            elif str(msg.message) not in raised_messages:
                warnings.warn(msg.message)
                raised_messages.append(str(msg.message))

        # Raise PNU warnings, iff the warning was raised in every pipeline
        if len(parameter_not_used_warnings) == len(self.allowed_pipelines) and len(
            parameter_not_used_warnings,
        ):
            final_message = set([])
            for msg in parameter_not_used_warnings:
                if len(final_message) == 0:
                    final_message = final_message.union(msg.components)
                else:
                    final_message = final_message.intersection(msg.components)

            warnings.warn(ParameterNotUsedWarning(final_message))
