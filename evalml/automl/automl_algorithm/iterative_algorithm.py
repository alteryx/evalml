"""An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance."""
import inspect
from operator import itemgetter

import numpy as np
from skopt.space import Categorical, Integer, Real

from .automl_algorithm import AutoMLAlgorithm, AutoMLAlgorithmException

from evalml.model_family import ModelFamily

_ESTIMATOR_FAMILY_ORDER = [
    ModelFamily.LINEAR_MODEL,
    ModelFamily.DECISION_TREE,
    ModelFamily.EXTRA_TREES,
    ModelFamily.RANDOM_FOREST,
    ModelFamily.XGBOOST,
    ModelFamily.LIGHTGBM,
    ModelFamily.CATBOOST,
    ModelFamily.ARIMA,
]


class IterativeAlgorithm(AutoMLAlgorithm):
    """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance.

    Args:
        allowed_pipelines (list(class)): A list of PipelineBase instances indicating the pipelines allowed in the search. The default of None indicates all pipelines for this problem type are allowed.
        max_iterations (int): The maximum number of iterations to be evaluated.
        tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        pipelines_per_batch (int): The number of pipelines to be evaluated in each batch, after the first batch. Defaults to 5.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to None.
        number_features (int): The number of columns in the input features. Defaults to None.
        ensembling (boolean): If True, runs ensembling in a separate batch after every allowed pipeline class has been iterated over. Defaults to False.
        text_in_ensembling (boolean): If True and ensembling is True, then n_jobs will be set to 1 to avoid downstream sklearn stacking issues related to nltk. Defaults to None.
        pipeline_params (dict or None): Pipeline-level parameters that should be passed to the proposed pipelines. Defaults to None.
        custom_hyperparameters (dict or None): Custom hyperparameter ranges specified for pipelines to iterate over. Defaults to None.
        _estimator_family_order (list(ModelFamily) or None): specify the sort order for the first batch. Defaults to None, which uses _ESTIMATOR_FAMILY_ORDER.
    """

    def __init__(
        self,
        allowed_pipelines=None,
        max_iterations=None,
        tuner_class=None,
        random_seed=0,
        pipelines_per_batch=5,
        n_jobs=-1,  # TODO remove
        number_features=None,  # TODO remove
        ensembling=False,
        text_in_ensembling=False,
        pipeline_params=None,
        custom_hyperparameters=None,
        _estimator_family_order=None,
    ):
        """An automl algorithm which first fits a base round of pipelines with default parameters, then does a round of parameter tuning on each pipeline in order of performance.

        Args:
            allowed_pipelines (list(class)): A list of PipelineBase instances indicating the pipelines allowed in the search. The default of None indicates all pipelines for this problem type are allowed.
            max_iterations (int): The maximum number of iterations to be evaluated.
            tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
            random_seed (int): Seed for the random number generator. Defaults to 0.
            pipelines_per_batch (int): The number of pipelines to be evaluated in each batch, after the first batch. Defaults to 5.
            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
            number_features (int): The number of columns in the input features. Defaults to None.
            ensembling (boolean): If True, runs ensembling in a separate batch after every allowed pipeline class has been iterated over. Defaults to False.
            text_in_ensembling (boolean): If True and ensembling is True, then n_jobs will be set to 1 to avoid downstream sklearn stacking issues related to nltk. Defaults to None.
            pipeline_params (dict or None): Pipeline-level parameters that should be passed to the proposed pipelines. Defaults to None.
            custom_hyperparameters (dict or None): Custom hyperparameter ranges specified for pipelines to iterate over. Defaults to None.
            _estimator_family_order (list[ModelFamily]): specify the sort order for the first batch. Defaults to None, which uses _ESTIMATOR_FAMILY_ORDER.
        """
        self._estimator_family_order = (
            _estimator_family_order or _ESTIMATOR_FAMILY_ORDER
        )
        indices = []
        pipelines_to_sort = []
        pipelines_end = []
        for pipeline in allowed_pipelines or []:
            if pipeline.model_family in self._estimator_family_order:
                indices.append(
                    self._estimator_family_order.index(pipeline.model_family)
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
        allowed_pipelines = pipelines_start + pipelines_end

        super().__init__(
            allowed_pipelines=allowed_pipelines,
            custom_hyperparameters=custom_hyperparameters,
            max_iterations=max_iterations,
            tuner_class=tuner_class,
            random_seed=random_seed,
        )
        self.pipelines_per_batch = pipelines_per_batch
        self.n_jobs = n_jobs
        self.number_features = number_features
        self._first_batch_results = []
        self._best_pipeline_info = {}
        self.ensembling = ensembling and len(self.allowed_pipelines) > 1
        self.text_in_ensembling = text_in_ensembling
        self._pipeline_params = pipeline_params or {}
        self._custom_hyperparameters = custom_hyperparameters or {}

        if custom_hyperparameters and not isinstance(custom_hyperparameters, dict):
            raise ValueError(
                f"If custom_hyperparameters provided, must be of type dict. Received {type(custom_hyperparameters)}"
            )

        for param_name_val in self._pipeline_params.values():
            for _, param_val in param_name_val.items():
                if isinstance(param_val, (Integer, Real, Categorical)):
                    raise ValueError(
                        "Pipeline parameters should not contain skopt.Space variables, please pass them "
                        "to custom_hyperparameters instead!"
                    )
        for hyperparam_name_val in self._custom_hyperparameters.values():
            for _, hyperparam_val in hyperparam_name_val.items():
                if not isinstance(hyperparam_val, (Integer, Real, Categorical)):
                    raise ValueError(
                        "Custom hyperparameters should only contain skopt.Space variables such as Categorical, Integer,"
                        " and Real!"
                    )

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
                    "No results were reported from the first batch"
                )
            self._first_batch_results = sorted(
                self._first_batch_results, key=itemgetter(0)
            )

        next_batch = []
        if self._batch_number == 0:
            next_batch = [
                pipeline.new(
                    parameters=self._transform_parameters(pipeline, {}),
                    random_seed=self.random_seed,
                )
                for pipeline in self.allowed_pipelines
            ]

        # One after training all pipelines one round
        elif (
            self.ensembling
            and self._batch_number != 1
            and (self._batch_number) % (len(self._first_batch_results) + 1) == 0
        ):
            next_batch = self._create_ensemble()
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
                    pipeline.new(parameters=parameters, random_seed=self.random_seed)
                )
        self._pipeline_number += len(next_batch)
        self._batch_number += 1
        return next_batch

    def add_result(self, score_to_minimize, pipeline, trained_pipeline_results):
        """Register results from evaluating a pipeline.

        Args:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
            trained_pipeline_results (dict): Results from training a pipeline.

        Raises:
            ValueError: If default parameters are not in the acceptable hyperparameter ranges.
        """
        if pipeline.model_family != ModelFamily.ENSEMBLE:
            if self.batch_number == 1:
                try:
                    super().add_result(
                        score_to_minimize, pipeline, trained_pipeline_results
                    )
                except ValueError as e:
                    if "is not within the bounds of the space" in str(e):
                        raise ValueError(
                            "Default parameters for components in pipeline {} not in the hyperparameter ranges: {}".format(
                                pipeline.name, e
                            )
                        )
                    else:
                        raise (e)
            else:
                super().add_result(
                    score_to_minimize, pipeline, trained_pipeline_results
                )
        if self.batch_number == 1:
            self._first_batch_results.append((score_to_minimize, pipeline))
        current_best_score = self._best_pipeline_info.get(
            pipeline.model_family, {}
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
                    }
                }
            )

    def _transform_parameters(self, pipeline, proposed_parameters):
        """Given a pipeline parameters dict, make sure n_jobs and number_features are set."""
        parameters = {}
        if "pipeline" in self._pipeline_params:
            parameters["pipeline"] = self._pipeline_params["pipeline"]

        for (
            name,
            component_instance,
        ) in pipeline.component_graph.component_instances.items():
            component_class = type(component_instance)
            component_parameters = proposed_parameters.get(name, {})
            init_params = inspect.signature(component_class.__init__).parameters
            # For first batch, pass the pipeline params to the components that need them
            if name in self._custom_hyperparameters and self._batch_number == 0:
                for param_name, value in self._custom_hyperparameters[name].items():
                    if isinstance(value, (Integer, Real)):
                        # get a random value in the space
                        component_parameters[param_name] = value.rvs(
                            random_state=self.random_seed
                        )[0]
                    # Categorical
                    else:
                        component_parameters[param_name] = value.rvs(
                            random_state=self.random_seed
                        )
            if name in self._pipeline_params and self._batch_number == 0:
                for param_name, value in self._pipeline_params[name].items():
                    component_parameters[param_name] = value
            # Inspects each component and adds the following parameters when needed
            if "n_jobs" in init_params:
                component_parameters["n_jobs"] = self.n_jobs
            if "number_features" in init_params:
                component_parameters["number_features"] = self.number_features
            if (
                name in self._pipeline_params
                and name == "Drop Columns Transformer"
                and self._batch_number > 0
            ):
                component_parameters["columns"] = self._pipeline_params[name]["columns"]
            if "pipeline" in self._pipeline_params:
                for param_name, value in self._pipeline_params["pipeline"].items():
                    if param_name in init_params:
                        component_parameters[param_name] = value
            parameters[name] = component_parameters
        return parameters
