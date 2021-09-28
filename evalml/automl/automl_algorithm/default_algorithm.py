"""An automl algorithm that consists of two modes: fast and long, where fast is a subset of long."""
import inspect

import numpy as np
from skopt.space import Categorical, Integer, Real

from .automl_algorithm import AutoMLAlgorithm

from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    RFClassifierSelectFromModel,
    RFRegressorSelectFromModel,
)
from evalml.pipelines.components.transformers.column_selectors import (
    SelectColumns,
)
from evalml.pipelines.components.utils import (
    get_estimators,
    handle_component_class,
)
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import is_regression


class DefaultAlgorithm(AutoMLAlgorithm):
    """An automl algorithm that consists of two modes: fast and long, where fast is a subset of long.

    1. Naive pipelines:
        a. run baseline with default preprocessing pipeline
        b. run naive linear model with default preprocessing pipeline
        c. run basic RF pipeline with default preprocessing pipeline
    2. Naive pipelines with feature selection
        a. subsequent pipelines will use the selected features with a SelectedColumns transformer

    At this point we have a single pipeline candidate for preprocessing and feature selection

    3. Pipelines with preprocessing components:
        a. scan rest of estimators (our current batch 1).
    4. First ensembling run

    Fast mode ends here. Begin long mode.

    6. Run top 3 estimators:
        a. Generate 50 random parameter sets. Run all 150 in one batch
    7. Second ensembling run
    8. Repeat these indefinitely until stopping criterion is met:
        a. For each of the previous top 3 estimators, sample 10 parameters from the tuner. Run all 30 in one batch
        b. Run ensembling

    Args:
        X (pd.DataFrame): Training data.
        y (pd.Series): Target data.
        problem_type (ProblemType): Problem type associated with training data.
        sampler_name (BaseSampler): Sampler to use for preprocessing.
        tuner_class (class): A subclass of Tuner, to be used to find parameters for each pipeline. The default of None indicates the SKOptTuner will be used.
        random_seed (int): Seed for the random number generator. Defaults to 0.
        pipeline_params (dict or None): Pipeline-level parameters that should be passed to the proposed pipelines. Defaults to None.
        custom_hyperparameters (dict or None): Custom hyperparameter ranges specified for pipelines to iterate over. Defaults to None.
        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines. Defaults to -1.
        text_in_ensembling (boolean): If True and ensembling is True, then n_jobs will be set to 1 to avoid downstream sklearn stacking issues related to nltk. Defaults to None.
        top_n (int): top n number of pipelines to use for long mode.
        num_long_explore_pipelines (int): number of pipelines to explore for each top n pipeline at the start of long mode.
        num_long_pipelines_per_batch (int): number of pipelines per batch for each top n pipeline through long mode.
    """

    def __init__(
        self,
        X,
        y,
        problem_type,
        sampler_name,
        tuner_class=None,
        random_seed=0,
        pipeline_params=None,
        custom_hyperparameters=None,
        n_jobs=-1,
        text_in_ensembling=None,
        top_n=3,
        num_long_explore_pipelines=50,
        num_long_pipelines_per_batch=10,
    ):
        super().__init__(
            allowed_pipelines=[],
            custom_hyperparameters=custom_hyperparameters,
            max_iterations=None,
            tuner_class=None,
            random_seed=random_seed,
        )

        self.X = X
        self.y = y
        self.problem_type = problem_type
        self.sampler_name = sampler_name

        self.n_jobs = n_jobs
        self._best_pipeline_info = {}
        self.text_in_ensembling = text_in_ensembling
        self._pipeline_params = pipeline_params or {}
        self._custom_hyperparameters = custom_hyperparameters or {}
        self._top_n_pipelines = None
        self.num_long_explore_pipelines = num_long_explore_pipelines
        self.num_long_pipelines_per_batch = num_long_pipelines_per_batch
        self.top_n = top_n
        if custom_hyperparameters and not isinstance(custom_hyperparameters, dict):
            raise ValueError(
                f"If custom_hyperparameters provided, must be of type dict. Received {type(custom_hyperparameters)}"
            )

        for param_name_val in self._pipeline_params.values():
            for param_val in param_name_val.values():
                if isinstance(param_val, (Integer, Real, Categorical)):
                    raise ValueError(
                        "Pipeline parameters should not contain skopt.Space variables, please pass them "
                        "to custom_hyperparameters instead!"
                    )
        for hyperparam_name_val in self._custom_hyperparameters.values():
            for hyperparam_val in hyperparam_name_val.values():
                if not isinstance(hyperparam_val, (Integer, Real, Categorical)):
                    raise ValueError(
                        "Custom hyperparameters should only contain skopt.Space variables such as Categorical, Integer,"
                        " and Real!"
                    )

    def _naive_estimators(self):
        if is_regression(self.problem_type):
            naive_estimators = [
                "Elastic Net Regressor",
                "Random Forest Regressor",
            ]
        else:
            naive_estimators = [
                "Logistic Regression Classifier",
                "Random Forest Classifier",
            ]
        estimators = [
            handle_component_class(estimator) for estimator in naive_estimators
        ]
        return estimators

    def _create_tuner(self, pipeline):
        pipeline_hyperparameters = pipeline.get_hyperparameter_ranges(
            self._custom_hyperparameters
        )
        self._tuners[pipeline.name] = self._tuner_class(
            pipeline_hyperparameters, random_seed=self.random_seed
        )

    def _create_pipelines_with_params(self, pipelines, parameters={}):
        return [
            pipeline.new(
                parameters=self._transform_parameters(pipeline, parameters),
                random_seed=self.random_seed,
            )
            for pipeline in pipelines
        ]

    def _create_naive_pipelines(self, use_features=False):
        feature_selector = None

        if use_features:
            feature_selector = [
                (
                    RFRegressorSelectFromModel
                    if is_regression(self.problem_type)
                    else RFClassifierSelectFromModel
                )
            ]
        else:
            feature_selector = []

        estimators = self._naive_estimators()
        parameters = self._pipeline_params if self._pipeline_params else None
        pipelines = [
            make_pipeline(
                self.X,
                self.y,
                estimator,
                self.problem_type,
                sampler_name=self.sampler_name,
                parameters=parameters,
                extra_components=feature_selector,
            )
            for estimator in estimators
        ]

        pipelines = self._create_pipelines_with_params(pipelines, parameters={})
        return pipelines

    def _create_fast_final(self):
        estimators = [
            estimator
            for estimator in get_estimators(self.problem_type)
            if estimator not in self._naive_estimators()
        ]
        parameters = self._pipeline_params if self._pipeline_params else {}
        parameters.update(
            {"Select Columns Transformer": {"columns": self._selected_cols}}
        )
        pipelines = [
            make_pipeline(
                self.X,
                self.y,
                estimator,
                self.problem_type,
                sampler_name=self.sampler_name,
                parameters=parameters,
                extra_components=[SelectColumns],
            )
            for estimator in estimators
        ]

        pipelines = self._create_pipelines_with_params(
            pipelines, {"Select Columns Transformer": {"columns": self._selected_cols}}
        )

        for pipeline in pipelines:
            self._create_tuner(pipeline)
        return pipelines

    def _create_n_pipelines(self, pipelines, n):
        next_batch = []
        for _ in range(n):
            for pipeline in pipelines:
                if pipeline.name not in self._tuners:
                    self._create_tuner(pipeline)
                proposed_parameters = self._tuners[pipeline.name].propose()
                parameters = self._transform_parameters(pipeline, proposed_parameters)
                parameters.update(
                    {"Select Columns Transformer": {"columns": self._selected_cols}}
                )
                next_batch.append(
                    pipeline.new(parameters=parameters, random_seed=self.random_seed)
                )
        return next_batch

    def _create_long_exploration(self, n):
        estimators = [
            (pipeline_dict["pipeline"].estimator, pipeline_dict["mean_cv_score"])
            for pipeline_dict in self._best_pipeline_info.values()
        ]
        estimators.sort(key=lambda x: x[1])
        estimators = estimators[:n]
        estimators = [estimator[0].__class__ for estimator in estimators]
        pipelines = [
            make_pipeline(
                self.X,
                self.y,
                estimator,
                self.problem_type,
                sampler_name=self.sampler_name,
                extra_components=[SelectColumns],
            )
            for estimator in estimators
        ]
        self._top_n_pipelines = pipelines
        return self._create_n_pipelines(pipelines, self.num_long_explore_pipelines)

    def next_batch(self):
        """Get the next batch of pipelines to evaluate.

        Returns:
            list(PipelineBase): a list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """
        if self._batch_number == 0:
            next_batch = self._create_naive_pipelines()
        elif self._batch_number == 1:
            next_batch = self._create_naive_pipelines(use_features=True)
        elif self._batch_number == 2:
            next_batch = self._create_fast_final()
        elif self.batch_number == 3:
            next_batch = self._create_ensemble()
        elif self.batch_number == 4:
            next_batch = self._create_long_exploration(n=self.top_n)
        elif self.batch_number % 2 != 0:
            next_batch = self._create_ensemble()
        else:
            next_batch = self._create_n_pipelines(
                self._top_n_pipelines, self.num_long_pipelines_per_batch
            )

        self._pipeline_number += len(next_batch)
        self._batch_number += 1
        return next_batch

    def add_result(self, score_to_minimize, pipeline, trained_pipeline_results):
        """Register results from evaluating a pipeline. In batch number 2, the selected column names from the feature selector are taken to be used in a column selector. Information regarding the best pipeline is updated here as well.

        Args:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
            trained_pipeline_results (dict): Results from training a pipeline.
        """
        if pipeline.model_family != ModelFamily.ENSEMBLE:
            if self.batch_number >= 3:
                super().add_result(
                    score_to_minimize, pipeline, trained_pipeline_results
                )

        if self.batch_number == 2 and self._selected_cols is None:
            if is_regression(self.problem_type):
                self._selected_cols = pipeline.get_component(
                    "RF Regressor Select From Model"
                ).get_names()
            else:
                self._selected_cols = pipeline.get_component(
                    "RF Classifier Select From Model"
                ).get_names()

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
        """Given a pipeline parameters dict, make sure pipeline_params, custom_hyperparameters, n_jobs are set properly."""
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
            if name in self._custom_hyperparameters and self._batch_number <= 2:
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
            if name in self._pipeline_params:
                for param_name, value in self._pipeline_params[name].items():
                    component_parameters[param_name] = value
            # Inspects each component and adds the following parameters when needed
            if "n_jobs" in init_params:
                component_parameters["n_jobs"] = self.n_jobs
            parameters[name] = component_parameters
        return parameters
