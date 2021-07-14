import inspect

import numpy as np
from skopt.space import Categorical, Integer, Real

from .automl_algorithm import AutoMLAlgorithm

from evalml.automl.utils import get_hyperparameter_ranges
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
from evalml.pipelines.utils import (
    _make_stacked_ensemble_pipeline,
    make_pipeline,
)
from evalml.problem_types import is_regression


class EvalMLAlgorithm(AutoMLAlgorithm):
    """An automl algorithm that consists of two modes: fast and long. Where fast is a subset of long.

    1. Naive pipelines:
        a. run baseline with default preprocessing pipeline
        b. run naive linear model with default preprocessing pipeline
        c. run basic RF pipeline (for feature selection) with default preprocessing pipeline
    2. Feature engineering and naive pipelines with feature selection:
        a. create feature selection component with previous batches’ RF estimator then add to another linear model
        b. Run feature engineering: leveraging featuretools and our DFSTransformer
    3. Naive pipelines with feature engineering
        a. Use FT component from previous batch with naive linear model and RF pipeline
    4. Naive pipelines with feature engineering and feature selection
        a. use previous RF estimator to run FS with naive linear model

    At this point we have a single pipeline candidate for preprocessing, feature engineering and feature selection

    5. Pipelines with preprocessing components:
        a. scan estimators (our current batch 1).
        b. Then run ensembling

    Fast mode ends here. Begin long mode.

    6. Run some random pipelines:
        a. Choose top 3 estimators. Generate 50 random parameter sets. Run all 150 in one batch
    7. Run ensembling
    8. Repeat these indefinitely until stopping criterion is met:
        a. For each of the previous top 3 estimators, sample 10 parameters from the tuner. Run all 30 in one batch
        b. Run ensembling
    """

    """
    Jeremy notes:
        Do we need to allow users to select what models and pipelines are allowed?
            - originally my thinking was we would remove that choice from user
            - however, how does the automl_search impl affect it?

        options:
            - top level user can select and pass down
            - do not let users select

        considerations
            - change `AutoMLAlgorithm` init impl to not create tuners for each pipeline?
            - ignore `AutoMLAlgorithm` init and do our own thing here
    """

    def __init__(
        self,
        X,
        y,
        problem_type,
        _sampler_name,
        tuner_class=None,
        random_seed=0,
        pipeline_params=None,
        custom_hyperparameters=None,
        _frozen_pipeline_parameters=None,
        n_jobs=-1,  # TODO: necessary?
        number_features=None,  # TODO: necessary?
        text_in_ensembling=None,
    ):  # TODO: necessary?

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
        self._sampler_name = _sampler_name

        self.n_jobs = n_jobs
        self.number_features = number_features
        self._best_pipeline_info = {}
        self.text_in_ensembling = text_in_ensembling
        self._pipeline_params = pipeline_params or {}
        self._custom_hyperparameters = custom_hyperparameters or {}
        self._frozen_pipeline_parameters = _frozen_pipeline_parameters or {}

        self._selected_cols = None
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

    def _naive_estimators(self):
        if is_regression(self.problem_type):
            naive_estimators = [
                "Linear Regressor",
                "Random Forest Regressor",
            ]
            estimators = [
                handle_component_class(estimator) for estimator in naive_estimators
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

    def _create_naive_pipelines(self):
        # don't need to add baseline as it is added as part of AutoMLSearch TODO: what to do here
        estimators = self._naive_estimators()
        return [
            make_pipeline(
                self.X,
                self.y,
                estimator,
                self.problem_type,
                parameters=self._frozen_pipeline_parameters,
                sampler_name=self._sampler_name,
            )
            for estimator in estimators
        ]

    def _create_naive_pipelines_with_feature_selection(self):
        feature_selector = (
            RFRegressorSelectFromModel
            if is_regression(self.problem_type)
            else RFClassifierSelectFromModel
        )
        estimators = self._naive_estimators()
        pipelines = [
            make_pipeline(
                self.X,
                self.y,
                estimator,
                self.problem_type,
                parameters=self._frozen_pipeline_parameters,
                sampler_name=self._sampler_name,
                extra_components=[feature_selector],
            )
            for estimator in estimators
        ]
        return pipelines

    def _create_fast_final(self):
        estimators = [
            estimator
            for estimator in get_estimators(self.problem_type)
            if estimator not in self._naive_estimators()
        ]
        pipelines = [
            make_pipeline(
                self.X,
                self.y,
                estimator,
                self.problem_type,
                parameters=self._frozen_pipeline_parameters,
                sampler_name=self._sampler_name,
                extra_components=[SelectColumns],
            )
            for estimator in estimators
        ]
        pipelines = [
            pipeline.new(
                parameters={
                    "Select Columns Transformer": {"columns": self._selected_cols}
                },
                random_seed=self.random_seed,
            )
            for pipeline in pipelines
        ]

        for pipeline in pipelines:
            pipeline_hyperparameters = get_hyperparameter_ranges(
                pipeline.component_graph, self._custom_hyperparameters
            )
            self._tuners[pipeline.name] = self._tuner_class(
                pipeline_hyperparameters, random_seed=self.random_seed
            )

        return pipelines

    def _create_ensemble(self):
        input_pipelines = []
        for pipeline_dict in self._best_pipeline_info.values():
            pipeline = pipeline_dict["pipeline"]
            pipeline_params = pipeline_dict["parameters"]
            parameters = self._transform_parameters(pipeline, pipeline_params)
            input_pipelines.append(
                pipeline.new(parameters=parameters, random_seed=self.random_seed)
            )
        n_jobs_ensemble = 1 if self.text_in_ensembling else self.n_jobs
        ensemble = _make_stacked_ensemble_pipeline(
            input_pipelines,
            input_pipelines[0].problem_type,
            random_seed=self.random_seed,
            n_jobs=n_jobs_ensemble,
        )
        return [ensemble]

    def next_batch(self):
        """Get the next batch of pipelines to evaluate

        Returns:
            list(PipelineBase): a list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """

        if self._batch_number == 0:
            next_batch = self._create_naive_pipelines()
        elif self._batch_number == 1:
            next_batch = self._create_naive_pipelines_with_feature_selection()
        elif self._batch_number == 2:
            next_batch = self._create_fast_final()
        elif self.batch_number == 3:
            next_batch = self._create_ensemble()

        self._pipeline_number += len(next_batch)
        self._batch_number += 1
        return next_batch

    def add_result(self, score_to_minimize, pipeline, trained_pipeline_results):
        """Register results from evaluating a pipeline

        Arguments:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
            trained_pipeline_results (dict): Results from training a pipeline.
        """
        if pipeline.model_family != ModelFamily.ENSEMBLE:
            if self.batch_number >= 3:
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
        """Given a pipeline parameters dict, make sure n_jobs and number_features are set."""
        parameters = {}
        if "pipeline" in self._pipeline_params:
            parameters["pipeline"] = self._pipeline_params["pipeline"]

        for name, component_class in pipeline.linearized_component_graph:
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
