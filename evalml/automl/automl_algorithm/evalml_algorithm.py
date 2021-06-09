import inspect
from operator import itemgetter

import numpy as np
from skopt.space import Categorical, Integer, Real

from .automl_algorithm import AutoMLAlgorithm, AutoMLAlgorithmException

from evalml.model_family import ModelFamily
from evalml.pipelines.utils import _make_stacked_ensemble_pipeline


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

        self.n_jobs = n_jobs
        self.number_features = number_features
        self._best_pipeline_info = {}
        self.text_in_ensembling = text_in_ensembling
        self._pipeline_params = pipeline_params or {}
        self._custom_hyperparameters = custom_hyperparameters or {}
        self._frozen_pipeline_parameters = _frozen_pipeline_parameters or {}

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
        """Get the next batch of pipelines to evaluate

        Returns:
            list(PipelineBase): a list of instances of PipelineBase subclasses, ready to be trained and evaluated.
        """

        pass
