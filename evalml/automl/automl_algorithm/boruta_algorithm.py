import numpy as np

from evalml.automl.automl_algorithm.default_algorithm import DefaultAlgorithm
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    RFClassifierBorutaSelector,
    RFRegressorBorutaSelector,
)
from evalml.problem_types import is_regression, is_time_series


class BorutaAlgorithm(DefaultAlgorithm):
    def _get_feature_selectors(self):
        return [
            (
                RFRegressorBorutaSelector
                if is_regression(self.problem_type)
                else RFClassifierBorutaSelector
            ),
        ]

    def add_result(
        self,
        score_to_minimize,
        pipeline,
        trained_pipeline_results,
        cached_data=None,
    ):
        """Register results from evaluating a pipeline. In batch number 2, the selected column names from the feature selector are taken to be used in a column selector. Information regarding the best pipeline is updated here as well.

        Args:
            score_to_minimize (float): The score obtained by this pipeline on the primary objective, converted so that lower values indicate better pipelines.
            pipeline (PipelineBase): The trained pipeline object which was used to compute the score.
            trained_pipeline_results (dict): Results from training a pipeline.
            cached_data (dict): A dictionary of cached data, where the keys are the model family. Expected to be of format
                {model_family: {hash1: trained_component_graph, hash2: trained_component_graph...}...}.
                Defaults to None.
        """
        cached_data = cached_data or {}
        if pipeline.model_family != ModelFamily.ENSEMBLE:
            if self.batch_number >= 3:
                super().add_result(
                    score_to_minimize,
                    pipeline,
                    trained_pipeline_results,
                )

        if (
            self.batch_number == 2
            and self._selected_cols is None
            and not is_time_series(self.problem_type)
        ):
            self._selected_cols = pipeline.get_component(
                self._get_feature_selectors()[0].name,
            ).get_names()
            self._parse_selected_categorical_features(pipeline)

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
