import sys
import traceback

import numpy as np

from evalml.automl.engine import EngineBase
from evalml.exceptions import PipelineScoreError
from evalml.model_family import ModelFamily
from evalml.objectives.utils import get_objective
from evalml.utils import get_logger

logger = get_logger(__file__)


class SequentialEngine(EngineBase):
    """The default engine for the AutoML search. Trains and scores pipelines locally, one after another."""

    def evaluate_batch(self, pipelines):
        """Evaluate a batch of pipelines using the current dataset and AutoML state.

        Arguments:
            pipelines (list(PipelineBase)): A batch of pipelines to be fitted and evaluated.

        Returns:
            list (int): a list of the new pipeline IDs which were created by the AutoML search.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset has not been loaded into the engine.")
        new_pipeline_ids = []
        index = 0
        while self._should_continue_callback() and index < len(pipelines):
            pipeline = pipelines[index]
            self._pre_evaluation_callback(pipeline)
            X, y = self.X_train, self.y_train
            if pipeline.model_family == ModelFamily.ENSEMBLE:
                X, y = self.X_train.iloc[self.ensembling_indices], self.y_train.iloc[self.ensembling_indices]
            elif self.ensembling_indices is not None:
                training_indices = [i for i in range(len(self.X_train)) if i not in self.ensembling_indices]
                X = self.X_train.iloc[training_indices]
                y = self.y_train.iloc[training_indices]
            evaluation_result = EngineBase.train_and_score_pipeline(pipeline, self.automl, X, y)
            new_pipeline_ids.append(self._post_evaluation_callback(pipeline, evaluation_result))
            index += 1
        return new_pipeline_ids

    def train_batch(self, pipelines):
        """Train a batch of pipelines using the current dataset.

        Arguments:
            pipelines (list(PipelineBase)): A batch of pipelines to fit.
        Returns:
            dict[str, PipelineBase]: Dict of fitted pipelines keyed by pipeline name.
        """
        super().train_batch(pipelines)

        X_train = self.X_train
        y_train = self.y_train
        if hasattr(self.automl.data_splitter, "transform_sample"):
            train_indices = self.automl.data_splitter.transform_sample(X_train, y_train)
            X_train = X_train.iloc[train_indices]
            y_train = y_train.iloc[train_indices]

        fitted_pipelines = {}
        for pipeline in pipelines:
            try:
                fitted_pipeline = EngineBase.train_pipeline(
                    pipeline, X_train, y_train,
                    self.automl.optimize_thresholds,
                    self.automl.objective
                )
                fitted_pipelines[fitted_pipeline.name] = fitted_pipeline
            except Exception as e:
                logger.error(f'Train error for {pipeline.name}: {str(e)}')
                tb = traceback.format_tb(sys.exc_info()[2])
                logger.error("Traceback:")
                logger.error("\n".join(tb))

        return fitted_pipelines

    def score_batch(self, pipelines, X, y, objectives):
        """Score a batch of pipelines.

        Arguments:
            pipelines (list(PipelineBase)): A batch of fitted pipelines to score.
            X (ww.DataTable, pd.DataFrame): Features to score on.
            y (ww.DataTable, pd.DataFrame): Data to score on.
            objectives (list(ObjectiveBase), list(str)): Objectives to score on.
        Returns:
            dict: Dict containing scores for all objectives for all pipelines. Keyed by pipeline name.
        """
        super().score_batch(pipelines, X, y, objectives)

        scores = {}
        objectives = [get_objective(o, return_instance=True) for o in objectives]
        for pipeline in pipelines:
            try:
                scores[pipeline.name] = pipeline.score(X, y, objectives)
            except Exception as e:
                logger.error(f"Score error for {pipeline.name}: {str(e)}")
                if isinstance(e, PipelineScoreError):
                    nan_scores = {objective: np.nan for objective in e.exceptions}
                    scores[pipeline.name] = {**nan_scores, **e.scored_successfully}
                else:
                    # Traceback already included in the PipelineScoreError so we only
                    # need to include it for all other errors
                    tb = traceback.format_tb(sys.exc_info()[2])
                    logger.error("Traceback:")
                    logger.error("\n".join(tb))
                    scores[pipeline.name] = {objective.name: np.nan for objective in objectives}

        return scores
