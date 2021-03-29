import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd

from evalml.automl.utils import (
    check_all_pipeline_names_unique,
    tune_binary_threshold
)
from evalml.exceptions import PipelineScoreError
from evalml.model_family import ModelFamily
from evalml.preprocessing import split_data
from evalml.problem_types import is_binary, is_classification, is_multiclass
from evalml.utils.logger import get_logger
from evalml.utils.woodwork_utils import _convert_woodwork_types_wrapper

logger = get_logger(__file__)


class EngineBase(ABC):
    """Base class for the engine API which handles the fitting and evaluation of pipelines during AutoML."""

    def __init__(self, X_train=None, y_train=None, ensembling_indices=None, automl=None, should_continue_callback=None, pre_evaluation_callback=None, post_evaluation_callback=None):
        """Base class for the engine API which handles the fitting and evaluation of pipelines during AutoML.

        Arguments:
            X_train (ww.DataTable): Training features
            y_train (ww.DataColumn): Training target
            ensembling_indices (list): Ensembling indices for ensembling data
            automl (AutoMLSearch): A reference to the AutoML search. Used to access configuration and by the error callback.
            should_continue_callback (function): Returns True if another pipeline from the list should be evaluated, False otherwise.
            pre_evaluation_callback (function): Optional callback invoked before pipeline evaluation.
            post_evaluation_callback (function): Optional callback invoked after pipeline evaluation, with args pipeline and evaluation results. Expected to return a list of pipeline IDs corresponding to each pipeline evaluation.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.automl = automl
        self._should_continue_callback = should_continue_callback
        self._pre_evaluation_callback = pre_evaluation_callback
        self._post_evaluation_callback = post_evaluation_callback
        self.ensembling_indices = ensembling_indices

    @abstractmethod
    def evaluate_batch(self, pipelines):
        """Evaluate a batch of pipelines using the current dataset and AutoML state.

        Arguments:
            pipeline_batch (list(PipelineBase)): A batch of pipelines to be fitted and evaluated

        Returns:
            list (int): A list of the new pipeline IDs which were created by the AutoML search.
        """

    @abstractmethod
    def train_batch(self, pipelines):
        """Train a batch of pipelines using the current dataset.

        Arguments:
            pipelines (list(PipelineBase)): A batch of pipelines to fit.
        Returns:
            list(PipelineBase): List of fitted pipelines
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Dataset has not been loaded into the engine.")

        check_all_pipeline_names_unique(pipelines)

    @abstractmethod
    def score_batch(self, pipelines, X, y, objectives):
        """Score a batch of pipelines.

        Arguments:
            pipelines (list(PipelineBase)): A batch of fitted pipelines to score.
            X (ww.DataTable, pd.DataFrame): Features to score on.
            y (ww.DataTable, pd.DataFrame): Data to score on.
            objectives (list(ObjectiveBase), list(str)): Objectives to score on.
        Returns:
            Dict[pipeline name, score]: Scores for all objectives for all pipelines.
        """
        check_all_pipeline_names_unique(pipelines)

    @staticmethod
    def train_pipeline(pipeline, X, y, optimize_thresholds, objective):
        """Train a pipeline and tune the threshold if necessary.

        Arguments:
            pipeline (PipelineBase): Pipeline to train.
            X (ww.DataTable, pd.DataFrame): Features to train on.
            y (ww.DataColumn, pd.Series): Target to train on.
            optimize_thresholds (bool): Whether to tune the threshold (if pipeline supports it).
            objective (ObjectiveBase): Objective used in threshold tuning.

        Returns:
            pipeline (PipelineBase) - trained pipeline.
        """
        X_threshold_tuning = None
        y_threshold_tuning = None
        if optimize_thresholds and pipeline.can_tune_threshold_with_objective(objective):
            X, X_threshold_tuning, y, y_threshold_tuning = split_data(X, y, pipeline.problem_type,
                                                                      test_size=0.2, random_seed=pipeline.random_seed)
        cv_pipeline = pipeline.clone()
        cv_pipeline.fit(X, y)
        tune_binary_threshold(cv_pipeline, objective, cv_pipeline.problem_type,
                              X_threshold_tuning, y_threshold_tuning)
        return cv_pipeline

    @staticmethod
    def train_and_score_pipeline(pipeline, automl, full_X_train, full_y_train):
        """Given a pipeline, config and data, train and score the pipeline and return the CV or TV scores

        Arguments:
            pipeline (PipelineBase): The pipeline to score
            automl (AutoMLSearch): The AutoML search, used to access config and for the error callback
            full_X_train (ww.DataTable): Training features
            full_y_train (ww.DataColumn): Training target

        Returns:
            dict: A dict containing cv_score_mean, cv_scores, training_time and a cv_data structure with details.
        """
        start = time.time()
        cv_data = []
        logger.info("\tStarting cross validation")
        X_pd = _convert_woodwork_types_wrapper(full_X_train.to_dataframe())
        y_pd = _convert_woodwork_types_wrapper(full_y_train.to_series())
        y_pd_encoded = y_pd
        # Encode target for classification problems so that we can support float targets. This is okay because we only use split to get the indices to split on
        if is_classification(automl.problem_type):
            y_mapping = {original_target: encoded_target for (encoded_target, original_target) in enumerate(y_pd.value_counts().index)}
            y_pd_encoded = y_pd.map(y_mapping)
        for i, (train, valid) in enumerate(automl.data_splitter.split(X_pd, y_pd_encoded)):
            if pipeline.model_family == ModelFamily.ENSEMBLE and i > 0:
                # Stacked ensembles do CV internally, so we do not run CV here for performance reasons.
                logger.debug(f"Skipping fold {i} because CV for stacked ensembles is not supported.")
                break
            logger.debug(f"\t\tTraining and scoring on fold {i}")
            X_train, X_valid = full_X_train.iloc[train], full_X_train.iloc[valid]
            y_train, y_valid = full_y_train.iloc[train], full_y_train.iloc[valid]
            if is_binary(automl.problem_type) or is_multiclass(automl.problem_type):
                diff_train = set(np.setdiff1d(full_y_train.to_series(), y_train.to_series()))
                diff_valid = set(np.setdiff1d(full_y_train.to_series(), y_valid.to_series()))
                diff_string = f"Missing target values in the training set after data split: {diff_train}. " if diff_train else ""
                diff_string += f"Missing target values in the validation set after data split: {diff_valid}." if diff_valid else ""
                if diff_string:
                    raise Exception(diff_string)
            objectives_to_score = [automl.objective] + automl.additional_objectives
            cv_pipeline = None
            try:
                logger.debug(f"\t\t\tFold {i}: starting training")
                cv_pipeline = EngineBase.train_pipeline(pipeline, X_train, y_train, automl.optimize_thresholds, automl.objective)
                logger.debug(f"\t\t\tFold {i}: finished training")
                if automl.optimize_thresholds and pipeline.can_tune_threshold_with_objective(automl.objective) and automl.objective.can_optimize_threshold:
                    logger.debug(f"\t\t\tFold {i}: Optimal threshold found ({cv_pipeline.threshold:.3f})")
                logger.debug(f"\t\t\tFold {i}: Scoring trained pipeline")
                scores = cv_pipeline.score(X_valid, y_valid, objectives=objectives_to_score)
                logger.debug(f"\t\t\tFold {i}: {automl.objective.name} score: {scores[automl.objective.name]:.3f}")
                score = scores[automl.objective.name]
            except Exception as e:
                if automl.error_callback is not None:
                    automl.error_callback(exception=e, traceback=traceback.format_tb(sys.exc_info()[2]), automl=automl,
                                          fold_num=i, pipeline=pipeline)
                if isinstance(e, PipelineScoreError):
                    nan_scores = {objective: np.nan for objective in e.exceptions}
                    scores = {**nan_scores, **e.scored_successfully}
                    scores = OrderedDict({o.name: scores[o.name] for o in [automl.objective] + automl.additional_objectives})
                    score = scores[automl.objective.name]
                else:
                    score = np.nan
                    scores = OrderedDict(zip([n.name for n in automl.additional_objectives], [np.nan] * len(automl.additional_objectives)))

            ordered_scores = OrderedDict()
            ordered_scores.update({automl.objective.name: score})
            ordered_scores.update(scores)
            ordered_scores.update({"# Training": y_train.shape[0]})
            ordered_scores.update({"# Validation": y_valid.shape[0]})

            evaluation_entry = {"all_objective_scores": ordered_scores, "score": score, 'binary_classification_threshold': None}
            if is_binary(automl.problem_type) and cv_pipeline is not None and cv_pipeline.threshold is not None:
                evaluation_entry['binary_classification_threshold'] = cv_pipeline.threshold
            cv_data.append(evaluation_entry)
        training_time = time.time() - start
        cv_scores = pd.Series([fold['score'] for fold in cv_data])
        cv_score_mean = cv_scores.mean()
        logger.info(f"\tFinished cross validation - mean {automl.objective.name}: {cv_score_mean:.3f}")
        return {'cv_data': cv_data, 'training_time': training_time, 'cv_scores': cv_scores, 'cv_score_mean': cv_score_mean}
