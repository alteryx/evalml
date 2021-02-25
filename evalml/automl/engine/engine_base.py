import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd

from evalml.automl.utils import tune_binary_threshold
from evalml.exceptions import PipelineScoreError
from evalml.model_family import ModelFamily
from evalml.preprocessing import split_data
from evalml.problem_types import is_binary, is_multiclass
from evalml.utils.logger import get_logger
from evalml.utils.woodwork_utils import _convert_woodwork_types_wrapper

logger = get_logger(__file__)


class EngineBase(ABC):
    """Base class for the engine API which handles the fitting and evaluation of pipelines during AutoML."""

    def __init__(self, X_train=None, y_train=None, automl=None, should_continue_callback=None, pre_evaluation_callback=None, post_evaluation_callback=None):
        """Base class for the engine API which handles the fitting and evaluation of pipelines during AutoML.

        Arguments:
            X_train (ww.DataTable): training features
            y_train (ww.DataColumn): training target
            automl (AutoMLSearch): a reference to the AutoML search. Used to access configuration and by the error callback.
            should_continue_callback (function): returns True if another pipeline from the list should be evaluated, False otherwise.
            pre_evaluation_callback (function): optional callback invoked before pipeline evaluation.
            post_evaluation_callback (function): optional callback invoked after pipeline evaluation, with args pipeline and evaluation results. Expected to return a list of pipeline IDs corresponding to each pipeline evaluation.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.automl = automl
        self._should_continue_callback = should_continue_callback
        self._pre_evaluation_callback = pre_evaluation_callback
        self._post_evaluation_callback = post_evaluation_callback

    @abstractmethod
    def evaluate_batch(self, pipelines):
        """Evaluate a batch of pipelines using the current dataset and AutoML state.

        Arguments:
            pipeline_batch (list(PipelineBase)): A batch of pipelines to be fitted and evaluated

        Returns:
            list (int): a list of the new pipeline IDs which were created by the AutoML search.
        """

    @staticmethod
    def train_and_score_pipeline(pipeline, automl, full_X_train, full_y_train):
        """Given a pipeline, config and data, train and score the pipeline and return the CV or TV scores

        Arguments:
            pipeline (PipelineBase): the pipeline to score
            automl (AutoMLSearch): the AutoML search, used to access config and for the error callback
            full_X_train (ww.DataTable): training features
            full_y_train (ww.DataColumn): training target

        Returns:
            dict: a dict containing cv_score_mean, cv_scores, training_time and a cv_data structure with details.
        """
        start = time.time()
        cv_data = []
        logger.info("\tStarting cross validation")
        X_pd = _convert_woodwork_types_wrapper(full_X_train.to_dataframe())
        y_pd = _convert_woodwork_types_wrapper(full_y_train.to_series())
        for i, (train, valid) in enumerate(automl.data_splitter.split(X_pd, y_pd)):
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
                X_threshold_tuning = None
                y_threshold_tuning = None
                if automl.optimize_thresholds and automl.objective.is_defined_for_problem_type(automl.problem_type) and \
                   automl.objective.can_optimize_threshold and is_binary(automl.problem_type):
                    X_train, X_threshold_tuning, y_train, y_threshold_tuning = split_data(X_train, y_train, automl.problem_type,
                                                                                          test_size=0.2,
                                                                                          random_seed=automl.random_seed)
                cv_pipeline = pipeline.clone()
                logger.debug(f"\t\t\tFold {i}: starting training")
                cv_pipeline.fit(X_train, y_train)
                logger.debug(f"\t\t\tFold {i}: finished training")
                tune_binary_threshold(cv_pipeline, automl.objective, automl.problem_type,
                                      X_threshold_tuning, y_threshold_tuning)
                if X_threshold_tuning:
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
