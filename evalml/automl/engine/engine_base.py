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
from evalml.utils.woodwork_utils import _convert_woodwork_types_wrapper


class EngineComputation(ABC):
    """Wrapper around the result of a (possibly asynchronous) engine computation."""

    @abstractmethod
    def get_result(self):
        """Get the computation result.
        Will block until the computation is finished.

        Raises Exception: If computation fails. Returns traceback.
        """

    @abstractmethod
    def done(self) -> bool:
        """Is the computation done?"""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the computation."""


class JobLogger:
    """Mimics the behavior of a RootLogger but stores all messages rather than actually logging them.

    This is used during engine jobs so that log messages are recorded after the job completes. This is desired so that
    all of the messages for a single job are grouped together in the log.
    """

    def __init__(self):
        self.logs = []

    def info(self, msg):
        """Store message at the info level."""
        self.logs.append(("info", msg))

    def debug(self, msg):
        """Store message at the debug level."""
        self.logs.append(("debug", msg))

    def warning(self, msg):
        """Store message at the warning level."""
        self.logs.append(("warning", msg))

    def error(self, msg):
        """Store message at the error level."""
        self.logs.append(("error", msg))

    def write_to_logger(self, logger):
        """Write all the messages to the logger. First in First Out order."""
        logger_method = {"info": logger.info,
                         "debug": logger.debug,
                         "warning": logger.warning,
                         "error": logger.warning}
        for level, message in self.logs:
            method = logger_method[level]
            method(message)


class EngineBase(ABC):

    @staticmethod
    def setup_job_log():
        return JobLogger()

    @abstractmethod
    def submit_evaluation_job(self, automl_data, pipeline, X, y):
        """Submit job for pipeline evaluation during AutoMLSearch."""

    @abstractmethod
    def submit_training_job(self, automl_data, pipeline, X, y):
        """Submit job for pipeline training."""

    @abstractmethod
    def submit_scoring_job(self, automl_data, pipeline, X, y, objectives):
        """Submit job for pipeline scoring."""


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


def train_and_score_pipeline(pipeline, automl_data, full_X_train, full_y_train, logger):
    """Given a pipeline, config and data, train and score the pipeline and return the CV or TV scores

    Arguments:
        pipeline (PipelineBase): The pipeline to score
        automl_data (AutoMLSearch): The AutoML search, used to access config and for the error callback
        full_X_train (ww.DataTable): Training features
        full_y_train (ww.DataColumn): Training target

    Returns:
        dict: A dict containing cv_score_mean, cv_scores, training_time and a cv_data structure with details.
    """
    start = time.time()
    cv_data = []
    logger.info("\t\tStarting cross validation")
    X_pd = _convert_woodwork_types_wrapper(full_X_train.to_dataframe())
    y_pd = _convert_woodwork_types_wrapper(full_y_train.to_series())
    cv_pipeline = pipeline
    for i, (train, valid) in enumerate(automl_data.data_splitter.split(X_pd, y_pd)):
        if pipeline.model_family == ModelFamily.ENSEMBLE and i > 0:
            # Stacked ensembles do CV internally, so we do not run CV here for performance reasons.
            logger.debug(f"Skipping fold {i} because CV for stacked ensembles is not supported.")
            break
        logger.debug(f"\t\t\tTraining and scoring on fold {i}")
        X_train, X_valid = full_X_train.iloc[train], full_X_train.iloc[valid]
        y_train, y_valid = full_y_train.iloc[train], full_y_train.iloc[valid]
        if is_binary(automl_data.problem_type) or is_multiclass(automl_data.problem_type):
            diff_train = set(np.setdiff1d(full_y_train.to_series(), y_train.to_series()))
            diff_valid = set(np.setdiff1d(full_y_train.to_series(), y_valid.to_series()))
            diff_string = f"Missing target values in the training set after data split: {diff_train}. " if diff_train else ""
            diff_string += f"Missing target values in the validation set after data split: {diff_valid}." if diff_valid else ""
            if diff_string:
                raise Exception(diff_string)
        objectives_to_score = [automl_data.objective] + automl_data.additional_objectives
        try:
            logger.debug(f"\t\t\t\tFold {i}: starting training")
            cv_pipeline = train_pipeline(pipeline, X_train, y_train, automl_data.optimize_thresholds, automl_data.objective)
            logger.debug(f"\t\t\t\tFold {i}: finished training")
            if automl_data.optimize_thresholds and pipeline.can_tune_threshold_with_objective(automl_data.objective):
                logger.debug(f"\t\t\t\tFold {i}: Optimal threshold found ({cv_pipeline.threshold:.3f})")
            logger.debug(f"\t\t\t\tFold {i}: Scoring trained pipeline")
            scores = cv_pipeline.score(X_valid, y_valid, objectives=objectives_to_score)
            logger.debug(f"\t\t\t\tFold {i}: {automl_data.objective.name} score: {scores[automl_data.objective.name]:.3f}")
            score = scores[automl_data.objective.name]
        except Exception as e:
            if automl_data.error_callback is not None:
                automl_data.error_callback(exception=e, traceback=traceback.format_tb(sys.exc_info()[2]), automl=automl_data,
                                           fold_num=i, pipeline=pipeline)
            if isinstance(e, PipelineScoreError):
                nan_scores = {objective: np.nan for objective in e.exceptions}
                scores = {**nan_scores, **e.scored_successfully}
                scores = OrderedDict({o.name: scores[o.name] for o in [automl_data.objective] + automl_data.additional_objectives})
                score = scores[automl_data.objective.name]
            else:
                score = np.nan
                scores = OrderedDict(zip([n.name for n in automl_data.additional_objectives], [np.nan] * len(automl_data.additional_objectives)))

        ordered_scores = OrderedDict()
        ordered_scores.update({automl_data.objective.name: score})
        ordered_scores.update(scores)
        ordered_scores.update({"# Training": y_train.shape[0]})
        ordered_scores.update({"# Validation": y_valid.shape[0]})

        evaluation_entry = {"all_objective_scores": ordered_scores, "score": score, 'binary_classification_threshold': None}
        if is_binary(automl_data.problem_type) and cv_pipeline is not None and cv_pipeline.threshold is not None:
            evaluation_entry['binary_classification_threshold'] = cv_pipeline.threshold
        cv_data.append(evaluation_entry)
    training_time = time.time() - start
    cv_scores = pd.Series([fold['score'] for fold in cv_data])
    cv_score_mean = cv_scores.mean()
    logger.info(f"\t\tFinished cross validation - mean {automl_data.objective.name}: {cv_score_mean:.3f}")
    return {'cv_data': cv_data, 'training_time': training_time, 'cv_scores': cv_scores, 'cv_score_mean': cv_score_mean}, cv_pipeline, logger


def evaluate_pipeline(pipeline, automl_data, X, y, logger):
    logger.info(f"\t{pipeline.name}:")

    X_train, y_train = X, y

    if pipeline.model_family == ModelFamily.ENSEMBLE:
        X_train, y_train = X.iloc[automl_data.ensembling_indices], y.iloc[automl_data.ensembling_indices]
    elif automl_data.ensembling_indices is not None:
        training_indices = [i for i in range(len(X)) if i not in automl_data.ensembling_indices]
        X_train = X.iloc[training_indices]
        y_train = y.iloc[training_indices]

    return train_and_score_pipeline(pipeline, automl_data=automl_data, full_X_train=X_train, full_y_train=y_train,
                                    logger=logger)


def score_pipeline(pipeline, X, y, objectives):
    return pipeline.score(X, y, objectives)
