import sys
import time
import traceback
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evalml.exceptions import PipelineScoreError
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)
from evalml.utils.logger import get_logger, update_pipeline

logger = get_logger(__file__)


class EngineBase(ABC):
    """ Base class for engines, which handles the fitting and evaluation of pipelines."""
    def __init__(self):
        """This class represents an "engine" for AutoML, which handles the evaluation of a list of pipelines generated from an AutoML search.

        To use this interface, you must define an `evaluate_batch` method and an `evaluate_pipeline` method.
        """
        self.name = "Base Engine"
        self.X = None
        self.y = None
        self.automl = None

    def load_data(self, X, y):
        """Loads the data to fit the pipeline on. Required to run `_compute_cv_scores`."""
        self.X = X
        self.y = y

    def load_search(self, search_obj):
        """Loads the current AutoML state. This includes information such as the search parameters and objectives, which is required to run `_compute_cv_scores`."""
        self.automl = search_obj

    @abstractmethod
    def evaluate_batch(self, pipeline_batch=None):
        """Evaluate a batch of pipelines using the current dataset and AutoML state.

        The abstract method includes checks to make sure that the dataset and an AutoML search object is loaded into the engine object. It is recommended that any implementation calls `super.evaluate_batch()` once before evaluating pipelines.

        Arguments:
            pipeline_batch (list(PipelineBase)): A batch of pipelines to be fitted and evaluated
        """
        if self.X is None or self.y is None:
            raise ValueError("Dataset has not been loaded into the engine. Call `load_data` with training data.")

        if self.automl is None:
            raise ValueError("Search info has not been loaded into the engine. Call `load_search` with search context.")

    def log_pipeline(self, pipeline):
        desc = f"{pipeline.name}"
        if len(desc) > self.automl._MAX_NAME_LEN:
            desc = desc[:self.automl._MAX_NAME_LEN - 3] + "..."
        desc = desc.ljust(self.automl._MAX_NAME_LEN)

        update_pipeline(logger,
                        desc,
                        len(self.automl._results['pipeline_results']) + 1,
                        self.automl.max_iterations,
                        self.automl._start,
                        1 if self.automl._automl_algorithm.batch_number == 0 else self.automl._automl_algorithm.batch_number,
                        self.automl.show_batch_output)

    def _add_result_callback(self, result_callback, pipeline, evaluation_result):
        """Calls the result callback function with the pipeline evaluation results and updates the search iteration plot.

        Arguments:
            result_callback (callable): Function called after a pipeline is finished evaluation
            pipeline (PipelineBase): An untrained pipeline with the parameters used during training
            evaluation_result (dict): The training results for the pipeline
        """
        parameters = pipeline.parameters
        logger.debug('Adding results for pipeline {}\nparameters {}\nevaluation_results {}'.format(pipeline.name, parameters, evaluation_result))
        result_callback(trained_pipeline=pipeline,
                        parameters=parameters,
                        training_time=evaluation_result['training_time'],
                        cv_data=evaluation_result['cv_data'],
                        cv_scores=evaluation_result['cv_scores'])
        logger.debug('Adding results complete')

        if self.automl.search_iteration_plot:
            self.automl.search_iteration_plot.update()

    def _handle_keyboard_interrupt(self):
        """Presents a prompt to the user asking if they want to stop the search.

        Returns:
            bool: If True, search should terminate early
        """
        leading_char = "\n"
        start_of_loop = time.time()
        while True:
            choice = input(leading_char + "Do you really want to exit search (y/n)? ").strip().lower()
            if choice == "y":
                logger.info("Exiting AutoMLSearch.")
                return True
            elif choice == "n":
                # So that the time in this loop does not count towards the time budget (if set)
                time_in_loop = time.time() - start_of_loop
                self.automl._start += time_in_loop
                return False
            else:
                leading_char = ""

    @staticmethod
    def _compute_cv_scores(pipeline, automl, X, y):
        start = time.time()
        cv_data = []
        logger.info("\tStarting cross validation")
        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)

        X_pd = _convert_woodwork_types_wrapper(X.to_dataframe())
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        for i, (train, test) in enumerate(automl.data_split.split(X_pd, y_pd)):
            if pipeline.model_family == ModelFamily.ENSEMBLE and i > 0:
                # Stacked ensembles do CV internally, so we do not run CV here for performance reasons.
                logger.debug(f"Skipping fold {i} because CV for stacked ensembles is not supported.")
                break
            logger.debug(f"\t\tTraining and scoring on fold {i}")
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            if automl.problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
                diff_train = set(np.setdiff1d(y.to_series(), y_train.to_series()))
                diff_test = set(np.setdiff1d(y.to_series(), y_test.to_series()))
                diff_string = f"Missing target values in the training set after data split: {diff_train}. " if diff_train else ""
                diff_string += f"Missing target values in the test set after data split: {diff_test}." if diff_test else ""
                if diff_string:
                    raise Exception(diff_string)
            objectives_to_score = [automl.objective] + automl.additional_objectives
            cv_pipeline = None
            try:
                X_threshold_tuning = None
                y_threshold_tuning = None
                if automl.optimize_thresholds and automl.objective.is_defined_for_problem_type(ProblemTypes.BINARY) and automl.objective.can_optimize_threshold:
                    X_train, X_threshold_tuning, y_train, y_threshold_tuning = train_test_split(X_train, y_train, test_size=0.2, random_state=automl.random_state)
                cv_pipeline = pipeline.clone(pipeline.random_state)
                logger.debug(f"\t\t\tFold {i}: starting training")
                cv_pipeline.fit(X_train, y_train)
                logger.debug(f"\t\t\tFold {i}: finished training")
                if automl.objective.is_defined_for_problem_type(ProblemTypes.BINARY):
                    cv_pipeline.threshold = 0.5
                    if automl.optimize_thresholds and automl.objective.can_optimize_threshold:
                        logger.debug(f"\t\t\tFold {i}: Optimizing threshold for {automl.objective.name}")
                        y_predict_proba = cv_pipeline.predict_proba(X_threshold_tuning)
                        if isinstance(y_predict_proba, pd.DataFrame):
                            y_predict_proba = y_predict_proba.iloc[:, 1]
                        else:
                            y_predict_proba = y_predict_proba[:, 1]
                        cv_pipeline.threshold = automl.objective.optimize_threshold(y_predict_proba, y_threshold_tuning, X=X_threshold_tuning)
                        logger.debug(f"\t\t\tFold {i}: Optimal threshold found ({cv_pipeline.threshold:.3f})")
                logger.debug(f"\t\t\tFold {i}: Scoring trained pipeline")
                scores = cv_pipeline.score(X_test, y_test, objectives=objectives_to_score)
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
            ordered_scores.update({"# Testing": y_test.shape[0]})

            evaluation_entry = {"all_objective_scores": ordered_scores, "score": score, 'binary_classification_threshold': None}
            if isinstance(cv_pipeline, BinaryClassificationPipeline) and cv_pipeline.threshold is not None:
                evaluation_entry['binary_classification_threshold'] = cv_pipeline.threshold
            cv_data.append(evaluation_entry)
        training_time = time.time() - start
        cv_scores = pd.Series([fold['score'] for fold in cv_data])
        cv_score_mean = cv_scores.mean()
        logger.info(f"\tFinished cross validation - mean {automl.objective.name}: {cv_score_mean:.3f}")
        return pipeline, {'cv_data': cv_data, 'training_time': training_time, 'cv_scores': cv_scores, 'cv_score_mean': cv_score_mean}
