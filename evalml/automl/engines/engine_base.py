from abc import ABC, abstractmethod
from evalml.utils.logger import get_logger, update_pipeline

# from evalml.problem_types import ProblemTypes
# import time
# import pandas as pd
# import numpy as np
# from collections import OrderedDict
# from evalml.exceptions import PipelineScoreError
# from evalml.pipelines import BinaryClassificationPipeline
# from sklearn.model_selection import train_test_split

logger = get_logger(__file__)


class EngineBase(ABC):
    def __init__(self):
        self.name = "Base Engine"

    def load_data(self, X, y, search):
        self.X = X
        self.y = y
        self.search = search

    @abstractmethod
    def evaluate_pipeline(self, pipeline):
        ""
        ""

    def log_pipeline(self, pipeline):
        desc = f"{pipeline.name}"
        if len(desc) > self.search._MAX_NAME_LEN:
            desc = desc[:self.search._MAX_NAME_LEN - 3] + "..."
        desc = desc.ljust(self.search._MAX_NAME_LEN)

        update_pipeline(logger, desc, len(self.search._results['pipeline_results']) + 1, self.search.max_iterations, self.search._start)

    # def _evaluate(self, pipeline, X, y):
    #     evaluation_results = self._compute_cv_scores(pipeline, X, y)
    #     return pipeline, evaluation_results

    # def _compute_cv_scores(self, pipeline, X, y):
    #     start = time.time()
    #     cv_data = []
    #     self.search.logger.info("\tStarting cross validation")
    #     for i, (train, test) in enumerate(self.search.data_split.split(X, y)):
    #         self.search.logger.debug(f"\t\tTraining and scoring on fold {i}")
    #         X_train, X_test = X.iloc[train], X.iloc[test]
    #         y_train, y_test = y.iloc[train], y.iloc[test]
    #         if self.search.problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
    #             diff_train = set(np.setdiff1d(y, y_train))
    #             diff_test = set(np.setdiff1d(y, y_test))
    #             diff_string = f"Missing target values in the training set after data split: {diff_train}. " if diff_train else ""
    #             diff_string += f"Missing target values in the test set after data split: {diff_test}." if diff_test else ""
    #             if diff_string:
    #                 raise Exception(diff_string)
    #         objectives_to_score = [self.search.objective] + self.search.additional_objectives
    #         cv_pipeline = None
    #         try:
    #             X_threshold_tuning = None
    #             y_threshold_tuning = None
    #             if self.search.optimize_thresholds and self.search.objective.problem_type == ProblemTypes.BINARY and self.search.objective.can_optimize_threshold:
    #                 X_train, X_threshold_tuning, y_train, y_threshold_tuning = train_test_split(X_train, y_train, test_size=0.2, random_state=self.search.random_state)
    #             cv_pipeline = pipeline.clone()
    #             self.search.logger.debug(f"\t\t\tFold {i}: starting training")
    #             cv_pipeline.fit(X_train, y_train)
    #             self.search.logger.debug(f"\t\t\tFold {i}: finished training")
    #             if self.search.objective.problem_type == ProblemTypes.BINARY:
    #                 cv_pipeline.threshold = 0.5
    #                 if self.search.optimize_thresholds and self.search.objective.can_optimize_threshold:
    #                     self.search.logger.debug(f"\t\t\tFold {i}: Optimizing threshold for {self.search.objective.name}")
    #                     y_predict_proba = cv_pipeline.predict_proba(X_threshold_tuning)
    #                     if isinstance(y_predict_proba, pd.DataFrame):
    #                         y_predict_proba = y_predict_proba.iloc[:, 1]
    #                     else:
    #                         y_predict_proba = y_predict_proba[:, 1]
    #                     cv_pipeline.threshold = self.search.objective.optimize_threshold(y_predict_proba, y_threshold_tuning, X=X_threshold_tuning)
    #                     self.search.logger.debug(f"\t\t\tFold {i}: Optimal threshold found ({cv_pipeline.threshold:.3f})")
    #             self.search.logger.debug(f"\t\t\tFold {i}: Scoring trained pipeline")
    #             scores = cv_pipeline.score(X_test, y_test, objectives=objectives_to_score)
    #             self.search.logger.debug(f"\t\t\tFold {i}: {self.search.objective.name} score: {scores[self.search.objective.name]:.3f}")
    #             score = scores[self.search.objective.name]
    #         except Exception as e:
    #             if isinstance(e, PipelineScoreError):
    #                 self.search.logger.info(f"\t\t\tFold {i}: Encountered an error scoring the following objectives: {', '.join(e.exceptions)}.")
    #                 self.search.logger.info(f"\t\t\tFold {i}: The scores for these objectives will be replaced with nan.")
    #                 self.search.logger.info(f"\t\t\tFold {i}: Please check {self.search.logger.handlers[1].baseFilename} for the current hyperparameters and stack trace.")
    #                 self.search.logger.debug(f"\t\t\tFold {i}: Hyperparameters:\n\t{pipeline.hyperparameters}")
    #                 self.search.logger.debug(f"\t\t\tFold {i}: Exception during automl search: {str(e)}")
    #                 nan_scores = {objective: np.nan for objective in e.exceptions}
    #                 scores = {**nan_scores, **e.scored_successfully}
    #                 scores = OrderedDict({o.name: scores[o.name] for o in [self.search.objective] + self.search.additional_objectives})
    #                 score = scores[self.search.objective.name]
    #             else:
    #                 self.search.logger.info(f"\t\t\tFold {i}: Encountered an error.")
    #                 self.search.logger.info(f"\t\t\tFold {i}: All scores will be replaced with nan.")
    #                 self.search.logger.info(f"\t\t\tFold {i}: Please check {self.search.logger.handlers[1].baseFilename} for the current hyperparameters and stack trace.")
    #                 self.search.logger.debug(f"\t\t\tFold {i}: Hyperparameters:\n\t{pipeline.hyperparameters}")
    #                 self.search.logger.debug(f"\t\t\tFold {i}: Exception during automl search: {str(e)}")
    #                 score = np.nan
    #                 scores = OrderedDict(zip([n.name for n in self.search.additional_objectives], [np.nan] * len(self.search.additional_objectives)))

    #         ordered_scores = OrderedDict()
    #         ordered_scores.update({self.search.objective.name: score})
    #         ordered_scores.update(scores)
    #         ordered_scores.update({"# Training": len(y_train)})
    #         ordered_scores.update({"# Testing": len(y_test)})

    #         evaluation_entry = {"all_objective_scores": ordered_scores, "score": score, 'binary_classification_threshold': None}
    #         if isinstance(cv_pipeline, BinaryClassificationPipeline) and cv_pipeline.threshold is not None:
    #             evaluation_entry['binary_classification_threshold'] = cv_pipeline.threshold
    #         cv_data.append(evaluation_entry)

    #     training_time = time.time() - start
    #     cv_scores = pd.Series([fold['score'] for fold in cv_data])
    #     cv_score_mean = cv_scores.mean()
    #     self.search.logger.info(f"\tFinished cross validation - mean {self.search.objective.name}: {cv_score_mean:.3f}")
    #     return {'cv_data': cv_data, 'training_time': training_time, 'cv_scores': cv_scores, 'cv_score_mean': cv_score_mean}
