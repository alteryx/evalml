import logging
import time

from evalml.utils.logger import get_logger


class Progress:
    def __init__(
        self,
        max_time=None,
        max_batches=None,
        max_iterations=None,
        patience=None,
        tolerance=None,
        automl_algorithm=None,
        objective=None,
        verbose=False,
    ):
        self.max_time = max_time
        self.current_time = None
        self._start_time = None
        self.max_batches = max_batches
        self.current_batches = 0
        self.max_iterations = max_iterations
        self.current_iterations = 0
        self.patience = patience
        self.tolerance = tolerance
        self.automl_algorithm = automl_algorithm
        self.objective = objective

        if verbose:
            self.logger = get_logger(f"{__name__}.verbose")
        else:
            self.logger = logging.getLogger(__name__)

    def should_continue(self, results, interrupted=False, mid_batch=False):
        """Given AutoML Results, return whether or not the search should continue.

        Returns:
            bool: True if search should continue, False otherwise.
        """
        if interrupted:
            return False
        # update and check max_time, max_iterations, and max_batches
        self.current_time = time.time()
        self.current_iterations = len(results["pipeline_results"])
        self.current_batches = self.automl_algorithm.batch_number

        elapsed = self.current_time - self._start_time

        if self.max_time and elapsed >= self.max_time:
            return False
        elif self.max_iterations and self.current_iterations >= self.max_iterations:
            return False
        elif (
            self.max_batches
            and self.current_batches >= self.max_batches
            and not mid_batch
        ):
            return False

        # check for early stopping
        if self.patience is None or self.tolerance is None:
            return True

        first_id = results["search_order"][0]
        best_score = results["pipeline_results"][first_id]["mean_cv_score"]
        num_without_improvement = 0
        for id in results["search_order"][1:]:
            curr_score = results["pipeline_results"][id]["mean_cv_score"]
            significant_change = (
                abs((curr_score - best_score) / best_score) > self.tolerance
            )
            score_improved = (
                curr_score > best_score
                if self.objective.greater_is_better
                else curr_score < best_score
            )
            if score_improved and significant_change:
                best_score = curr_score
                num_without_improvement = 0
            else:
                num_without_improvement += 1
            if num_without_improvement >= self.patience:
                self.logger.info(
                    "\n\n{} iterations without improvement. Stopping search early...".format(
                        self.patience,
                    ),
                )
                return False
        return True

    def _build_progress_dict(self, stopping_criteria, current_state, end_state, unit):
        progress_dict = {}
        progress_dict["stopping_criteria"] = stopping_criteria
        progress_dict["current_state"] = current_state
        progress_dict["end_state"] = end_state
        progress_dict["unit"] = unit
        return progress_dict

    def return_progress(self):
        """Return information about current and end state of each stopping criteria in order of priority.

        Returns:
            List[Dict[str, unit]]: list of dictionaries containing information of each stopping criteria.
        """
        progress = []
        if self.max_time:
            progress.append(
                self._build_progress_dict(
                    "max_time",
                    self.current_time - self._start_time,
                    self.max_time,
                    "seconds",
                ),
            )
        if self.max_iterations or self.max_batches:
            max_iterations = (
                self.max_iterations
                if self.max_iterations
                else sum(
                    [
                        self.automl_algorithm.num_pipelines_per_batch(n)
                        for n in range(self.max_batches)
                    ],
                )
            )
            progress.append(
                self._build_progress_dict(
                    "max_iterations",
                    self.current_iterations,
                    max_iterations,
                    "iterations",
                ),
            )
        if self.max_batches:
            progress.append(
                self._build_progress_dict(
                    "max_batches",
                    self.current_batches,
                    self.max_batches,
                    "batches",
                ),
            )
        return progress
