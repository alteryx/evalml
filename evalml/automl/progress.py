"""Progress abstraction holding stopping criteria and progress information."""
import logging
import time

from evalml.utils.logger import get_logger


class Progress:
    """Progress object holding stopping criteria and progress information.

    Args:
        max_time (int): Maximum time to search for pipelines.
        max_iterations (int): Maximum number of iterations to search.
        max_batches (int): The maximum number of batches of pipelines to search. Parameters max_time, and
            max_iterations have precedence over stopping the search.
        patience (int): Number of iterations without improvement to stop search early.
        tolerance (float): Minimum percentage difference to qualify as score improvement for early stopping.
        automl_algorithm (str): The automl algorithm to use. Used to calculate iterations if max_batches is selected as stopping criteria.
        objective (str, ObjectiveBase): The objective used in search.
        verbose (boolean): Whether or not to log out stopping information.
    """

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
        self.start_time = None
        self.max_batches = max_batches
        self.current_batch = 0
        self.max_iterations = max_iterations
        self.current_iterations = 0
        self.patience = patience
        self.tolerance = tolerance
        self.automl_algorithm = automl_algorithm
        self.objective = objective
        self._best_score = None
        self._without_improvement = 0
        self._last_id = 0

        if verbose:
            self.logger = get_logger(f"{__name__}.verbose")
        else:
            self.logger = logging.getLogger(__name__)

    def start_timing(self):
        """Sets start time to current time."""
        self.start_time = time.time()

    def elapsed(self):
        """Return time elapsed using the start time and current time."""
        return self.current_time - self.start_time

    def should_continue(self, results, interrupted=False, mid_batch=False):
        """Given AutoML Results, return whether or not the search should continue.

        Args:
            results (dict): AutoMLSearch results.
            interrupted (bool): whether AutoMLSearch was given an keyboard interrupt. Defaults to False.
            mid_batch (bool): whether this method was called while in the middle of a batch or not. Defaults to False.

        Returns:
            bool: True if search should continue, False otherwise.
        """
        if interrupted:
            return False
        # update and check max_time, max_iterations, and max_batches
        self.current_time = time.time()
        self.current_iterations = len(results["pipeline_results"])
        self.current_batch = self.automl_algorithm.batch_number

        if self.max_time and self.elapsed() >= self.max_time:
            return False
        elif self.max_iterations and self.current_iterations >= self.max_iterations:
            return False
        elif (
            self.max_batches
            and self.current_batch >= self.max_batches
            and not mid_batch
        ):
            return False

        # check for early stopping
        if self.patience is not None and self.tolerance is not None:
            last_id = results["search_order"][-1]
            curr_score = results["pipeline_results"][last_id]["mean_cv_score"]
            if self._best_score is None:
                self._best_score = curr_score
                return True
            elif last_id > self._last_id:
                self._last_id = last_id
                score_improved = (
                    curr_score > self._best_score
                    if self.objective.greater_is_better
                    else curr_score < self._best_score
                )
                significant_change = (
                    abs((curr_score - self._best_score) / self._best_score)
                    > self.tolerance
                )
                if score_improved and significant_change:
                    self._best_score = curr_score
                    self._without_improvement = 0
                else:
                    self._without_improvement += 1
                if self._without_improvement >= self.patience:
                    self.logger.info(
                        "\n\n{} iterations without improvement. Stopping search early...".format(
                            self.patience,
                        ),
                    )
                    return False
        return True

    def return_progress(self):
        """Return information about current and end state of each stopping criteria in order of priority.

        Returns:
            List[Dict[str, unit]]: list of dictionaries containing information of each stopping criteria.
        """
        progress = []
        if self.max_time:
            progress.append(
                {
                    "stopping_criteria": "max_time",
                    "current_state": self.elapsed(),
                    "end_state": self.max_time,
                    "unit": "seconds",
                },
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
                {
                    "stopping_criteria": "max_iterations",
                    "current_state": self.current_iterations,
                    "end_state": max_iterations,
                    "unit": "iterations",
                },
            )
        if self.max_batches:
            progress.append(
                {
                    "stopping_criteria": "max_batches",
                    "current_state": self.current_batch,
                    "end_state": self.max_batches,
                    "unit": "batches",
                },
            )
        return progress
