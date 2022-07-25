import time


class Progress:
    def __init__(
        self,
        max_time=None,
        max_batches=None,
        max_iterations=None,
        patience=None,
        tolerance=None,
        automl_algorithm=None,
    ):
        # store all stopping criteria as well as store current state
        self.max_time = max_time
        self.current_time = None
        self.max_batches = max_batches
        self.current_batches = 0
        self.max_iterations = max_iterations
        self.current_iterations = 0
        self.patience = patience
        self.tolerance = tolerance
        self.automl_algorithm = automl_algorithm

    def should_continue(self, results, interrupted=False):
        """Given AutoML Results, return whether or not the search should continue.

        Returns:
            bool: True if search should continue, False otherwise.
        """
        if interrupted:
            return False

        # check max_time, max_iterations, and max_batches
        self.current_time = time.time()
        elapsed = self.current_time - self._start
        if self.max_time and elapsed >= self.max_time:
            return False
        elif self.max_iterations and self.current_iterations >= self.max_iterations:
            return False
        elif self.max_batches and self.current_batches > self.max_batches:
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

    def return_progress():
        # returns list of dictionaries housing all held criteria and their current state
        # in order of importance .. time > iterations >= batches
        # does not return early stopping information
        # if max_batches but not max_iterations return conversion as well by
        # using automl_algorithm.num_pipelines_per_batch
        #   returns [{
        #     "stopping_criteria": "max_time",
        #     "current_state": 2mins,
        #     "end_State": 15mins,
        #     "unit": time
        #   }]
        pass
