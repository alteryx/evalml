from abc import abstractmethod, ABCMeta
import pytest

from evalml import AutoMLSearch

class TestBinaryObjective(metaclass=ABCMeta):
    __test__ = False

    def assign_problem_type(self):
        self.problem_type = 'binary'

    @abstractmethod
    def assign_objective(self, **kwargs):
        """Get objective object using specified parameters
        """

    def run_pipeline(self, X_y_binary, **kwargs):
        self.X, self.y = X_y_binary
        automl = AutoMLSearch(X_train=self.X, y_train=self.y, problem_type=self.problem_type, objective=self.objective, max_iterations=1)
        automl.search()

        pipeline = automl.best_pipeline
        pipeline.fit(self.X, self.y)
        pipeline.predict(self.X, self.objective)
        pipeline.predict_proba(self.X)
        pipeline.score(self.X, self.y, [self.objective])

    # def test_input_contains_nan():

    # def test_input_contains_inf():

    # def test_input_lengths():

    # def test_zero_input_lengths():

    # def test_binary_more_than_two_unique_values():

    # def test_objective_score():