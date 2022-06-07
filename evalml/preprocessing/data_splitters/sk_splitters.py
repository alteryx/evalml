"""SKLearn data splitter wrapper classes."""
from sklearn.model_selection import KFold, StratifiedKFold


class KFold(KFold):
    """Wrapper class for sklearn's KFold splitter."""

    @property
    def is_cv(self):
        """Returns whether or not the data splitter is a cross-validation data splitter.

        Returns:
            bool: If the splitter is a cross-validation data splitter
        """
        return True


class StratifiedKFold(StratifiedKFold):
    """Wrapper class for sklearn's Stratified KFold splitter."""

    @property
    def is_cv(self):
        """Returns whether or not the data splitter is a cross-validation data splitter.

        Returns:
            bool: If the splitter is a cross-validation data splitter
        """
        return True


class StratifiedSegmentKFold(StratifiedKFold):
    """Wrapper class for sklearn's Stratified KFold splitter."""

    def __init__(self, n_splits=3, random_state=0, shuffle=True, segment=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.segment = segment
        super().__init__(n_splits=self.n_splits, random_state=self.random_state, shuffle=self.shuffle)
    
    def split(self, X, y):
        """Ignores y, creates new y to use based on the segment."""
        y_new = X[self.segment]
        X = X.drop(self.segment, axis=1)
        return super().split(X, y_new)

    @property
    def is_cv(self):
        """Returns whether or not the data splitter is a cross-validation data splitter.

        Returns:
            bool: If the splitter is a cross-validation data splitter
        """
        return True
