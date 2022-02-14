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
