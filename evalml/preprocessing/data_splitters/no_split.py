"""Empty Data Splitter class."""
import numpy as np
from sklearn.model_selection._split import BaseCrossValidator


class NoSplit(BaseCrossValidator):
    """Does not split the training data into training and validation sets.

    All data is passed as the training set, test data is simply an array of
    `None`. To be used for future unsupervised learning, should not be used
    in any of the currently supported pipelines.

    Args:
        random_seed (int): The seed to use for random sampling. Defaults to 0. Not used.
    """

    def __init__(
        self,
        random_seed=0,
    ):
        self.random_seed = random_seed

    @staticmethod
    def get_n_splits():
        """Return the number of splits of this object.

        Returns:
            int: Always returns 0.
        """
        return 0

    @property
    def is_cv(self):
        """Returns whether or not the data splitter is a cross-validation data splitter.

        Returns:
            bool: If the splitter is a cross-validation data splitter
        """
        return False

    def split(self, X, y=None):
        """Divide the data into training and testing sets, where the testing set is empty.

        Args:
            X (pd.DataFrame): Dataframe of points to split
            y (pd.Series): Series of points to split

        Returns:
            list: Indices to split data into training and test set
        """
        return iter([(np.arange(X.shape[0]), [])])
