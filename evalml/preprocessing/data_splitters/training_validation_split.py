"""Training Validation Split class."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import BaseCrossValidator


class TrainingValidationSplit(BaseCrossValidator):
    """Split the training data into training and validation sets.

    Args:
        test_size (float): What percentage of data points should be included in the validation
            set. Defalts to the complement of `train_size` if `train_size` is set, and 0.25 otherwise.
        train_size (float): What percentage of data points should be included in the training set.
            Defaults to the complement of `test_size`
        shuffle (boolean): Whether to shuffle the data before splitting. Defaults to False.
        stratify (list): Splits the data in a stratified fashion, using this argument as class labels.
            Defaults to None.
        random_seed (int): The seed to use for random sampling. Defaults to 0.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        ...
        >>> X = pd.DataFrame([i for i in range(10)], columns=["First"])
        >>> y = pd.Series([i for i in range(10)])
        ...
        >>> tv_split = TrainingValidationSplit()
        >>> split_ = next(tv_split.split(X, y))
        >>> assert (split_[0] == np.array([0, 1, 2, 3, 4, 5, 6])).all()
        >>> assert (split_[1] == np.array([7, 8, 9])).all()
        ...
        ...
        >>> tv_split = TrainingValidationSplit(test_size=0.5)
        >>> split_ = next(tv_split.split(X, y))
        >>> assert (split_[0] == np.array([0, 1, 2, 3, 4])).all()
        >>> assert (split_[1] == np.array([5, 6, 7, 8, 9])).all()
        ...
        ...
        >>> tv_split = TrainingValidationSplit(shuffle=True)
        >>> split_ = next(tv_split.split(X, y))
        >>> assert (split_[0] == np.array([9, 1, 6, 7, 3, 0, 5])).all()
        >>> assert (split_[1] == np.array([2, 8, 4])).all()
        ...
        ...
        >>> y = pd.Series([i % 3 for i in range(10)])
        >>> tv_split = TrainingValidationSplit(shuffle=True, stratify=y)
        >>> split_ = next(tv_split.split(X, y))
        >>> assert (split_[0] == np.array([1, 9, 3, 2, 8, 6, 7])).all()
        >>> assert (split_[1] == np.array([0, 4, 5])).all()
    """

    def __init__(
        self,
        test_size=None,
        train_size=None,
        shuffle=False,
        stratify=None,
        random_seed=0,
    ):
        self.test_size = test_size
        self.train_size = train_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.random_seed = random_seed

    @staticmethod
    def get_n_splits():
        """Return the number of splits of this object.

        Returns:
            int: Always returns 1.
        """
        return 1

    @property
    def is_cv(self):
        """Returns whether or not the data splitter is a cross-validation data splitter.

        Returns:
            bool: If the splitter is a cross-validation data splitter
        """
        return False

    def split(self, X, y=None):
        """Divide the data into training and testing sets.

        Args:
            X (pd.DataFrame): Dataframe of points to split
            y (pd.Series): Series of points to split

        Returns:
            list: Indices to split data into training and test set
        """
        train, test = train_test_split(
            np.arange(X.shape[0]),
            test_size=self.test_size,
            train_size=self.train_size,
            shuffle=self.shuffle,
            stratify=self.stratify,
            random_state=self.random_seed,
        )
        return iter([(train, test)])
