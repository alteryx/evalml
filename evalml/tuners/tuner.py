from abc import ABC, abstractmethod


class Tuner(ABC):
    """Base Tuner class"""

    def __init__(self, space, random_state=0):
        """Init Tuner

        Arguments:
            space (dict): search space for hyperparameters
            random_state (int): random state

        Returns:
            Tuner: self
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, parameters, score):
        """ Add score to sample

        Arguments:
            parameters (dict): hyperparameters
            score (float): associated score

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def propose(self):
        """ Returns hyperparameters based off search space and samples

        Returns:
            dict: proposed hyperparameters
        """
        raise NotImplementedError
