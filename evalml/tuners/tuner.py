from abc import ABC, abstractmethod


class Tuner(ABC):
    """Defines API for Tuners

    Tuners implement different strategies for sampling from a search space. They're used in EvalML to search the space of pipeline hyperparameters.
    """

    def __init__(self, space, random_state=0):
        """Init Tuner

        Arguments:
            space (dict): search space for hyperparameters
            random_state (int, np.random.RandomState): The random state

        Returns:
            Tuner: self
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, parameters, score):
        """ Register a set of hyperparameters with the score obtained from training a pipeline with those hyperparameters.

        Arguments:
            parameters (dict): hyperparameters
            score (float): associated score

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def propose(self):
        """ Returns a set of hyperparameters to train a pipeline with, based off the search space dimensions and prior samples

        Returns:
            dict: proposed hyperparameters
        """
        raise NotImplementedError
