from abc import ABC, abstractmethod
import numpy as np


class Logger(ABC):
    """
    Abstract base class for a logger.

    """

    def __init__(self) -> None:
        super().__init__()

    # This method
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def log(self, data: dict) -> bool:
        """
        A function that defines the information to log.

        Parameters
        ----------
            data: dict (composed of {name_variabile: value} to log)

        Returns
        -------
            done: bool (flag to truncate a simulation early). Default value=False.

        Notes
        -------
            In the configuration file (a yaml file) there should be a namespace with the name of the log you are creating.
            By default, it does not truncate episode early.

        """
        done = False
        return done

    @abstractmethod
    def close(self):
        pass
