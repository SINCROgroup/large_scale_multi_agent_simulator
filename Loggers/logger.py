from abc import ABC, abstractmethod
import numpy as np


class Logger(ABC):
    """
    An interface that defines all the methods that a logger should implement
    Arguments
    -------
    name: string containing the name of the logger
    done: flag for stopping the episode early
    activate: flag to save logger
    log_freq: print information every log_freq steps
    save_freq: save information every save_freq steps

    Methods
    -------
    reset(self):
        Creates files
    log(self, populations, env):
        Returns all the information to log.
    close(self, populations, env):
        Close logger
    """

    def __init__(self) -> None:
        super().__init__()

    # This method
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def log(self):
        return False  # By default do not truncate episode early

    @abstractmethod
    def close(self):
        pass
