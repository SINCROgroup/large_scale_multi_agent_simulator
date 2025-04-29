from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import add_entry, get_positions
import numpy as np


class PositionLogger(BaseLogger):
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)

    def log_internal_data(self, save_mode=['mat']):
        super().log_internal_data(save_mode)
        get_positions(self.global_info, self.populations, save_mode)
