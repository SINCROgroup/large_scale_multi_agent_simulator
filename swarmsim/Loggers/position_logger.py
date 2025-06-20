from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import add_entry, get_positions,append_entry
import numpy as np


class PositionLogger(BaseLogger):
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)

    def log_internal_data(self, save_mode=['csv','npz','mat']):
        if self.step_count % self.save_freq == 0:
            get_positions(self.global_info, self.populations, save_mode)
            append_entry(self.global_info, save_mode, **{"u": np.mean(self.populations[0].u)})
            super().log_internal_data(save_mode)
        
