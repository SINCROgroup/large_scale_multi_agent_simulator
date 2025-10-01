from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import add_entry, get_positions,append_entry
import numpy as np


class PositionLogger(BaseLogger):
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)

    def log_step_data(self):
        # Append positions of all agents at every step
        get_positions(self.step_data, self.populations,
                      save_modes=('csv', 'npz', 'mat'))

    def log(self, data: dict | None = None, end_experiment: bool = False):
        super().log(data, end_experiment)
