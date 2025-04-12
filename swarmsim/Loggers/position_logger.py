from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import add_entry
import numpy as np


class PositionLogger(BaseLogger):
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)

    def log_internal_data(self, print_flag=True, txt_flag=True, csv_flag=True, npz_flag=False):
        super().log_internal_data(print_flag, txt_flag, csv_flag, False)
        for pop in self.populations:
            for d in range(pop.state_dim):
                col_name = str(pop.id + str(d))
                value = pop.x[:, d]
                add_entry(self.current_info, print_flag, txt_flag, csv_flag, True, **{col_name: value})
