from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import add_entry
import numpy as np


class PositionLogger(BaseLogger):
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)

    def log_internal_data(self, print_flag=False, txt_flag=False, csv_flag=False, npz_flag=False, mat_flag=True):
        super().log_internal_data(print_flag, txt_flag, csv_flag, npz_flag, mat_flag)
        for pop in self.populations:
            for d in range(pop.state_dim):
                col_name = str(pop.id + '_state' + str(d))
                value = pop.x[:, d]
                add_entry(self.current_info, print_flag, txt_flag, csv_flag, npz_flag, mat_flag, **{col_name: value})
