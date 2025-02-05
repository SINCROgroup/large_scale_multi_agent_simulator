from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import add_entry
import numpy as np


class PositionLogger(BaseLogger):
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)

    def log(self, data: dict = None):
        self.current_info = {}
        self.done = False

        if self.activate:
            # Include desired information
            add_entry(self.current_info, step=self.step_count)  # Get timestamp
            for pop in self.populations:
                for d in range(pop.state_dim):
                    col_name = str(pop.id + str(d))
                    value = pop.x[:, d]
                    add_entry(self.current_info, **{col_name: value})
            if data is not None:
                for key, value in data.items():
                    add_entry(self.current_info, **{key: value})

            # Print line if wanted
            if self.log_freq > 0:
                if self.step_count % self.log_freq == 0:
                    self.print_log()

            # Save line if wanted
            if self.save_freq > 0:
                if self.step_count % self.save_freq == 0:
                    self.save()

            self.step_count += 1  # Update step counter

        return self.done
