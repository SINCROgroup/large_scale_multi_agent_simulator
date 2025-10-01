from typing import Tuple, Any

from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import get_done_shepherding, xi_shepherding, get_positions


class ShepherdingLogger(BaseLogger):
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)
        self.xi = 0
        self.get_position = self.logger_config.get('get_position', False)

    def log(self, data: dict | None = None, end_experiment: bool = False):
        self.xi = self.get_xi()
        super().log(data=data, end_experiment=end_experiment)

    def log_step_data(self):
        self._add_entry(('csv', 'print', 'mat', 'npz'), step=True, xi=self.xi)
        if self.get_position:
            get_positions(self.step_data, self.populations,
                          save_modes=('csv', 'npz', 'mat'))
        super().log_step_data()

    def log_experiment_data(self):
        self._add_entry(('print',), step=False, episode=self.exp_idx)
        super().log_experiment_data()

    def get_xi(self) -> float:
        """
        Get metric for shepherding xi, i.e., fraction of captured targets.
        Returns
        -------
            float: fraction of captured targets

        """
        return xi_shepherding(self.populations[0], self.environment)

    def get_event(self) -> bool:
        """
        Verify if every target is inside the goal region
        Returns
        -------
            bool: true is every target is inside the goal region, false otherwise
        """
        return get_done_shepherding(self.populations[0], self.environment)
