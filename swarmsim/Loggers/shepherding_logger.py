from swarmsim.Loggers import BaseLogger
from swarmsim.Utils import add_entry, get_done_shepherding, xi_shepherding


class ShepherdingLogger(BaseLogger):
    """
    Specialized logger for shepherding simulations with task-specific metrics.

    This logger extends the BaseLogger to capture and analyze shepherding-specific
    metrics such as target capture rates, completion status, and task progression.
    It monitors the effectiveness of shepherding algorithms by tracking how many
    targets are successfully guided to goal regions.

    The logger computes the shepherding metric xi (ξ), which represents the fraction
    of targets successfully captured, and monitors task completion to enable early
    termination when all targets reach the goal region.

    Parameters
    ----------
    populations : list of Population
        List of population objects in the shepherding simulation.
        Typically includes target agents (index 0) and optionally shepherd agents.
    environment : Environment
        Shepherding environment object containing goal regions and spatial boundaries.
        Must support shepherding-specific geometric calculations.
    config_path : str
        Path to the YAML configuration file containing logger parameters.

    Attributes
    ----------
    populations : list of Population
        Population objects being monitored, inherited from BaseLogger.
        First population (index 0) is typically the target agents.
    environment : Environment
        Shepherding environment with goal regions, inherited from BaseLogger.
    xi : float
        Current shepherding metric (fraction of captured targets).
        Range: [0.0, 1.0] where 1.0 indicates all targets captured.
    done : bool
        Task completion flag indicating whether all targets are captured.
    current_info : dict
        Current timestep information including shepherding metrics.
    global_info : dict
        Accumulated simulation data across all timesteps.

    Config Requirements
    -------------------
    The YAML configuration file must contain logger parameters under the class section:

    ShepherdingLogger : dict
        Configuration section for the shepherding logger:
        - ``activate`` : bool, optional
            Enable/disable logging. Default: ``True``
        - ``log_freq`` : int, optional
            Print frequency (0 = never). Default: ``0``
        - ``save_freq`` : int, optional
            Save frequency (0 = never). Default: ``1``
        - ``save_data_freq`` : int, optional
            Data save frequency. Default: ``0``
        - ``save_global_data_freq`` : int, optional
            Global data save frequency. Default: ``0``
        - ``log_path`` : str, optional
            Output directory path. Default: ``"./logs"``
        - ``log_name`` : str, optional
            Log file name suffix. Default: ``""``
        - ``comment_enable`` : bool, optional
            Enable experiment comments. Default: ``False``

    Notes
    -----
    **Shepherding Metrics:**

    The logger computes several task-specific metrics:

    - **Xi (ξ)**: Fraction of targets successfully captured in goal region
    - **Task Completion**: Boolean flag indicating complete task success
    - **Temporal Progression**: Evolution of capture rate over time

    **Early Termination:**

    The logger can trigger early simulation termination when all targets
    are successfully shepherded to the goal region, improving computational
    efficiency for successful trials.

    **Integration with Shepherding Utils:**

    Uses specialized utility functions:

    - ``xi_shepherding()``: Computes capture fraction metric
    - ``get_done_shepherding()``: Determines task completion status

    Examples
    --------
    **Basic Configuration:**

    .. code-block:: yaml

        ShepherdingLogger:
            activate: true
            log_freq: 10
            save_freq: 1
            log_path: "./shepherding_logs"
            log_name: "shepherding_experiment"

    """
    def __init__(self, populations: list, environment: object, config_path: str) -> None:
        super().__init__(populations, environment, config_path)
        self.xi = 0

    def log(self, data: dict = None) -> bool:
        """
        A function that defines the information to log.

        Parameters
        ----------
            data: dict  composed of {name_variabile: value} to log

        Returns
        -------
            done: bool flag to truncate a simulation early. Default value=False.

        Notes
        -------
            In the configuration file (a yaml file) there should be a namespace with the name of the log you are creating.
            By default, it does not truncate episode early.
            See add_data from Utils/logger_utils.py to quickly add variables to log.

        """
        if self.activate:
            # Get metrics
            self.xi = self.get_xi()
            self.done = self.get_event()
            super().log(data)

        return self.done

    def log_internal_data(self, save_mode=['print', 'txt']):
        super().log_internal_data(save_mode)
        add_entry(self.current_info, save_mode, xi=self.xi)

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
