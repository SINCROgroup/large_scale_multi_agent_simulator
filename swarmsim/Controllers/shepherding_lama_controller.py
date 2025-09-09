from swarmsim.Controllers import Controller
import numpy as np
from typing import cast

from swarmsim.Utils import compute_distances
from swarmsim.Environments import ShepherdingEnvironment
from swarmsim.Populations import Population


class ShepherdingLamaController(Controller):
    """
    Implementation of the shepherding controller implemented in Lama (2024).

    This controller implements the herding control law from Lama (2024) for
    shepherding applications. It coordinates herder agents to guide target agents toward
    a goal region by positioning herders behind the most distant targets and applying
    repulsive forces to drive the targets toward the goal.

    The controller selects the farthest target from the goal within each herder's sensing
    radius and positions the herder behind that target at a specified distance. This
    creates a shepherding behavior where herders push targets toward the goal region.

    Parameters
    ----------
    population : Population
        The herder population that will be controlled by this controller.
    targets : Population
        The target population that needs to be shepherded to the goal.
    environment : ShepherdingEnvironment, optional
        The shepherding environment containing goal information. Default is None.
    config_path : str, optional
        Path to the YAML configuration file containing controller parameters. Default is None.

    Attributes
    ----------
    herders : Population
        Reference to the herder population (same as population).
    targets : Population
        The target population being shepherded.
    environment : ShepherdingEnvironment
        The shepherding environment with goal position information.
    xi : float
        Sensing radius of herder agents for target detection.
    v_h : float
        Speed of herders when no targets are detected within sensing radius.
    alpha : float
        Attraction force constant toward the selected target position.
    lmbda : float
        Lambda parameter (currently not used in implementation).
    delta : float
        Displacement distance to position herders behind targets.
    rho_g : float
        Radius of the goal region.

    Config Requirements
    -------------------
    The YAML configuration file must contain the following parameters under the controller's section:

    dt : float
        Sampling time of the controller (time interval between control actions).
    xi : float, optional
        Sensing radius of herder agents. Default is ``15``.
    v_h : float, optional
        Speed of herders when no targets are detected. Default is ``12``.
    alpha : float, optional
        Attraction force constant to the selected target. Default is ``3``.
    lambda : float, optional
        Lambda parameter (not currently used). Default is ``3``.
    delta : float, optional
        Displacement to go behind a target. Default is ``1.5``.
    rho_g : float, optional
        Radius of the goal region. Default is ``5``.

    Notes
    -----
    The controller algorithm works as follows:

    1. **Target Selection**: Each herder identifies targets within its sensing radius `xi`
    2. **Distance Calculation**: Among detected targets, select the one farthest from the goal
    3. **Position Calculation**: Position herder behind the selected target at distance `delta`
    4. **Control Action**: Apply attractive force with strength `alpha` toward the desired position
    5. **No Target Behavior**: If no targets detected, move away from goal with speed `v_h`

    The controller assumes a shepherding environment with a defined goal position and
    requires both herder and target populations to be properly initialized.

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        ShepherdingLamaController:
            dt: 0.1
            xi: 15
            v_h: 12
            alpha: 3
            lambda: 3
            delta: 1.5
            rho_g: 5

    This defines a `ShepherdingLamaController` with xi=15, v_h=12, alpha=3, lambda=3, delta=1.5, rho_g=5.
    This controller is able to steer a population of targets to the goal region.

    References
    ----------
    Lama, Andrea, and Mario di Bernardo. "Shepherding and herdability in complex multiagent systems." Physical Review Research 6.3 (2024): L032012

    """

    def __init__(self, population: Population,
                 targets: Population,
                 environment: ShepherdingEnvironment =None,
                 config_path: str =None) -> None:
        """
        Initialize the LAMA shepherding controller.

        Parameters
        ----------
        population : Population
            The herder population that will be controlled.
        targets : Population
            The target population to be shepherded.
        environment : ShepherdingEnvironment, optional
            The shepherding environment containing goal information. Default is None.
        config_path : str, optional
            Path to the configuration file. Default is None.

        Raises
        ------
        TypeError
            If environment is not a ShepherdingEnvironment instance.
        """

        super().__init__(population, environment, config_path, [targets])
        self.herders: Population = self.population
        self.targets: Population = targets
        self.environment = cast(ShepherdingEnvironment, self.environment)

        self.xi: float = self.config.get('xi', 15)
        self.v_h: float = self.config.get('v_h', 12)
        self.alpha: float = self.config.get('alpha', 3)
        self.lmbda: float = self.config.get('lambda', 3)
        self.delta: float = self.config.get('delta', 1.5)
        self.rho_g: float = self.config.get('rho_g', 5)

    def get_action(self) -> np.ndarray:
        """
        Compute the shepherding control action for herder agents.

        This method implements the LAMA shepherding algorithm by:
        1. Finding targets within each herder's sensing radius
        2. Selecting the target farthest from the goal for each herder
        3. Computing desired herder positions behind selected targets
        4. Apply attractive force toward desired herder positions

        Returns
        -------
        np.ndarray
            Control actions of shape (N_herders, 2) representing force vectors
            for each herder agent in the x and y directions.

        Notes
        -----
        The algorithm follows these steps:

        1. **Target Detection**: For each herder, find all targets within sensing radius `xi`
        2. **Target Selection**: Each herder selects the target farthest from the goal
        3. **Position Calculation**: Compute desired position behind selected target
        4. **Force Calculation**: Computes attractive force toward desired position

        If no targets are within sensing range:
        - Herders inside goal region (distance < rho_g) remain stationary
        - Herders outside goal region move away from origin with speed v_h

        If a target is selected:
        - Apply attractive force: alpha * (desired_position - current_position)
        - Desired position is behind target: target_pos + delta * target_unit_vector
        """


        # Extract herder and target positions from the observation
        herder_pos = self.herders.x  # Shape (N, 2)
        target_pos = self.targets.x[:, :2]  # Shape (M, 2)

        distances, _ = compute_distances(self.herders.x, target_pos)  # Shape (N, M)

        target_distance_from_goal, _ = compute_distances(target_pos, self.environment.goal_pos)  # Shape (M, 2)

        # Find the index of the closest herder for each target
        closest_herders = np.argmin(distances, axis=0)  # Shape (M,)

        # Create a boolean mask where each target is only considered if it's closer to the current herder
        closest_mask = np.zeros_like(distances, dtype=bool)
        np.put_along_axis(closest_mask, closest_herders[np.newaxis, :], True, axis=0)

        # Create a boolean mask where distances are less than xi and the herder is the closest one
        mask = (distances < self.xi) & closest_mask  # Shape (N, M)

        # Calculate the absolute distances from the origin for the targets
        absolute_distances = np.linalg.norm(target_pos, axis=1)  # Shape (M,)

        # Use broadcasting to expand the absolute distances to match the shape of the mask
        expanded_absolute_distances = np.tile(absolute_distances, (self.herders.N, 1))  # Shape (N, M)

        # Apply the mask to get valid distances only
        valid_absolute_distances = np.where(mask, expanded_absolute_distances, -np.inf)  # Shape (N, M)

        # Find the index of the target with the maximum absolute distance from the origin for each herder
        selected_target_indices = np.argmax(valid_absolute_distances, axis=1)  # Shape (N,)

        # Create a mask to identify herders that have no valid targets
        no_valid_target_mask = np.all(~mask, axis=1)

        # Replace invalid indices with -1 (indicating no target)
        selected_target_indices = np.where(no_valid_target_mask, -1, selected_target_indices)

        # Create a vector (N, 2) to store the absolute position of the selected target for each herder
        selected_target_positions = np.zeros((self.herders.N, 2))
        selected_target_positions[~no_valid_target_mask] = target_pos[
            selected_target_indices[~no_valid_target_mask]]

        # Calculate unit vectors for herders and selected targets
        herder_unit_vectors = herder_pos / np.linalg.norm(herder_pos, axis=1, keepdims=True)  # Shape (N, 2)
        selected_target_unit_vectors = np.zeros((self.herders.N, 2))
        selected_target_unit_vectors[~no_valid_target_mask] = (
                target_pos[selected_target_indices[~no_valid_target_mask]] / np.linalg.norm(
            target_pos[selected_target_indices[~no_valid_target_mask]], axis=1, keepdims=True
        )
        )

        # Calculate actions for each herder
        actions = np.zeros((self.herders.N, 2))
        herder_abs_distances = np.linalg.norm(herder_pos, axis=1)  # Absolute distances of herders from the origin

        # If no target is selected and the herder's distance is less than rho_g, action is zero
        # Otherwise, action is v_h * herder_unit_vector
        no_target_selected = no_valid_target_mask & (herder_abs_distances < self.rho_g)
        actions[no_valid_target_mask & ~no_target_selected] = - self.v_h * herder_unit_vectors[
            no_valid_target_mask & ~no_target_selected]

        # If a target is selected, action is
        # alpha * (herder_pos - (selected_target_pos + delta * selected_target_unit_vector))
        actions[~no_valid_target_mask] = - self.alpha * (
                herder_pos[~no_valid_target_mask] - (
                selected_target_positions[~no_valid_target_mask] +
                self.delta * selected_target_unit_vectors[~no_valid_target_mask]
        )
        )

        return actions
