from swarmsim.Controllers import Controller
import numpy as np
from typing import cast

from swarmsim.Utils import compute_distances
from swarmsim.Environments import ShepherdingEnvironment
from swarmsim.Populations import Population


class ShepherdingLamaController(Controller):
    """
    Implementation of the herders control law from [Lama and di Bernardo, 2024]


    Arguments
    ---------
        population : Population
            The population where the control is exerted
        targets : Population
            The population of targets
        environment : ShepherdingEnvironment
            The environment where the agents live (must be a Sheperding Environment)
        config_path: str
            The path of the configuration file

    Config requirements
    -------------------
    The YAML configuration file must contain the following parameters under the population's section:

    dt: float
        Sampling time of the controller (time interval between two consecutive control actions)
    xi: float 
        Sensing radius of the herder agents
    v_h: float
        Speed of the herders when no targets are detected
    alpha: float
        Attraction force constant to the selected target
    lmbda: float
        Not USED 
    delta: float
        Displacement to go behind a target
    rho_g: float
        Radius of the goal region

    Examples
    --------
    Example YAML configuration:

    .. code-block:: yaml

        ShepherdingLamaController:
            xi: 15
            v_h: 12
            alpha: 3
            lambda: 3
            delta: 1.5
            rho_g: 5

    This defines a `ShepherdingLamaController` with xi=15, v_h=12, alpha=3, lambda=3, delta=1.5, rho_g=5.
    This controller is able to steer a population of targets to the goal region.

    """

    def __init__(self, population: Population,
                 targets: Population,
                 environment: ShepherdingEnvironment =None,
                 config_path: str =None) -> None:

        super().__init__(population, environment, config_path, [targets])
        self.herders: Population = self.population
        self.targets: Population = targets
        self.environment = cast(ShepherdingEnvironment, self.environment)

        self.xi: float = self.config.get('xi', 100000)
        self.v_h: float = self.config.get('v_h', 12)
        self.alpha: float = self.config.get('alpha', 3)
        self.lmbda: float = self.config.get('lambda', 2.5)
        self.delta: float = self.config.get('delta', 1.25)
        self.rho_g: float = self.config.get('rho_g', 10)

        self.xi = 5

    def get_action(self) -> np.ndarray:

        """
        STEFANO O ITALO TODO

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
