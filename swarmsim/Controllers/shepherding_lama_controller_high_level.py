from swarmsim.Controllers import Controller
import numpy as np

from swarmsim.Utils import compute_distances


class ShepherdingLamaControllerHighLevel(Controller):
    """
    Implementation of the herders control law from [Lama and di Bernardo, 2024]
    Arguments
    -------
    population : The controlled population
    environment : The environment
    other_populations : A list of other populations

    Methods
    -------
    get_action(self):
        Returns the control action for the herders

    """

    def __init__(self, population, targets, environment=None, config_path=None) -> None:
        super().__init__(population, environment, config_path)
        self.herders = self.population
        self.targets = targets

        self.xi = self.params.get('xi', 1000)
        self.v_h = self.params.get('v_h', 12)
        self.alpha = self.params.get('alpha', 3)
        self.lmbda = self.params.get('lambda', 3)
        self.delta = self.params.get('delta', 1.5)
        self.rho_g = self.params.get('rho_g', 5)

    def get_action(self):
        # Extract herder and target positions from the observation
        herder_pos = self.herders.x  # Shape (N, 2)
        target_pos = self.targets.x[:, :2]  # Shape (M, 2)

        distances, _ = compute_distances(self.herders.x, target_pos)  # Shape (N, M)

        target_distance_from_goal, _ = compute_distances(target_pos, self.environment.goal_pos)  # Shape (M, 2)

        selectable_targets = ((distances < self.xi) &
                              (np.tile(target_distance_from_goal, self.herders.N).T > self.environment.goal_radius))

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

        return selected_target_indices
