import numpy as np


def get_target_distance(targets, environment):
    goal_region_center = environment.goal_pos
    target_radii = np.linalg.norm(targets.x[:, :len(environment.dimensions)] - goal_region_center, axis=1)
    return target_radii


def xi_shepherding(targets, environment):
    target_radii = get_target_distance(targets, environment)
    n_in = np.sum(target_radii < environment.goal_radius)
    xi = n_in/targets.N
    return xi


def get_done_shepherding(populations, environment, xi=None, threshold=1):
    if xi is None:
        xi = xi_shepherding(populations, environment)
    if xi >= threshold:
        return True
    else:
        return False

