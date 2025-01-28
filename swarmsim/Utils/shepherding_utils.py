import numpy as np


def get_target_distance(targets, environment):
    goal_region_center = environment.goal_pos
    target_radii = np.linalg.norm(targets.x[:, :2] - goal_region_center, axis=1)
    return target_radii


def xi_shepherding(targets, environment):
    target_radii = get_target_distance(targets, environment)
    goal_region_radius = environment.goal_radius
    n_in = np.sum(target_radii < goal_region_radius)
    xi = n_in/targets.N
    return xi


def get_done_shepherding(populations, environment):
    xi = xi_shepherding(populations, environment)
    if xi == 1:
        return True
    else:
        return False
