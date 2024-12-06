import numpy as np


def xi_shepherding(populations, environment):
    goal_region_center = environment.goal_pos
    target_radii = np.linalg.norm(populations[0].x - goal_region_center, axis=1)
    goal_region_radius = environment.goal_radius
    n_in = np.sum(target_radii < goal_region_radius)
    xi = n_in/populations[0].N
    return xi


def get_done_shepherding(populations, environment):
    xi = xi_shepherding(populations, environment)
    if xi == 1:
        return True
    else:
        return False
