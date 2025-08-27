"""
Specialized utility functions for shepherding task analysis and evaluation.

This module provides analysis tools specifically designed for shepherding simulations,
including success metrics, termination conditions, and performance evaluation utilities.
"""

import numpy as np


def get_target_distance(targets, environment):
    """
    Compute distances from target agents to the goal region center.

    This function calculates the Euclidean distance from each target agent
    to the center of the goal region, providing essential information for
    shepherding analysis and termination conditions.

    Parameters
    ----------
    targets : Population
        Target population (typically sheep) with position attribute 'x'.
    environment : Environment
        Environment instance containing goal position ('goal_pos') and dimensions.

    Returns
    -------
    np.ndarray
        Array of distances from each target to goal center. Shape: (n_targets,).

    Mathematical Details
    --------------------
    Distance calculation:

    .. math::
        d_i = \\|\\mathbf{p}_i - \\mathbf{g}\\|_2

    Where :math:`\\mathbf{p}_i` is the position of target i and :math:`\\mathbf{g}` is the goal center.

    Notes
    -----
    - Uses only spatial dimensions from target positions
    - Compatible with arbitrary environment dimensions
    - Optimized for repeated calls during simulation
    """
    goal_region_center = environment.goal_pos
    target_radii = np.linalg.norm(targets.x[:, :len(environment.dimensions)] - goal_region_center, axis=1)
    return target_radii


def xi_shepherding(targets, environment):
    """
    Compute the shepherding success metric (fraction of targets in goal region).

    This function calculates the key performance metric for shepherding tasks:
    the fraction of target agents that are currently within the goal region.
    This metric is commonly denoted as ξ (xi) in shepherding literature.

    Parameters
    ----------
    targets : Population
        Target population with position attribute 'x'.
    environment : Environment
        Environment with 'goal_pos' and 'goal_radius' attributes.

    Returns
    -------
    float
        Fraction of targets within goal region, range [0, 1].

    Mathematical Definition
    -----------------------
    The shepherding metric is defined as:

    .. math::
        \\xi = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbf{1}[d_i < r_{goal}]

    Where:
    - :math:`N` is the total number of targets
    - :math:`d_i` is the distance from target i to goal center
    - :math:`r_{goal}` is the goal region radius
    - :math:`\\mathbf{1}[\\cdot]` is the indicator function


    Applications
    ------------
    - **Performance Evaluation**: Primary metric for shepherding success
    - **Termination Conditions**: End episodes when ξ reaches threshold
    - **Real-time Monitoring**: Track progress during simulation

    Notes
    -----
    - Returns value in range [0, 1] where 1 indicates perfect shepherding
    - Commonly used threshold values: 0.8-0.95 for task completion
    - Efficient computation suitable for real-time evaluation
    - Compatible with arbitrary goal region shapes (uses distance to center)
    """
    target_radii = get_target_distance(targets, environment)
    n_in = np.sum(target_radii < environment.goal_radius)
    xi = n_in/targets.N
    return xi


def get_done_shepherding(populations, environment, xi=None, threshold=1):
    """
    Determine if shepherding task has reached completion criteria.

    This function evaluates whether a shepherding task should be considered complete
    based on the fraction of targets within the goal region. It provides flexible
    termination logic for both simulation and reinforcement learning applications.

    Parameters
    ----------
    populations : list or Population
        Target population(s) to evaluate. If list, uses first element.
    environment : Environment
        Environment containing goal region specifications.
    xi : float, optional
        Pre-computed shepherding success metric. If None, computes from populations.
    threshold : float, optional
        Success threshold for task completion. Default is 1.0 (100% success).

    Returns
    -------
    bool
        True if shepherding task is complete, False otherwise.

    Threshold Guidelines
    --------------------
    Common threshold values and their applications:

    - **threshold = 1.0**: Perfect shepherding (all targets in goal)
    - **threshold = 0.95**: Near-perfect (95% success, practical for noisy environments)  
    - **threshold = 0.9**: High success (90% success, commonly used benchmark)
    - **threshold = 0.8**: Moderate success (80% success, for challenging scenarios)

    
    Applications
    ------------
    - **Episode Termination**: End RL episodes when task is complete
    - **Simulation Control**: Stop simulations at successful completion
    - **Performance Benchmarking**: Define success criteria for algorithm comparison

    Notes
    -----
    - Returns immediately if xi is provided, avoiding recomputation
    - Threshold of 1.0 requires perfect shepherding (may be too strict for noisy systems)
    - Compatible with both single populations and population lists
    - Commonly used in conjunction with xi_shepherding function
    """
    if xi is None:
        xi = xi_shepherding(populations, environment)
    if xi >= threshold:
        return True
    else:
        return False

