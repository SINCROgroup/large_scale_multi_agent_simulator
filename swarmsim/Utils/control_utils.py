"""
Control and interaction utility functions for multi-agent systems.

This module provides efficient computational utilities for control systems and
agent interactions, including distance calculations and spatial field generation.
"""

import numpy as np


def compute_distances(positions_1, positions_2):
    """
    Compute pairwise Euclidean distances between two sets of agent positions.

    This function efficiently calculates all pairwise distances between agents
    in two different populations using vectorized operations. It's optimized
    for large-scale multi-agent systems where distance-based interactions are common.

    Parameters
    ----------
    positions_1 : np.ndarray
        Positions of first agent population. Shape: (N, spatial_dim) or (spatial_dim,) for single agent.
    positions_2 : np.ndarray
        Positions of second agent population. Shape: (M, spatial_dim) or (spatial_dim,) for single agent.

    Returns
    -------
    distances : np.ndarray
        Pairwise Euclidean distances. Shape: (N, M).
    relative_positions : np.ndarray
        Relative position vectors from positions_2 to positions_1. Shape: (N, M, spatial_dim).

    Algorithm Details
    -----------------
    The function uses numpy broadcasting to efficiently compute all pairwise distances:

    1. **Shape Expansion**: Input arrays are expanded to enable broadcasting
    2. **Relative Positions**: Computed as positions_1 - positions_2
    3. **Distance Calculation**: Euclidean norm along spatial dimension

    Mathematical formulation:

    .. math::
        d_{ij} = \\|\\mathbf{p}_i^{(1)} - \\mathbf{p}_j^{(2)}\\|_2

    Where :math:`\\mathbf{p}_i^{(1)}` and :math:`\\mathbf{p}_j^{(2)}` are position vectors.

    Applications
    ------------
    - **Interaction Forces**: Repulsion, attraction, alignment calculations
    - **Neighbor Detection**: Finding agents within sensing or communication range
    - **Collision Avoidance**: Distance-based safety constraints

    Notes
    -----
    - Input arrays are automatically expanded to support broadcasting
    - Handles both single agents and populations seamlessly
    - Relative positions point from positions_2 to positions_1
    - Optimized for repeated calls in simulation loops
    - Compatible with 2D, 3D, or higher-dimensional spaces
    """

    # Extract agents positions and expand them to broadcast distance computation
    if positions_1.ndim > 1:
        positions_1 = positions_1[:, np.newaxis, :]   # Shape (N, 1, 2)
    if positions_2.ndim > 1:
        positions_2 = positions_2[np.newaxis, :, :]   # Shape (1, M, 2)

    # Calculate the relative positions
    relative_positions = positions_1 - positions_2  # Shape (N, M, 2)

    # Compute the Euclidean distances between herders and targets
    distances = np.linalg.norm(relative_positions, axis=2)  # Shape (N, M)

    return distances, relative_positions

def gaussian_input(x, y, A=5.0, mu_x=0.0, mu_y=0.0, sigma_x=1.0, sigma_y=1.0):
    """
    Evaluate a 2D Gaussian distribution at specified points with diagonal covariance.

    This function computes the value of a 2D Gaussian distribution at given coordinates,
    assuming a diagonal covariance matrix. It's commonly used for generating spatial
    fields, attraction/repulsion landscapes, and bio-inspired navigation signals.

    Parameters
    ----------
    x : float or np.ndarray
        X-coordinates where to evaluate the Gaussian.
    y : float or np.ndarray  
        Y-coordinates where to evaluate the Gaussian.
    A : float, optional
        Maximum amplitude (peak value) of the Gaussian. Default is 5.0.
    mu_x : float, optional
        Mean (center) of the Gaussian in the x-direction. Default is 0.0.
    mu_y : float, optional
        Mean (center) of the Gaussian in the y-direction. Default is 0.0.
    sigma_x : float, optional
        Standard deviation in the x-direction. Default is 1.0.
    sigma_y : float, optional
        Standard deviation in the y-direction. Default is 1.0.

    Returns
    -------
    np.ndarray or float
        Gaussian function values at the specified coordinates. Same shape as input arrays.

    Mathematical Formulation
    ------------------------
    The 2D Gaussian with diagonal covariance is defined as:

    .. math::
        f(x, y) = A \\exp\\left(-\\frac{(x - \\mu_x)^2}{2\\sigma_x^2} - \\frac{(y - \\mu_y)^2}{2\\sigma_y^2}\\right)

    Where:
    - :math:`A` is the amplitude (maximum value)
    - :math:`(\\mu_x, \\mu_y)` is the center point
    - :math:`(\\sigma_x, \\sigma_y)` are the standard deviations


    Applications
    ------------
    - **Spatial Control Fields**: Creating attraction/repulsion landscapes
    - **Potential Field Methods**: Path planning and obstacle avoidance

    Performance Notes
    -----------------
    - Vectorized operations support efficient grid evaluations
    - Computational complexity: O(N) where N is number of evaluation points
    - Memory efficient for large spatial grids
    - Compatible with numpy broadcasting

    Notes
    -----
    - Assumes diagonal covariance matrix (no correlation between x and y)
    - Peak value occurs at (mu_x, mu_y) with value A
    """
    return A * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2)) - ((y - mu_y)**2 / (2 * sigma_y**2)))
