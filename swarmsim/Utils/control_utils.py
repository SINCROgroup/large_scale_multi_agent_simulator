import numpy as np


def compute_distances(positions_1, positions_2):

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
    This method, given the parametes of a Gaussin and some points where to evaluate the function,
    computes a 2D Gaussian distribution. Assumption: the covariance matrix is diagonal
    Arguments
    ---
    A (float):      Maximum amplitude of the signal
    mu_x(float):    Average on the first dimension
    mu_y(float):    Average on the second dimension
    sigma_x(float): Standard deviation in the first dimension
    sigma_y(float): Standard deviation in the second dimension
    """
    return A * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2)) - ((y - mu_y)**2 / (2 * sigma_y**2)))
