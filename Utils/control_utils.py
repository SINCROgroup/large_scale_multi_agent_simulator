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

    return distances
