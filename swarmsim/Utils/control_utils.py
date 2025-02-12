import numpy as np
from numba import njit


# def compute_distances(positions_1, positions_2):
#
#     # Extract agents positions and expand them to broadcast distance computation
#     if positions_1.ndim > 1:
#         positions_1 = positions_1[:, np.newaxis, :]   # Shape (N, 1, 2)
#     if positions_2.ndim > 1:
#         positions_2 = positions_2[np.newaxis, :, :]   # Shape (1, M, 2)
#
#     # Calculate the relative positions
#     relative_positions = positions_1 - positions_2  # Shape (N, M, 2)
#
#     # Compute the Euclidean distances between herders and targets
#     distances = np.linalg.norm(relative_positions, axis=2)  # Shape (N, M)
#
#     return distances, relative_positions


@njit
def compute_distances(positions_1, positions_2):
    N = positions_1.shape[0]
    M = positions_2.shape[0]

    distances = np.zeros((N, M))
    relative_positions = np.zeros((N, M, 2))  # Store relative positions

    for i in range(N):
        for j in range(M):
            dx = positions_1[i, 0] - positions_2[j, 0]
            dy = positions_1[i, 1] - positions_2[j, 1]
            relative_positions[i, j, 0] = dx
            relative_positions[i, j, 1] = dy
            distances[i, j] = np.sqrt(dx * dx + dy * dy)

    return distances, relative_positions

