"""Creating an SDF grid"""
import numpy as np


def sdf_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An SDF grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with positive values outside the shape and negative values inside.
    """

    # ###############
    # TODO: Implement
#     sdf_grid = np.empty((resolution, resolution, resolution))
#     for a in np.arange(resolution):
#         for b in np.arange(resolution):
#             for c in np.arange(resolution):
#                 i = a/(resolution-1) - 0.5
#                 j = b/(resolution-1) - 0.5
#                 k = c/(resolution-1) - 0.5
#                 sdf_grid[a][b][c] = sdf_function(i, j, k)
                
    x_range = y_range = z_range = np.linspace(-0.50, 0.50, resolution).astype(np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    sdf_values = sdf_function(grid_x, grid_y, grid_z)
    sdf_grid = np.empty((resolution, resolution, resolution))
    count = 0
    for a in range(resolution):
        for b in range(resolution):
            for c in range(resolution):
                sdf_grid[a][b][c] = sdf_values[count]
                count += 1
    return sdf_grid
    raise NotImplementedError
    # ###############
