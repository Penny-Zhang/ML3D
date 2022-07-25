"""SDF to Occupancy Grid"""
import numpy as np


def occupancy_grid(sdf_function, resolution):
    """
    Create an occupancy grid at the specified resolution given the implicit representation.
    :param sdf_function: A function that takes in a point (x, y, z) and returns the sdf at the given point.
    Points may be provides as vectors, i.e. x, y, z can be scalars or 1D numpy arrays, such that (x[0], y[0], z[0])
    is the first point, (x[1], y[1], z[1]) is the second point, and so on
    :param resolution: Resolution of the occupancy grid
    :return: An occupancy grid of specified resolution (i.e. an array of dim (resolution, resolution, resolution) with value 0 outside the shape and 1 inside.
    """

    # ###############
    # TODO: Implement
#     sdf_values_1 = []
#     sdf_grid = np.empty((resolution + 1, resolution + 1, resolution + 1))
#     for a in np.arange(resolution + 1):
#         for b in np.arange(resolution + 1):
#             for c in np.arange(resolution + 1):
#                 i = a/resolution - 0.5
#                 j = b/resolution - 0.5
#                 k = c/resolution - 0.5
#                 sdf_grid[a][b][c] = sdf_function(i, j, k)
#                 sdf_values_1.append(sdf_grid[a][b][c])
                

    
    x_range = y_range = z_range = np.linspace(-0.50, 0.50, resolution).astype(np.float32)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    sdf_values = sdf_function(grid_x, grid_y, grid_z)

    
#     x = np.linspace(-0.5, 0.5, resolution + 1)
#     y = np.linspace(-0.5, 0.5, resolution + 1)
#     z = np.linspace(-0.5, 0.5, resolution + 1)
#     coordinate = []
#     for a in range(resolution + 1):
#         for b in range(resolution + 1):
#             for c in range(resolution + 1):
#                 temp = [x[a], y[b], z[c]]
#                 coordinate.append(temp)  
#     x = np.array([m[0] for m in coordinate])
#     y = np.array([m[1] for m in coordinate])
#     z = np.array([m[2] for m in coordinate])
    
#     sdf_values = sdf_function(x, y, z)

    sdf_grid = np.empty((resolution, resolution, resolution))
    count = 0
    for a in range(resolution):
        for b in range(resolution):
            for c in range(resolution):
                sdf_grid[a][b][c] = sdf_values[count]
                count += 1
                
    for a in range(resolution):
        for b in range(resolution):
            for c in range(resolution):
                if sdf_grid[a][b][c] > 0:
                    sdf_grid[a][b][c] = 0
                else: 
                    sdf_grid[a][b][c] = 1
    occupancy_grid = sdf_grid
    return occupancy_grid
    raise NotImplementedError
    # ###############
