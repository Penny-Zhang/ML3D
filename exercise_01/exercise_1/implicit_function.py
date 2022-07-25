"""Definitions for Signed Distance Fields"""
import numpy as np


def signed_distance_sphere(x, y, z, r, x_0, y_0, z_0):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a sphere of radius r, centered at (x_0, y_0, z_0)
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :param r: radius of the sphere
    :param x_0: x coordinate of the center of the sphere
    :param y_0: y coordinate of the center of the sphere
    :param z_0: z coordinate of the center of the sphere
    :return: signed distance from the surface of the sphere
    """
    # ###############
    # TODO: Implement
    x_dis = np.square(x - x_0)
    y_dis = np.square(y - y_0)
    z_dis = np.square(z - z_0)
    distance = np.sqrt(x_dis + y_dis + z_dis)
    signed_distance = distance - r
    return signed_distance
    raise NotImplementedError
    # ###############


def signed_distance_torus(x, y, z, R, r, x_0, y_0, z_0):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a torus of minor radius r and major radius R, centered at (x_0, y_0, z_0)
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :param R: major radius of the torus
    :param r: minor radius of the torus
    :param x_0: x coordinate of the center of the torus
    :param y_0: y coordinate of the center of the torus
    :param z_0: z coordinate of the center of the torus
    :return: signed distance from the surface of the torus
    """
    # ###############
    # TODO: Implement
    x_dis = np.square(x - x_0)
    z_dis = np.square(z - z_0)
    a = np.sqrt(x_dis + z_dis) - R
    y_dis = np.square(y - y_0)
    distance = np.sqrt(np.square(a) + y_dis)
    signed_distance = distance - r
    return signed_distance
    raise NotImplementedError
    # ###############


def signed_distance_atom(x, y, z):
    """
    Returns the signed distance value of a given point (x, y, z) from the surface of a hydrogen atom consisting of a spherical proton, a torus orbit, and one spherical electron
    :param x: x coordinate(s) of point(s) at which the SDF is evaluated
    :param y: y coordinate(s) of point(s) at which the SDF is evaluated
    :param z: z coordinate(s) of point(s) at which the SDF is evaluated
    :return: signed distance from the surface of the hydrogen atom
    """
    proton_center = (0, 0, 0) #质子
    proton_radius = 0.1
    orbit_radius = 0.35  # The major radius of the orbit torus
    orbit_thickness = 0.01  # The minor radius of the orbit torus
    electron_center = (orbit_radius, 0, 0) #电子
    electron_radius = 0.05
    # ###############
    # TODO: Implement
    signed_distance_1 = signed_distance_sphere(x, y, z, 0.1, 0, 0, 0) # proton
    signed_distance_2 = signed_distance_torus(x, y, z, 0.35, 0.01, 0, 0, 0) # orbit
    signed_distance_3 = signed_distance_sphere(x, y, z, 0.05, 0.35, 0, 0)# electron
    signed_distance = np.minimum(np.array(signed_distance_1), np.array(signed_distance_2))
    signed_distance = np.minimum(signed_distance, np.array(signed_distance_3)) # union

    return signed_distance
    raise NotImplementedError
    # ###############
