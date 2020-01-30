import math


def euclid_dist_2D(p1: tuple, p2: tuple):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def magnitude(vec: tuple):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def polar_to_cartesian(vec_polar: tuple):
    return (vec_polar[0] * math.cos(vec_polar[1]), vec_polar[0] * math.sin(vec_polar[1]))


def cartesian_to_polar(vec: tuple):
    m = magnitude(vec)
    a = math.atan2(vec[1], vec[0])
    return (m, a)


def rotate(vec: tuple, angle, center: tuple = (0.0, 0.0)):
    #rotate a point around the given center by the given angle(in radians)
    x = math.cos(angle) * (vec[0] - center[0]) - math.sin(angle) * (vec[1] - center[1]) + center[0]
    y = math.sin(angle) * (vec[0] - center[0]) - math.cos(angle) * (vec[1] - center[1]) + center[1]
    return (x,y)
