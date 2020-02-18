import math


def euclid_dist_2D(p1: tuple, p2: tuple):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def magnitude(vec: tuple):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)

def polar_to_cartesian(vec_polar: tuple):
    return (vec_polar[0] * math.cos(vec_polar[1]), vec_polar[0] * math.sin(vec_polar[1]))

def normalized(vec: tuple):
    m = magnitude(vec)
    return(vec[0]/m, vec[1]/m)

def angle_unsigned(vec1: tuple, vec2: tuple):
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    return math.acos(dot_product)

def angle_signed(vec1: tuple, vec2: tuple):
    return math.atan2(vec2[1], vec2[0]) - math.atan2(vec1[1],vec1[0])