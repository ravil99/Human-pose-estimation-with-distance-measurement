import numpy as np

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'

    :param array v1 : vector to first object (human)
    :param array v2 : vector to second object (human)

    :return: angle beetwen two vetors in rad
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def distance_beetween2objects(v1_magnitude, v2_magnitude, angle):
    """
    :param array v1_magnitude : vector magnitude to first object (human)
    :param array v2_magnitude : vector magnitude to second object (human)
    :param rad angle: angle beetwen vector v1 and v2 

    :return: the distance beetwen two points in 3D world
    """

    return np.sqrt(v1_magnitude**2 + v2_magnitude**2 - 2 * v1_magnitude * v2_magnitude * np.cos(angle))

"""
Example

a = angle_between(np.array((1, 0, 0)), np.array((0, 1, 0)))
print(a)

d = distance_beetween2objects(10, 10, a)

print(d)
"""