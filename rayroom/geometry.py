import numpy as np


def normalize(v):
    """
    Normalize a vector to unit length.

    :param v: Input vector.
    :type v: np.ndarray
    :return: Normalized vector.
    :rtype: np.ndarray
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def ray_plane_intersection(ray_origin, ray_dir, plane_point, plane_normal):
    """
    Calculate the intersection point of a ray and a plane.

    :param ray_origin: Origin point of the ray [x, y, z].
    :type ray_origin: np.ndarray
    :param ray_dir: Direction vector of the ray [x, y, z].
    :type ray_dir: np.ndarray
    :param plane_point: A point on the plane [x, y, z].
    :type plane_point: np.ndarray
    :param plane_normal: Normal vector of the plane [x, y, z].
    :type plane_normal: np.ndarray
    :return: The distance t along the ray to the intersection point, or None if no intersection.
             Intersection point = ray_origin + t * ray_dir.
    :rtype: float or None
    """
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-6:
        return None

    t = np.dot(plane_point - ray_origin, plane_normal) / denom
    if t < 0:
        return None  # Intersection behind ray

    return t


def is_point_in_polygon(point, vertices, normal):
    """
    Check if a point lying on the polygon's plane is inside the polygon.

    Uses the crossing number algorithm projected to 2D (dropping the dimension with
    the largest normal component).

    :param point: Point to check [x, y, z].
    :type point: np.ndarray
    :param vertices: Vertices of the polygon.
    :type vertices: np.ndarray
    :param normal: Normal vector of the polygon's plane.
    :type normal: np.ndarray
    :return: True if the point is inside, False otherwise.
    :rtype: bool
    """
    # Project 3D to 2D by dropping the dimension with largest normal component
    abs_n = np.abs(normal)
    if abs_n[0] > abs_n[1] and abs_n[0] > abs_n[2]:
        # Drop x, use y, z
        proj_p = point[1:]
        proj_v = vertices[:, 1:]
    elif abs_n[1] > abs_n[0] and abs_n[1] > abs_n[2]:
        # Drop y, use x, z
        proj_p = np.array([point[0], point[2]])
        proj_v = vertices[:, [0, 2]]
    else:
        # Drop z, use x, y
        proj_p = point[:2]
        proj_v = vertices[:, :2]

    # Crossing number algorithm
    inside = False
    n = len(proj_v)
    p1 = proj_v[0]
    for i in range(n + 1):
        p2 = proj_v[i % n]
        if min(p1[1], p2[1]) < proj_p[1] <= max(p1[1], p2[1]):
            if proj_p[0] <= max(p1[0], p2[0]):
                if p1[1] != p2[1]:
                    xinters = (proj_p[1] - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                if p1[0] == p2[0] or proj_p[0] <= xinters:
                    inside = not inside
        p1 = p2

    return inside


def reflect_vector(incident, normal):
    """
    Calculate the reflection of a vector.

    :param incident: Incident vector.
    :type incident: np.ndarray
    :param normal: Surface normal vector.
    :type normal: np.ndarray
    :return: Reflected vector.
    :rtype: np.ndarray
    """
    return incident - 2 * np.dot(incident, normal) * normal


def random_direction_hemisphere(normal):
    """
    Generate a random direction in the hemisphere defined by a normal vector.
    Uses cosine-weighted sampling, often used for diffuse reflection.

    :param normal: The normal vector defining the hemisphere.
    :type normal: np.ndarray
    :return: A normalized random direction vector.
    :rtype: np.ndarray
    """
    # Create a random coordinate system
    if abs(normal[0]) > 0.9:
        u = np.array([0.0, 1.0, 0.0])
    else:
        u = np.array([1.0, 0.0, 0.0])

    u = normalize(np.cross(u, normal))
    v = np.cross(normal, u)

    # Random samples
    phi = 2 * np.pi * np.random.random()
    r2 = np.random.random()
    r = np.sqrt(r2)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(1 - r2)

    # Transform to world coordinates
    direction = x * u + y * v + z * normal
    return normalize(direction)
