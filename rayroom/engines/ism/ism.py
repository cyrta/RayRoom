import numpy as np
from ...objects import AmbisonicReceiver, Receiver
from ...physics import C_SOUND, air_absorption_coefficient
from ...geometry import ray_plane_intersection, is_point_in_polygon, normalize


class ImageSource:
    def __init__(self, position, order, parent=None, generating_wall=None):
        self.position = np.array(position)
        self.order = order
        self.parent = parent
        self.generating_wall = generating_wall


class ImageSourceEngine:
    """
    Implements the Image Source Method for calculating early specular reflections.
    """
    def __init__(self, room, temperature=20.0, humidity=50.0):
        self.room = room
        self.temperature = temperature
        self.humidity = humidity
        # Match RayTracer's reference frequency
        self.air_absorption_db_m = air_absorption_coefficient(
            1000.0, temperature, humidity
        )

    def run(self, source, max_order=2, verbose=True):
        """
        Compute early reflections for a source-receiver pair.
        Populates receiver.energy_histogram with deterministic early reflections.
        """
        if verbose:
            print(f"ISM: Processing Source {source.name}...")

            # 1. Generate Image Sources
        images = self._generate_image_sources(source, max_order)

        if verbose:
            print(
                f"  Generated {len(images)} image sources "
                f"(Order <= {max_order})"
            )

        # 2. Check visibility and record for each receiver
        for receiver in self.room.receivers:
            self._process_receiver(source, receiver, images)

    def _generate_image_sources(self, source, max_order):
        images = []

        # Add original source as order 0
        original = ImageSource(source.position, 0, None, None)
        images.append(original)

        self._recursive_images(original, images, max_order)
        return images

    def _recursive_images(self, current_image, all_images, max_depth):
        if current_image.order >= max_depth:
            return

        for wall in self.room.walls:
            # Don't reflect back across the same wall immediately
            if current_image.generating_wall == wall:
                continue

            # Check if source is in front of the wall
            # Vector from wall point to source
            vec = current_image.position - wall.vertices[0]
            dist = np.dot(vec, wall.normal)

            # Standard ISM reflects everything and filters later.
            if abs(dist) < 1e-6:
                continue  # On the plane

            # Reflect: P' = P - 2 * dist * N
            reflected_pos = current_image.position - 2 * dist * wall.normal

            new_image = ImageSource(
                reflected_pos, current_image.order + 1, current_image, wall
            )
            all_images.append(new_image)

            self._recursive_images(new_image, all_images, max_depth)

    def _process_receiver(self, real_source, receiver, images):
        # For each image, check visibility path to receiver
        for img in images:
            result = self._construct_path(img, receiver)

            if result is None:
                continue

            path_points, walls_hit = result

            # Verify validity (intersections within polygons) and Occlusion
            if self._validate_path(path_points, walls_hit):
                # Calculate energy and time
                self._record_reflection(
                    real_source, receiver, img, path_points, walls_hit
                )

    def _construct_path(self, image, receiver):
        """
        Backtrack from receiver to image source to find interaction points.
        Returns tuple (list of points, list of walls) or None.
        Points: [ReceiverPos, Hit1, Hit2, ..., SourcePos]
        Walls: [Wall1, Wall2, ...] corresponding to Hit1, Hit2...
        """
        path_points = [receiver.position]
        walls_hit = []

        current_target = receiver.position
        current_img = image

        # Backtrack
        while current_img.parent is not None:
            wall = current_img.generating_wall
            parent = current_img.parent

            vec = current_img.position - current_target
            dist_to_img = np.linalg.norm(vec)
            if dist_to_img < 1e-9:
                return None

            ray_dir = vec / dist_to_img

            t = ray_plane_intersection(
                current_target, ray_dir, wall.vertices[0], wall.normal
            )

            if t is None or t < 1e-5 or t > dist_to_img + 1e-5:
                return None

            intersection_point = current_target + t * ray_dir
            path_points.append(intersection_point)
            walls_hit.append(wall)

            current_target = intersection_point
            current_img = parent

        path_points.append(current_img.position)  # Real source position

        return path_points, walls_hit

    def _validate_path(self, points, walls):
        """
        Check if:
        1. Intersection points are inside their respective wall polygons.
        2. Segments are not occluded by *other* walls or furniture.
        """
        # 1. Check Polygons
        # Note: points[1] corresponds to walls[0] (Hit1)
        for i, wall in enumerate(walls):
            hit_point = points[i+1]
            if not is_point_in_polygon(hit_point, wall.vertices, wall.normal):
                return False

        # 2. Check Occlusion
        # Segments: (Points[i] -> Points[i+1])
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]

            ignore_walls = []
            if i > 0:
                ignore_walls.append(walls[i-1])
            if i < len(walls):
                ignore_walls.append(walls[i])

            if self._is_occluded(p1, p2, ignore_walls):
                return False

        return True

    def _is_occluded(self, p1, p2, ignore_walls):
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist < 1e-5:
            return False

        ray_dir = vec / dist

        # Check Walls
        for wall in self.room.walls:
            if wall in ignore_walls:
                continue

            t = ray_plane_intersection(
                p1, ray_dir, wall.vertices[0], wall.normal
            )
            if t is not None and 1e-4 < t < dist - 1e-4:
                # Hit a wall plane between p1 and p2
                # Check if inside polygon
                hit_p = p1 + t * ray_dir
                if is_point_in_polygon(hit_p, wall.vertices, wall.normal):
                    return True

        # Check Furniture
        for furn in self.room.furniture:
            # Check each face of the furniture
            for f_idx, normal in enumerate(furn.face_normals):
                plane_pt = furn.face_planes[f_idx]
                t = ray_plane_intersection(p1, ray_dir, plane_pt, normal)

                if t is not None and 1e-4 < t < dist - 1e-4:
                    face_verts = furn.vertices[furn.faces[f_idx]]
                    hit_p = p1 + t * ray_dir
                    if is_point_in_polygon(hit_p, face_verts, normal):
                        return True

        return False

    def _record_reflection(
            self, source, receiver, image, path_points, walls_hit
    ):
        # Calculate total distance
        total_dist = 0.0
        for i in range(len(path_points) - 1):
            total_dist += np.linalg.norm(path_points[i+1] - path_points[i])

        time = total_dist / C_SOUND

        # Geometric Spreading: Power * Area / (4 * pi * r^2)
        receiver_area = np.pi * receiver.radius**2
        geom_factor = receiver_area / (4 * np.pi * total_dist**2 + 1e-12)

        # Directivity
        # Direction from source (last point) to first hit (second to last)
        dir_vec = path_points[-2] - path_points[-1]
        dir_vec = normalize(dir_vec)

        gain = 1.0
        if hasattr(source, 'directivity') and \
           source.directivity != "omnidirectional":
            if hasattr(source, 'orientation'):
                cos_theta = np.dot(dir_vec, source.orientation)
                if source.directivity == "cardioid":
                    gain = 0.5 * (1.0 + cos_theta)
                elif source.directivity == "subcardioid":
                    gain = 0.7 + 0.3 * cos_theta
                elif source.directivity == "hypercardioid":
                    gain = np.abs(0.25 + 0.75 * cos_theta)
                elif source.directivity == "bidirectional":
                    gain = np.abs(cos_theta)

        energy = source.power * gain * geom_factor

        # Air Absorption
        # E = E0 * 10^(-alpha * dist / 10)
        energy *= 10**(-self.air_absorption_db_m * total_dist / 10.0)

        # Wall Absorption (Reflection coefficients)
        for wall in walls_hit:
            mat = wall.material
            abs_coeff = np.mean(mat.absorption) \
                if np.ndim(mat.absorption) > 0 else mat.absorption
            energy *= (1.0 - abs_coeff)

        # Record
        if isinstance(receiver, AmbisonicReceiver):
            # Direction from last bounce (or source) to receiver
            arrival_dir = normalize(path_points[0] - path_points[1])
            receiver.record(time, energy, arrival_dir)
        elif isinstance(receiver, Receiver):
            receiver.record(time, energy)
