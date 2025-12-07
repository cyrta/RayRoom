import numpy as np
from .materials import get_material


class Object3D:
    """
    Base class for 3D objects in the simulation.
    """

    def __init__(self, name, position, material=None):
        """
        Initialize an Object3D.

        :param name: Name of the object.
        :type name: str
        :param position: [x, y, z] coordinates of the object's center or reference point.
        :type position: list or np.ndarray
        :param material: Material properties of the object. Defaults to "default" material.
        :type material: rayroom.materials.Material, optional
        """
        self.name = name
        self.position = np.array(position, dtype=float)
        self.material = material if material else get_material("default")


class Source(Object3D):
    """
    Represents a sound source.
    """

    def __init__(self, name, position, power=1.0, orientation=None, directivity="omnidirectional"):
        """
        Initialize a Source.

        :param name: Name of the source.
        :type name: str
        :param position: [x, y, z] coordinates.
        :type position: list or np.ndarray
        :param power: Sound power of the source. Defaults to 1.0.
        :type power: float
        :param orientation: [x, y, z] vector pointing in the forward direction of the source.
        :type orientation: list or np.ndarray, optional
        :param directivity: Directivity pattern. Options: "omnidirectional", "cardioid",
                            "hypercardioid", "bidirectional", "subcardioid".
        :type directivity: str
        """
        super().__init__(name, position)
        self.power = power  # Scalar or array for bands
        self.orientation = np.array(orientation if orientation else [1, 0, 0], dtype=float)
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation /= norm
        self.directivity = directivity


class Receiver(Object3D):
    """
    Represents a microphone or listening point.
    """

    def __init__(self, name, position, radius=0.1):
        """
        Initialize a Receiver.

        :param name: Name of the receiver.
        :type name: str
        :param position: [x, y, z] coordinates.
        :type position: list or np.ndarray
        :param radius: Radius of the receiver sphere for ray intersection. Defaults to 0.1.
        :type radius: float
        """
        super().__init__(name, position)
        self.radius = radius
        self.amplitude_histogram = []  # To store arriving energy packets (time, amplitude)

    def record(self, time, energy):
        """
        Record an energy packet arrival.

        :param time: Arrival time in seconds.
        :type time: float
        :param energy: Energy value of the arriving packet.
        :type energy: float or np.ndarray
        """
        # Convert energy to amplitude
        if energy >= 0:
            self.amplitude_histogram.append((time, np.sqrt(energy)))


class AmbisonicReceiver(Object3D):
    """
    Represents a first-order Ambisonic microphone.
    """

    def __init__(self, name, position, orientation=None, radius=0.01):
        """
        Initialize an AmbisonicReceiver.

        :param name: Name of the receiver.
        :type name: str
        :param position: [x, y, z] coordinates.
        :type position: list or np.ndarray
        :param orientation: [x, y, z] vector pointing in the forward direction (X-axis).
        :type orientation: list or np.ndarray, optional
        :param radius: Radius for ray intersection tests.
        :type radius: float
        """
        super().__init__(name, position)
        self.radius = radius

        # Histograms for W, X, Y, Z channels
        self.w_histogram = []
        self.x_histogram = []
        self.y_histogram = []
        self.z_histogram = []

        # Define the orientation of the microphone capsules
        self.orientation = np.array(orientation if orientation is not None else [1, 0, 0], dtype=float)
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation /= norm

        # Create an orthonormal basis for the microphone's local coordinate system
        self.x_axis = self.orientation  # Forward

        # Ensure the up vector is not parallel to the forward vector
        up_global = np.array([0., 0., 1.])
        if np.allclose(np.abs(np.dot(self.x_axis, up_global)), 1.0):
            # If forward is aligned with global Z, use global Y as up
            up_global = np.array([0., 1., 0.])

        self.y_axis = np.cross(up_global, self.x_axis)  # Left
        self.y_axis /= np.linalg.norm(self.y_axis)

        self.z_axis = np.cross(self.x_axis, self.y_axis)  # Up
        self.z_axis /= np.linalg.norm(self.z_axis)

    def record(self, time, energy, direction):
        """
        Record an energy packet arrival from a specific direction.

        :param time: Arrival time in seconds.
        :type time: float
        :param energy: Energy value of the arriving packet.
        :type energy: float or np.ndarray
        :param direction: Normalized vector indicating the direction of arrival.
        :type direction: np.ndarray
        """
        if energy < 0:
            return

        amplitude = np.sqrt(energy)

        # W channel (omnidirectional)
        gain_w = 1.0
        self.w_histogram.append((time, amplitude * gain_w))

        # X, Y, Z channels (figure-of-eight / bidirectional)
        # Gain is the projection of the arrival direction onto the capsule's axis
        gain_x = np.dot(direction, self.x_axis)
        gain_y = np.dot(direction, self.y_axis)
        gain_z = np.dot(direction, self.z_axis)

        self.x_histogram.append((time, amplitude * gain_x))
        self.y_histogram.append((time, amplitude * gain_y))
        self.z_histogram.append((time, amplitude * gain_z))


class Furniture(Object3D):
    """
    Represents a complex 3D object (mesh) in the room, like a table or chair.
    """

    def __init__(self, name, vertices, faces, material=None):
        """
        Initialize Furniture.

        :param name: Name of the object.
        :type name: str
        :param vertices: List of [x, y, z] coordinates for the mesh vertices.
        :type vertices: list
        :param faces: List of faces, where each face is a list of vertex indices.
        :type faces: list[list[int]]
        :param material: Material properties.
        :type material: rayroom.materials.Material, optional
        """
        super().__init__(name, [0, 0, 0], material)  # Position is relative or origin
        self.vertices = np.array(vertices)
        self.faces = faces  # List of lists of indices

        # Precompute normals and plane equations for faces
        self.face_normals = []
        self.face_planes = []  # Point on plane

        for face in self.faces:
            p0 = self.vertices[face[0]]
            p1 = self.vertices[face[1]]
            p2 = self.vertices[face[2]]

            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            self.face_normals.append(normal)
            self.face_planes.append(p0)


class Person(Furniture):
    """
    Represents a person, approximated as a rectangular box.
    """

    def __init__(self, name, position, height=1.7, width=0.5, depth=0.3, material_name="human"):
        """
        Initialize a Person object.

        :param name: Name of the person.
        :type name: str
        :param position: [x, y, z] coordinates of the feet center.
        :type position: list or np.ndarray
        :param height: Height of the person in meters. Defaults to 1.7.
        :type height: float
        :param width: Width of the person (shoulder width) in meters. Defaults to 0.5.
        :type width: float
        :param depth: Depth of the person (chest depth) in meters. Defaults to 0.3.
        :type depth: float
        :param material_name: Name of the material to use (looked up via get_material). Defaults to "human".
        :type material_name: str
        """
        # Create box vertices centered at position (x,y) standing on z=position[2] or centered z?
        # Usually position is feet location.
        x, y, z = position
        w, d, h = width, depth, height

        # 8 corners
        verts = [
            [x-w/2, y-d/2, z],   [x+w/2, y-d/2, z],   [x+w/2, y+d/2, z],   [x-w/2, y+d/2, z],  # Bottom
            [x-w/2, y-d/2, z+h], [x+w/2, y-d/2, z+h], [x+w/2, y+d/2, z+h], [x-w/2, y+d/2, z+h]  # Top
        ]

        # 6 faces (quads)
        faces = [
            [0, 1, 2, 3],  # Bottom
            [4, 7, 6, 5],  # Top
            [0, 4, 5, 1],  # Front
            [1, 5, 6, 2],  # Right
            [2, 6, 7, 3],  # Back
            [3, 7, 4, 0]  # Left
        ]

        super().__init__(name, verts, faces, get_material(material_name))
