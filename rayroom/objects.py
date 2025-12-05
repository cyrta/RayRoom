import numpy as np
from .materials import get_material

class Object3D:
    def __init__(self, name, position, material=None):
        self.name = name
        self.position = np.array(position, dtype=float)
        self.material = material if material else get_material("default")

class Source(Object3D):
    def __init__(self, name, position, power=1.0, orientation=None, directivity="omnidirectional"):
        """
        Source with optional directivity.
        orientation: vector [x,y,z] pointing forward.
        directivity: "omnidirectional", "cardioid", "hypercardioid", "bidirectional", or custom function(angle).
        """
        super().__init__(name, position)
        self.power = power # Scalar or array for bands
        self.orientation = np.array(orientation if orientation else [1, 0, 0], dtype=float)
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation /= norm
        self.directivity = directivity

class Receiver(Object3D):
    def __init__(self, name, position, radius=0.1):
        super().__init__(name, position)
        self.radius = radius
        self.energy_histogram = [] # To store arriving energy packets (time, energy)

    def record(self, time, energy):
        self.energy_histogram.append((time, energy))

class Furniture(Object3D):
    def __init__(self, name, vertices, faces, material=None):
        """
        Simple mesh definition.
        vertices: list of [x,y,z]
        faces: list of [v1_idx, v2_idx, v3_idx, ...]
        """
        super().__init__(name, [0,0,0], material) # Position is relative or origin
        self.vertices = np.array(vertices)
        self.faces = faces # List of lists of indices
        
        # Precompute normals and plane equations for faces
        self.face_normals = []
        self.face_planes = [] # Point on plane
        
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
    Approximated as a box for simplicity.
    """
    def __init__(self, name, position, height=1.7, width=0.5, depth=0.3, material_name="heavy_curtain"):
        # Create box vertices centered at position (x,y) standing on z=position[2] or centered z?
        # Usually position is feet location.
        x, y, z = position
        w, d, h = width, depth, height
        
        # 8 corners
        verts = [
            [x-w/2, y-d/2, z],   [x+w/2, y-d/2, z],   [x+w/2, y+d/2, z],   [x-w/2, y+d/2, z], # Bottom
            [x-w/2, y-d/2, z+h], [x+w/2, y-d/2, z+h], [x+w/2, y+d/2, z+h], [x-w/2, y+d/2, z+h]  # Top
        ]
        
        # 6 faces (quads)
        faces = [
            [0, 1, 2, 3], # Bottom
            [4, 7, 6, 5], # Top
            [0, 4, 5, 1], # Front
            [1, 5, 6, 2], # Right
            [2, 6, 7, 3], # Back
            [3, 7, 4, 0]  # Left
        ]
        
        super().__init__(name, verts, faces, get_material(material_name))

