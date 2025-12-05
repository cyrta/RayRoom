import numpy as np


class Material:
    """
    Defines the acoustic properties of a surface.
    """

    def __init__(self, name, absorption=0.1, transmission=0.0, scattering=0.0):
        """
        Initialize a Material.

        :param name: Name of the material.
        :type name: str
        :param absorption: Absorption coefficient (alpha). 0 = perfect reflection, 1 = perfect absorption.
                           Can be a single float or array for frequency bands. Defaults to 0.1.
        :type absorption: float or np.array
        :param transmission: Transmission coefficient (tau). 0 = opaque, 1 = fully transparent.
                             Defaults to 0.0.
        :type transmission: float or np.array
        :param scattering: Scattering coefficient (s). 0 = specular, 1 = diffuse. Defaults to 0.0.
        :type scattering: float or np.array
        """
        self.name = name
        self.absorption = np.array(absorption) if isinstance(absorption, (list, tuple)) else np.array([absorption])
        self.transmission = np.array(transmission) if isinstance(
            transmission, (list, tuple)) else np.array([transmission])
        self.scattering = np.array(scattering) if isinstance(scattering, (list, tuple)) else np.array([scattering])

    def __repr__(self):
        return f"Material({self.name}, abs={self.absorption}, trans={self.transmission})"

# Common Materials Library


def get_material(name):
    """
    Retrieve a standard material by name.

    :param name: Name of the material (e.g., "concrete", "glass", "wood").
    :type name: str
    :return: A Material object.
    :rtype: rayroom.materials.Material
    """
    # Simplified values, ideally these would be frequency dependent arrays
    materials = {
        "concrete": Material("Concrete", absorption=0.05, transmission=0.0),
        "brick": Material("Brick", absorption=0.03, transmission=0.0),
        "glass": Material("Glass", absorption=0.03, transmission=0.1),  # Partial transmission
        "heavy_curtain": Material("Heavy Curtain", absorption=0.6, transmission=0.2),
        "wood": Material("Wood", absorption=0.15, transmission=0.01),
        "plaster": Material("Plaster", absorption=0.1, transmission=0.0),
        "air": Material("Air", absorption=0.0, transmission=1.0),  # Fully transparent
        "transparent_wall": Material("TransparentWall", absorption=0.1, transmission=0.8),
        "human": Material("Human", absorption=0.5, transmission=0.0, scattering=0.5),
        "asphalt": Material("Asphalt", absorption=0.1, transmission=0.0, scattering=0.1),
        "grass": Material("Grass", absorption=0.5, transmission=0.0, scattering=0.6),
        "soil": Material("Soil", absorption=0.3, transmission=0.0, scattering=0.7),
        "metal": Material("Metal", absorption=0.05, transmission=0.0, scattering=0.1),
    }
    return materials.get(name, Material("Default", 0.1, 0.0))
