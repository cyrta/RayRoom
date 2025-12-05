import numpy as np


def air_absorption_coefficient(freq, temperature=20.0, humidity=50.0, pressure=101325.0):
    """
    Calculate air absorption coefficient (alpha) in dB/m using ISO 9613-1.

    :param freq: Frequency in Hz.
    :type freq: float
    :param temperature: Temperature in Celsius. Defaults to 20.0.
    :type temperature: float
    :param humidity: Relative humidity in percent (0-100). Defaults to 50.0.
    :type humidity: float
    :param pressure: Atmospheric pressure in Pascals. Defaults to 101325.0.
    :type pressure: float
    :return: Absorption coefficient in dB/m.
    :rtype: float
    """
    # Constants for ISO 9613-1
    p_ref = 101325.0  # Reference pressure (Pa)
    T_ref = 293.15    # Reference temperature (K) (20 C)
    T_triple = 273.16  # Triple point isotherm temperature (K)

    # Convert temperature to Kelvin
    T = temperature + 273.15

    # Ratios
    p_ratio = pressure / p_ref
    T_ratio = T / T_ref

    # Saturation vapor pressure (psat) / p_ref
    # Formula: 10 ^ ( -6.8346 * (T_triple/T)^1.261 + 4.6151 )
    exponent = -6.8346 * ((T_triple / T)**1.261) + 4.6151
    psat_ratio = 10**exponent

    # Molar concentration of water vapor (h) in percent
    h = humidity * psat_ratio / p_ratio

    # Oxygen relaxation frequency (frO)
    frO = p_ratio * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))

    # Nitrogen relaxation frequency (frN)
    frN = p_ratio * (T_ratio**-0.5) * (9 + 280 * h * np.exp(-4.170 * ((T_ratio**(-1/3)) - 1)))

    # Attenuation coefficient alpha (dB/m)
    term1 = 1.84e-11 * (p_ratio**-1) * (T_ratio**0.5)
    term2 = (T_ratio**-2.5) * (
        (0.01275 * np.exp(-2239.1 / T) / (frO + (freq**2 / frO))) +
        (0.1068 * np.exp(-3352.0 / T) / (frN + (freq**2 / frN)))
    )

    alpha = 8.686 * (freq**2) * (term1 + term2)

    return alpha
