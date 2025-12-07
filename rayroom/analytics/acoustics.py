import numpy as np
from scipy.signal import butter, lfilter


def schroeder_integration(rir):
    """
    Compute the Schroeder integral of a room impulse response.

    :param rir: The room impulse response signal.
    :type rir: np.ndarray
    :return: The Schroeder decay curve in dB.
    :rtype: np.ndarray
    """
    # Use 64-bit floats to avoid overflow with squared values
    energy = np.power(rir.astype(np.float64), 2)
    sch = np.cumsum(energy[::-1])[::-1]
    
    max_sch = np.max(sch)
    if max_sch < 1e-20: # If signal is essentially silent
        return np.full_like(sch, -200.0)

    sch_db = 10 * np.log10(sch / max_sch + 1e-20)
    return sch_db


def calculate_rt60(sch_db, fs, t_start=None, t_end=None):
    """
    Calculate RT60 from a Schroeder decay curve using linear regression.
    The regression is performed between -5 dB and -25 dB (for T20),
    but is adapted if the signal has a lower dynamic range.

    :param sch_db: Schroeder decay curve in dB.
    :type sch_db: np.ndarray
    :param fs: Sampling frequency in Hz.
    :type fs: int
    :param t_start: Start time for the linear fit (in dB, e.g., -5).
    :type t_start: float, optional
    :param t_end: End time for the linear fit (in dB, e.g., -25 for T20).
    :type t_end: float, optional
    :return: The calculated RT60 value in seconds.
    :rtype: float
    """
    if t_start is None:
        t_start = -5
    if t_end is None:
        t_end = -25

    # Find start of decay (first point under max)
    try:
        start_idx = np.where(sch_db < sch_db[0] - 1e-6)[0][0]
    except IndexError:
        start_idx = 0  # Flat curve

    # Find points for the linear fit
    try:
        fit_start_idx = np.where(sch_db[start_idx:] <= t_start)[0][0] + start_idx
        fit_end_idx = np.where(sch_db[fit_start_idx:] <= t_end)[0][0] + fit_start_idx
    except IndexError:
        # Not enough decay, try to fit on a smaller range
        try:
            fit_start_idx = np.where(sch_db <= -5)[0][0]
            # Find the point where it has decayed at least 15 dB more
            end_val = sch_db[fit_start_idx] - 15
            fit_end_idx = np.where(sch_db[fit_start_idx:] <= end_val)[0][0] + fit_start_idx
        except IndexError:
            return np.nan

    if fit_end_idx <= fit_start_idx + 10:  # Need at least a few points
        return np.nan

    # Time vector for the selected range
    t = np.arange(fit_start_idx, fit_end_idx) / fs

    # Linear regression
    coeffs = np.polyfit(t, sch_db[fit_start_idx:fit_end_idx], 1)
    slope = coeffs[0]

    # RT60 is the time to decay by 60 dB
    if slope >= -1e-3:  # Effectively zero or positive slope
        return np.nan

    rt60 = -60 / slope
    return rt60


def octave_band_filter(data, fs, center_freq, order=4):
    """
    Filter a signal into an octave band.

    :param data: Input signal.
    :type data: np.ndarray
    :param fs: Sampling frequency.
    :type fs: int
    :param center_freq: Center frequency of the octave band.
    :type center_freq: float
    :param order: Order of the Butterworth filter.
    :type order: int, optional
    :return: The filtered signal.
    :rtype: np.ndarray
    """
    # Octave band limits (factor of sqrt(2))
    f_low = center_freq / np.sqrt(2)
    f_high = center_freq * np.sqrt(2)
    # Nyquist frequency
    nyquist = 0.5 * fs
    # Critical frequencies (normalized)
    low = f_low / nyquist
    high = f_high / nyquist
    # Avoid issues at boundaries
    if high >= 1.0:
        high = 0.9999
    if low <= 0:
        low = 1e-6
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def get_octave_bands(subdivisions=1):
    """
    Returns standard octave band center frequencies, with optional subdivisions.

    :param subdivisions: Number of points per octave interval.
                         e.g. 1 for standard bands, 5 for 3 intermediate points.
    :type subdivisions: int
    """
    base_bands = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    if subdivisions <= 1:
        return np.round(base_bands).astype(int)

    all_bands = []
    for i in range(len(base_bands) - 1):
        start_freq = base_bands[i]
        end_freq = base_bands[i+1]
        # Use endpoint=False to avoid duplicating frequencies at interval boundaries
        sub_bands = np.geomspace(start_freq, end_freq, subdivisions, endpoint=False)
        all_bands.extend(sub_bands)

    all_bands.append(base_bands[-1])  # Manually add the last band center

    return np.round(all_bands).astype(int)


def calculate_clarity(rir, fs, time_threshold_ms):
    """
    Calculate a clarity metric (C50, C80) from a Room Impulse Response.

    :param rir: The room impulse response.
    :param fs: Sampling frequency.
    :param time_threshold_ms: The time threshold in milliseconds (50 for C50, 80 for C80).
    :return: Clarity value in dB.
    """
    is_ambisonic = rir.ndim > 1
    rir_to_process = rir[:, 0] if is_ambisonic else rir

    # Find the index of the direct sound (maximum of the RIR)
    direct_sound_idx = np.argmax(np.abs(rir_to_process))

    # Convert time threshold to samples
    threshold_samples = int((time_threshold_ms / 1000.0) * fs)

    # Early energy: from direct sound up to the threshold
    early_energy_end_idx = direct_sound_idx + threshold_samples
    early_energy = np.sum(rir_to_process[direct_sound_idx:early_energy_end_idx]**2)

    # Late energy: from the threshold to the end of the RIR
    late_energy = np.sum(rir_to_process[early_energy_end_idx:]**2)

    if late_energy < 1e-20:  # Avoid division by zero if there's no late energy
        return 100.0  # Return a very high dB value

    clarity = 10 * np.log10(early_energy / late_energy)
    return clarity


def calculate_drr(rir, fs, direct_sound_window_ms=5):
    """
    Calculate the Direct-to-Reverberant Ratio (DRR).

    :param rir: The room impulse response.
    :param fs: Sampling frequency.
    :param direct_sound_window_ms: The window size in ms to consider as direct sound around the peak.
    :return: DRR value in dB.
    """
    is_ambisonic = rir.ndim > 1
    rir_to_process = rir[:, 0] if is_ambisonic else rir

    # Find the index of the direct sound (maximum of the RIR)
    direct_sound_idx = np.argmax(np.abs(rir_to_process))

    # Define window for direct sound in samples
    window_samples = int((direct_sound_window_ms / 1000.0) * fs)
    direct_energy_start = max(0, direct_sound_idx - window_samples // 2)
    direct_energy_end = min(len(rir_to_process), direct_sound_idx + window_samples // 2)

    direct_energy = np.sum(rir_to_process[direct_energy_start:direct_energy_end]**2)

    # Reverberant energy is everything else
    reverberant_energy = np.sum(rir_to_process**2) - direct_energy

    if reverberant_energy < 1e-20:
        return 100.0  # Very high dB if no reverberant energy

    drr = 10 * np.log10(direct_energy / reverberant_energy)
    return drr
