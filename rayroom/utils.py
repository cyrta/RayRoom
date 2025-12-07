import numpy as np

def generate_rir(histogram, fs=44100, duration=2.0, random_phase=True):
    """
    Generate a Room Impulse Response (RIR) from an energy histogram.
    The histogram contains tuples of (time, amplitude).
    """
    if not histogram:
        return np.zeros(int(fs * duration))

    # Sort by time
    histogram.sort(key=lambda x: x[0])

    times = np.array([t for t, a in histogram])
    amplitudes = np.array([a for t, a in histogram])

    # Discard late reflections
    valid_indices = times < duration
    times = times[valid_indices]
    amplitudes = amplitudes[valid_indices]

    if len(times) == 0:
        return np.zeros(int(fs * duration))

    if random_phase:
        # Apply random sign flips to break phase coherence for diffuse sounds
        signs = np.random.choice([-1, 1], size=len(amplitudes))
        amplitudes *= signs

    # Create RIR
    rir_len = int(fs * duration)
    rir = np.zeros(rir_len)

    # Place amplitudes in the RIR
    indices = (times * fs).astype(int)

    # Handle multiple arrivals in the same sample bin
    np.add.at(rir, indices, amplitudes)

    return rir
