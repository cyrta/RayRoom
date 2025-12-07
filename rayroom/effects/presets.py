import numpy as np
from pedalboard import (
    Pedalboard,
    Compressor,
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Phaser,
    Reverb,
    NoiseGate,
)


def apply_vocal_enhancement_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    A full vocal chain to enhance dialogue or singing.
    Includes a noise gate, de-esser (using a filter), compressor, and EQ.
    """
    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-16, ratio=3),
        Gain(gain_db=3)
    ])
    return board(audio, sample_rate)


def apply_telephone_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Simulates the sound of a telephone with distortion and a narrow frequency range.
    """
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=300),
        LowpassFilter(cutoff_frequency_hz=3400),
        Distortion(drive_db=12),
        Gain(gain_db=-3)
    ])
    return board(audio, sample_rate)


def apply_radio_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Simulates the sound of an AM radio broadcast.
    """
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=200),
        LowpassFilter(cutoff_frequency_hz=3000),
        Compressor(threshold_db=-15, ratio=5),
        Distortion(drive_db=10),
        Gain(gain_db=-3)
    ])
    return board(audio, sample_rate)


def apply_radio_2_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Simulates the sound of an AM radio broadcast with a noise floor of -45 dB.
    """
    board = Pedalboard([
        NoiseGate(threshold_db=-35, ratio=2),
        HighpassFilter(cutoff_frequency_hz=300),
        LowpassFilter(cutoff_frequency_hz=2500),
        Compressor(threshold_db=-24, ratio=5, attack_ms=100, release_ms=1000),
        Distortion(drive_db=15),
        Gain(gain_db=-5)
    ])
    return board(audio, sample_rate)


def apply_large_hall_reverb_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Adds a large reverb effect, simulating a concert hall or cathedral.
    """
    board = Pedalboard([
        Reverb(room_size=0.9, damping=0.7, wet_level=0.4, dry_level=0.5, width=0.8)
    ])
    return board(audio, sample_rate)


def apply_ambient_wash_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Creates a dreamy, ambient texture with long delays and reverb.
    """
    board = Pedalboard([
        Delay(delay_seconds=0.5, feedback=0.4, mix=0.5),
        Reverb(room_size=0.95, damping=0.8, wet_level=0.5, dry_level=0.5),
        LowpassFilter(cutoff_frequency_hz=6000),
    ])
    return board(audio, sample_rate)


def apply_phaser_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Adds a swirling phaser effect, great for guitars and keyboards.
    """
    board = Pedalboard([
        Phaser(rate_hz=0.5, depth=0.5, mix=0.5)
    ])
    return board(audio, sample_rate)


EFFECTS = {
    "vocal_enhancement": apply_vocal_enhancement_effect,
    "telephone": apply_telephone_effect,
    "radio": apply_radio_effect,
    "radio_2": apply_radio_2_effect,
    "large_hall_reverb": apply_large_hall_reverb_effect,
    "ambient_wash": apply_ambient_wash_effect,
    "phaser": apply_phaser_effect,
}


def get_effect(name: str):
    """
    Returns the effect function for the given name.
    """
    return EFFECTS.get(name)
