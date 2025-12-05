import os
import numpy as np
from .core import RayTracer
from scipy.io import wavfile
from scipy.signal import fftconvolve


class AudioRenderer:
    """
    Handles the audio rendering pipeline for a Room.

    Manages sources, audio data, ray tracing, Room Impulse Response (RIR) generation,
    convolution, and mixing to produce the final audio output for each receiver.
    """

    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0):
        """
        Initialize the AudioRenderer.

        :param room: The Room object to render.
        :type room: rayroom.room.Room
        :param fs: Sampling rate in Hz. Defaults to 44100.
        :type fs: int
        :param temperature: Temperature in Celsius. Defaults to 20.0.
        :type temperature: float
        :param humidity: Relative humidity in percent. Defaults to 50.0.
        :type humidity: float
        """
        self.room = room
        self.fs = fs
        self.source_audios = {}  # Map source_obj -> audio_array
        self.source_gains = {}  # Map source_obj -> linear gain
        self._tracer = RayTracer(room, temperature=temperature, humidity=humidity)

    def set_source_audio(self, source, audio_data, gain=1.0):
        """
        Assign audio data to a Source object.

        :param source: The Source object in the room.
        :type source: rayroom.objects.Source
        :param audio_data: Audio data as a numpy array or a path to a WAV file.
        :type audio_data: np.ndarray or str
        :param gain: Linear gain factor for this source's audio. Defaults to 1.0.
        :type gain: float
        """
        if isinstance(audio_data, str):
            # Load from file
            data = self._load_wav(audio_data)
            self.source_audios[source] = data
        else:
            self.source_audios[source] = np.array(audio_data)

        self.source_gains[source] = gain

    def _load_wav(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        fs, data = wavfile.read(path)

        # Convert to float
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0

        # Mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample if needed (Basic check only)
        if fs != self.fs:
            print(
                f"Warning: Sample rate mismatch {fs} vs {self.fs}. "
                "Playback speed will change. Resampling not fully implemented."
            )
            # TODO: Implement resampling

        return data

    def render(self, n_rays=20000, max_hops=50, rir_duration=2.0,
               verbose=True, record_paths=False, interference=False):
        """
        Run the full rendering pipeline.

        1. Traces rays for each source.
        2. Generates an energy histogram for each receiver.
        3. Converts histograms to RIRs.
        4. Convolves source audio with RIRs.
        5. Mixes output for each receiver.

        :param n_rays: Number of rays per source. Defaults to 20000.
        :type n_rays: int
        :param max_hops: Maximum reflections. Defaults to 50.
        :type max_hops: int
        :param rir_duration: Duration of the generated Impulse Response in seconds. Defaults to 2.0.
        :type rir_duration: float
        :param verbose: Print progress. Defaults to True.
        :type verbose: bool
        :param record_paths: Return ray paths for visualization. Defaults to False.
        :type record_paths: bool
        :param interference: If True, enables deterministic phase for interference effects. Defaults to False.
        :type interference: bool
        :return: If record_paths is False, returns a dict {receiver_name: mixed_audio_array}.
                 If True, returns tuple (receiver_outputs, paths_data).
        :rtype: dict or tuple
        """
        # Initialize outputs for each receiver
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        all_paths = {} if record_paths else None

        # Iterate over sources that have audio assigned
        # Only render sources that are in the room AND have audio
        valid_sources = [
            s for s in self.room.sources
            if s in self.source_audios
        ]

        if not valid_sources:
            print("No sources with assigned audio found in the room.")
            return receiver_outputs

        for source in valid_sources:
            if verbose:
                print(f"Rendering Source: {source.name}")

            # 1. Clear Receiver Histograms
            for rx in self.room.receivers:
                rx.energy_histogram = []

            # 2. Run Ray Tracing (Source -> All Receivers)
            # We use internal tracer but target specific source
            # core.py's RayTracer.run iterates all room sources.
            # We want to run just one source.
            # RayTracer._trace_source is internal but we can use it if we subclass or modify core.
            # Alternatively, we temporarily set room.sources to [source]

            original_sources = self.room.sources
            self.room.sources = [source]
            paths = self._tracer.run(n_rays=n_rays, max_hops=max_hops,
                                     record_paths=record_paths)  # Prints "Simulating Source..."
            if record_paths and paths:
                all_paths.update(paths)
            self.room.sources = original_sources

            # 3. For each receiver, generate RIR and Convolve
            source_audio = self.source_audios[source]
            # Apply gain
            gain = self.source_gains.get(source, 1.0)

            for rx in self.room.receivers:
                # Generate RIR
                rir = generate_rir(rx.energy_histogram, fs=self.fs,
                                   duration=rir_duration,
                                   random_phase=not interference)

                # Convolve
                # Apply source gain to audio before convolution
                # Note: source_audio is shared, so we multiply on the fly or copy.
                # FFT convolve is linear: conv(gain*audio, rir) = gain*conv(audio, rir)

                processed = fftconvolve(source_audio * gain, rir, mode='full')

                # Mix into receiver output
                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Pad
                    current = receiver_outputs[rx.name]
                    if len(processed) > len(current):
                        padding = np.zeros(len(processed) - len(current))
                        current = np.concatenate((current, padding))
                        receiver_outputs[rx.name] = current  # Update reference
                    elif len(current) > len(processed):
                        padding = np.zeros(len(current) - len(processed))
                        processed = np.concatenate((processed, padding))

                    receiver_outputs[rx.name] += processed

        # Normalize final outputs
        for name, audio in receiver_outputs.items():
            if audio is not None and np.max(np.abs(audio)) > 0:
                receiver_outputs[name] = audio / np.max(np.abs(audio))

        if record_paths:
            return receiver_outputs, all_paths
        return receiver_outputs


def generate_rir(energy_histogram, fs=44100, duration=None, random_phase=True):
    """
    Convert an energy histogram to a Room Impulse Response (RIR).

    It reconstructs the impulse response by placing impulses
    at arrival times with amplitudes scaled by the square root of energy.

    :param energy_histogram: List of (time, energy) tuples.
    :type energy_histogram: list[tuple]
    :param fs: Sampling rate in Hz. Defaults to 44100.
    :type fs: int
    :param duration: Length of IR in seconds. If None, fits to max arrival time.
    :type duration: float, optional
    :param random_phase: If True, randomizes phase of impulses. If False,
                         uses positive phase. Set to False to enable
                         deterministic interference. Defaults to True.
    :type random_phase: bool
    :return: The audio impulse response array.
    :rtype: np.ndarray
    """
    if not energy_histogram:
        return np.zeros(int(fs * (duration if duration else 1.0)))

    times, energies = zip(*energy_histogram)
    times = np.array(times)
    energies = np.array(energies)

    max_time = np.max(times)
    if duration is None:
        duration = max_time + 0.1

    n_samples = int(duration * fs)
    rir = np.zeros(n_samples)

    # Map times to indices
    indices = (times * fs).astype(int)

    # Filter out of bounds
    valid = indices < n_samples
    indices = indices[valid]
    energies = energies[valid]

    # RIR amplitude ~ sqrt(Energy) * sign
    if random_phase:
        signs = np.random.choice([-1, 1], size=len(indices))
    else:
        signs = np.ones(len(indices))

    amplitudes = signs * np.sqrt(energies)

    np.add.at(rir, indices, amplitudes)

    return rir


def convolve_and_mix(sources_data, fs=44100):
    """
    Legacy helper: Convolve source audios with their RIRs and mix.

    Kept for backward compatibility or manual usage.

    :param sources_data: List of dicts with 'audio' (numpy array) and 'rir' (numpy array).
    :type sources_data: list[dict]
    :param fs: Sampling rate. Defaults to 44100.
    :type fs: int
    :return: Mixed audio array.
    :rtype: np.ndarray
    """
    mixed = None

    for src in sources_data:
        audio = src['audio']
        rir = src['rir']

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        processed = fftconvolve(audio, rir, mode='full')

        if mixed is None:
            mixed = processed
        else:
            if len(processed) > len(mixed):
                padding = np.zeros(len(processed) - len(mixed))
                mixed = np.concatenate((mixed, padding))
            elif len(mixed) > len(processed):
                padding = np.zeros(len(mixed) - len(processed))
                processed = np.concatenate((processed, padding))

            mixed += processed

    if mixed is not None and np.max(np.abs(mixed)) > 0:
        mixed = mixed / np.max(np.abs(mixed))

    return mixed
