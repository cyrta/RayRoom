import os
import numpy as np
from .core import RayTracer
from ...objects import AmbisonicReceiver
from scipy.io import wavfile
from scipy.signal import fftconvolve
from ...utils import generate_rir


class RaytracingRenderer:
    """
    Handles the audio rendering pipeline for a Room using ray tracing.

    Manages sources, audio data, ray tracing, Room Impulse Response (RIR) generation,
    convolution, and mixing to produce the final audio output for each receiver.
    """

    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0):
        """
        Initialize the RaytracingRenderer.

        :param room: The Room object to render.
        :type room: rayroom.room.Room
        :param fs: Sampling rate in Hz. Defaults to 44100.
        :type fs: int
        :param temperature: Temperature in Celsius. Defaults to 20.0.
        :type temperature: float
        :param humidity: Relative humidity in percent. Defaults to 50.0.
        :type humidity: float
        """
        self._tracer = RayTracer(room, temperature, humidity)
        self.room = room
        self.fs = fs
        self.source_audios = {}  # Map source_obj -> audio_array
        self.source_gains = {}  # Map source_obj -> linear gain

    def set_source_audio(self, source, audio, gain=1.0):
        """
        Assign audio data to a Source object.

        :param source: The Source object in the room.
        :type source: rayroom.objects.Source
        :param audio: Audio data as a numpy array or a path to a WAV file.
        :type audio: np.ndarray or str
        :param gain: Linear gain factor for this source's audio. Defaults to 1.0.
        :type gain: float
        """
        if isinstance(audio, str):
            # Load from file
            data = self._load_wav(audio)
        else:
            data = np.array(audio)

        self.source_audios[source] = data
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
        :param interference: If True, enables deterministic phase for interference effects. Defaults to True.
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
                print(f"Simulating Source: {source.name}")

            # Clear receiver histograms for this source
            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    rx.w_histogram, rx.x_histogram, rx.y_histogram, rx.z_histogram = [], [], [], []
                else:
                    rx.amplitude_histogram = []

            # Run ray tracer for the single source
            paths = self._tracer.run(source, n_rays, max_hops, record_paths=record_paths)

            if record_paths and paths:
                all_paths.update(paths)

            # Generate RIR and convolve for each receiver
            source_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)

            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    # Generate 4 RIRs for W, X, Y, Z
                    rir_w = generate_rir(
                        rx.w_histogram, fs=self.fs, duration=rir_duration,
                        random_phase=not interference
                    )
                    rir_x = generate_rir(
                        rx.x_histogram, fs=self.fs, duration=rir_duration,
                        random_phase=not interference
                    )
                    rir_y = generate_rir(
                        rx.y_histogram, fs=self.fs, duration=rir_duration,
                        random_phase=not interference
                    )
                    rir_z = generate_rir(
                        rx.z_histogram, fs=self.fs, duration=rir_duration,
                        random_phase=not interference
                    )

                    # Convolve each channel
                    processed_w = fftconvolve(
                        source_audio * gain, rir_w, mode='full'
                    )
                    processed_x = fftconvolve(
                        source_audio * gain, rir_x, mode='full'
                    )
                    processed_y = fftconvolve(
                        source_audio * gain, rir_y, mode='full'
                    )
                    processed_z = fftconvolve(
                        source_audio * gain, rir_z, mode='full'
                    )

                    # Stack into a 4-channel array
                    max_len = max(
                        len(processed_w), len(processed_x),
                        len(processed_y), len(processed_z)
                    )

                    def pad(arr, length):
                        if len(arr) < length:
                            return np.pad(arr, (0, length - len(arr)))
                        return arr

                    processed_w = pad(processed_w, max_len)
                    processed_x = pad(processed_x, max_len)
                    processed_y = pad(processed_y, max_len)
                    processed_z = pad(processed_z, max_len)

                    processed = np.stack(
                        [processed_w, processed_x, processed_y, processed_z],
                        axis=1
                    )
                else:  # Standard Receiver
                    rir = generate_rir(
                        rx.amplitude_histogram, fs=self.fs,
                        duration=rir_duration, random_phase=not interference
                    )
                    processed = fftconvolve(
                        source_audio * gain, rir, mode='full'
                    )

                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Correctly pad and add signals
                    current = receiver_outputs[rx.name]
                    is_ambisonic = len(processed.shape) > 1

                    if len(processed) > len(current):
                        if is_ambisonic:
                            padding = np.zeros(
                                (len(processed) - len(current), 4)
                            )
                        else:
                            padding = np.zeros(len(processed) - len(current))
                        current = np.concatenate((current, padding))
                        receiver_outputs[rx.name] = current
                    elif len(current) > len(processed):
                        if is_ambisonic:
                            padding = np.zeros(
                                (len(current) - len(processed), 4)
                            )
                        else:
                            padding = np.zeros(len(current) - len(processed))
                        processed = np.concatenate((processed, padding))

                    receiver_outputs[rx.name] += processed

        # Normalize final output
        for name, audio in receiver_outputs.items():
            if audio is not None and np.max(np.abs(audio)) > 0:
                receiver_outputs[name] /= np.max(np.abs(audio))

        if record_paths:
            return receiver_outputs, all_paths
        return receiver_outputs
