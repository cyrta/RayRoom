import os
import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile
from ..raytracer.core import RayTracer
from ..ism import ImageSourceEngine
from ...objects import AmbisonicReceiver
from ...utils import generate_rir


class HybridRenderer:
    """
    Implements a hybrid acoustic rendering method combining Image Source Method (ISM)
    for early reflections and Ray Tracing for late reverberation.
    """

    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0):
        """
        Initialize the HybridRenderer.

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
        self._tracer = RayTracer(room, temperature, humidity)
        self.ism_engine = ImageSourceEngine(room, temperature, humidity)
        self.source_audios = {}
        self.source_gains = {}

    def set_source_audio(self, source, audio, gain=1.0):
        """
        Assign audio data to a Source object.
        """
        if isinstance(audio, str):
            data = self._load_wav(audio)
        else:
            data = np.array(audio)
        self.source_audios[source] = data
        self.source_gains[source] = gain

    def _load_wav(self, path):
        """
        Load a WAV file and convert it to a mono float array.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        fs, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if fs != self.fs:
            print(f"Warning: Sample rate mismatch {fs} vs {self.fs}. Playback speed will change.")
        return data

    def render(self, n_rays=10000, max_hops=50, rir_duration=2.0,
               verbose=True, record_paths=False, interference=False,
               ism_order=3):
        """
        Run the hybrid rendering pipeline.
        """
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        all_paths = {} if record_paths else None
        valid_sources = [s for s in self.room.sources if s in self.source_audios]

        if not valid_sources:
            print("No sources with assigned audio found.")
            if record_paths:
                return receiver_outputs, all_paths
            return receiver_outputs

        for source in valid_sources:
            if verbose:
                print(f"Simulating Source: {source.name} (ISM Order: {ism_order})")

            # Reset histograms
            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    rx.w_histogram, rx.x_histogram, rx.y_histogram, rx.z_histogram = [], [], [], []
                else:
                    rx.amplitude_histogram = []

            # 1. ISM for early reflections
            self.ism_engine.run(source, max_order=ism_order)

            # 2. Ray Tracing for late reverberation
            paths = self._tracer.run(source, n_rays, max_hops, record_paths=record_paths)
            if record_paths and paths:
                all_paths.update(paths)

            # 3. Combine Histograms & Generate RIRs
            for rx in self.room.receivers:
                # Histograms are now combined on the receiver objects
                if isinstance(rx, AmbisonicReceiver):
                    # For Ambisonic, ISM provides only one histogram. We can add it to the 'W' channel.
                    # This is a simplification. A more accurate approach would require directional ISM.
                    rirs = [
                        generate_rir(rx.w_histogram, self.fs, rir_duration, not interference),
                        generate_rir(rx.x_histogram, self.fs, rir_duration, not interference),
                        generate_rir(rx.y_histogram, self.fs, rir_duration, not interference),
                        generate_rir(rx.z_histogram, self.fs, rir_duration, not interference),
                    ]
                else:
                    rirs = [generate_rir(rx.amplitude_histogram, self.fs, rir_duration, not interference)]

                # 4. Convolve and Mix
                source_audio = self.source_audios[source]
                gain = self.source_gains.get(source, 1.0)

                if isinstance(rx, AmbisonicReceiver):
                    processed_channels = [fftconvolve(source_audio * gain, rir, mode='full') for rir in rirs]
                    max_len = max(len(pc) for pc in processed_channels)
                    padded_channels = [np.pad(pc, (0, max_len - len(pc))) for pc in processed_channels]
                    processed = np.stack(padded_channels, axis=1)
                else:
                    processed = fftconvolve(source_audio * gain, rirs[0], mode='full')

                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Pad and add
                    current_len = receiver_outputs[rx.name].shape[0]
                    new_len = processed.shape[0]
                    if new_len > current_len:
                        padding_shape = (new_len - current_len,) + receiver_outputs[rx.name].shape[1:]
                        receiver_outputs[rx.name] = np.concatenate([receiver_outputs[rx.name], np.zeros(padding_shape)])
                    elif current_len > new_len:
                        padding_shape = (current_len - new_len,) + processed.shape[1:]
                        processed = np.concatenate([processed, np.zeros(padding_shape)])
                    receiver_outputs[rx.name] += processed

        # Normalize final output
        for name, audio in receiver_outputs.items():
            if audio is not None and np.max(np.abs(audio)) > 0:
                receiver_outputs[name] /= np.max(np.abs(audio))

        if record_paths:
            return receiver_outputs, all_paths
        return receiver_outputs
