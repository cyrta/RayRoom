import numpy as np
from scipy.signal import fftconvolve

from .core import RadiositySolver
from ...core.utils import generate_rir
from ..hybrid.hybrid import HybridRenderer
from ...room.objects import Receiver, AmbisonicReceiver


class RadiosityRenderer(HybridRenderer):
    """
    A Hybrid Renderer combining ISM (Early Specular) and Radiosity (Late Diffuse).
    Replaces the stochastic Ray Tracing tail with a smooth Radiosity calculation.
    """
    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0, patch_size=0.5):
        super().__init__(room, fs, temperature, humidity)
        self.radiosity_solver = RadiositySolver(room, patch_size=patch_size)

    def render(self, ism_order=2, rir_duration=1.5, verbose=True):
        """
        Render using ISM + Radiosity.
        """
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        valid_sources = [s for s in self.room.sources if s in self.source_audios]
        for source in valid_sources:
            if verbose:
                print(f"Radiosity Rendering Source: {source.name}")
            # 1. ISM (Early Specular)
            # Clear histograms
            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    rx.w_histogram, rx.x_histogram, rx.y_histogram, rx.z_histogram = [], [], [], []
                elif isinstance(rx, Receiver):
                    rx.energy_histogram = []
            if verbose:
                print("  Phase 1: ISM (Early Specular)...")
            self.ism_engine.run(source, max_order=ism_order, verbose=False)

            # 2. Radiosity (Late Diffuse)
            if verbose:
                print("  Phase 2: Radiosity (Late Diffuse)...")
            # Solve energy flow
            # Time step for radiosity needs to be fine enough for RIR but coarse enough for speed.
            # 10ms (0.01s) is common for energy envelopes.
            # But for audio convolution, we need finer structure?
            # No, we reconstruct noise with this envelope.
            # Let's use 5ms.
            dt_rad = 0.005
            energy_history = self.radiosity_solver.solve(source, duration=rir_duration, time_step=dt_rad)
            # Collect at receivers
            for rx in self.room.receivers:
                # Get diffuse histogram
                diffuse_hist = self.radiosity_solver.collect_at_receiver(rx, energy_history, dt_rad)
                # Merge histograms
                # NOTE: For Ambisonic, the diffuse energy from Radiosity is omnidirectional.
                # It will only be added to the W channel. This is a limitation of combining
                # a non-directional method (Radiosity) with a directional one (Ambisonics).
                # The specular reflections from ISM will still be directional.

                # Convert diffuse energy history to amplitude before adding to histograms
                diffuse_amps = [(t, np.sqrt(e)) for t, e in diffuse_hist if e >= 0]

                if isinstance(rx, AmbisonicReceiver):
                    rx.w_histogram.extend(diffuse_amps)
                else:
                    rx.amplitude_histogram.extend(diffuse_amps)
            # 3. Generate RIR and Convolve
            source_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)

            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    # Generate 4 RIRs for W, X, Y, Z
                    rir_w = generate_rir(rx.w_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    rir_x = generate_rir(rx.x_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    rir_y = generate_rir(rx.y_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    rir_z = generate_rir(rx.z_histogram, fs=self.fs, duration=rir_duration, random_phase=True)

                    # Convolve each channel
                    processed_w = fftconvolve(source_audio * gain, rir_w, mode='full')
                    processed_x = fftconvolve(source_audio * gain, rir_x, mode='full')
                    processed_y = fftconvolve(source_audio * gain, rir_y, mode='full')
                    processed_z = fftconvolve(source_audio * gain, rir_z, mode='full')

                    # Stack into a 4-channel array
                    max_len = max(len(processed_w), len(processed_x), len(processed_y), len(processed_z))

                    def pad(arr, length):
                        if len(arr) < length:
                            return np.pad(arr, (0, length - len(arr)))
                        return arr

                    processed_w = pad(processed_w, max_len)
                    processed_x = pad(processed_x, max_len)
                    processed_y = pad(processed_y, max_len)
                    processed_z = pad(processed_z, max_len)
                    processed = np.stack([processed_w, processed_x, processed_y, processed_z], axis=1)

                else:  # Standard Receiver
                    rir = generate_rir(rx.amplitude_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    processed = fftconvolve(source_audio * gain, rir, mode='full')
                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Pad and add
                    curr = receiver_outputs[rx.name]
                    is_ambisonic = len(processed.shape) > 1

                    if len(processed) > len(curr):
                        if is_ambisonic:
                            curr = np.pad(curr, ((0, len(processed) - len(curr)), (0, 0)))
                        else:
                            curr = np.pad(curr, (0, len(processed) - len(curr)))
                    elif len(curr) > len(processed):
                        if is_ambisonic:
                            processed = np.pad(processed, ((0, len(curr) - len(processed)), (0, 0)))
                        else:
                            processed = np.pad(processed, (0, len(curr) - len(processed)))

                    receiver_outputs[rx.name] = curr + processed
        # Normalize
        for k, v in receiver_outputs.items():
            if v is not None:
                m = np.max(np.abs(v))
                if m > 0:
                    receiver_outputs[k] = v / m
        return receiver_outputs
