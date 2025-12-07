import numpy as np
from scipy.signal import fftconvolve
from scipy.signal import butter, sosfilt, resample

from .fdtd import FDTDSolver
from ..hybrid.hybrid import HybridRenderer


def add_to_mix(mix, signal):
    """ Helper to add numpy arrays, padding if needed. """
    if mix is None:
        return signal
    if signal is None:
        return mix

    # Ensure both are numpy arrays for shape checks
    mix = np.asarray(mix)
    signal = np.asarray(signal)

    # Handle Ambisonic mixing: mono signal should be added to W-channel
    if mix.ndim > 1 and signal.ndim == 1:  # mix is ambisonic, signal is mono
        shorter = min(len(mix), len(signal))
        # Create a copy to avoid modifying the original array in place
        new_mix = mix.copy()
        new_mix[:shorter, 0] += signal[:shorter]
        return new_mix
    elif signal.ndim > 1 and mix.ndim == 1:  # signal is ambisonic, mix is mono
        shorter = min(len(mix), len(signal))
        new_signal = signal.copy()
        new_signal[:shorter, 0] += mix[:shorter]
        return new_signal

    # Standard mono or ambisonic mixing (shapes must now be compatible)
    if len(signal) > len(mix):
        if mix.ndim > 1:
            padding = np.zeros((len(signal) - len(mix), mix.shape[1]))
        else:
            padding = np.zeros(len(signal) - len(mix))
        mix = np.concatenate((mix, padding))
    elif len(mix) > len(signal):
        if signal.ndim > 1:
            padding = np.zeros((len(mix) - len(signal), signal.shape[1]))
        else:
            padding = np.zeros(len(mix) - len(signal))
        signal = np.concatenate((signal, padding))

    return mix + signal


class SpectralRenderer(HybridRenderer):
    """
    A Spectral Hybrid Renderer.
    Combines Wave-based FDTD (Low Frequency) and Geometric ISM/RayTracing (High Frequency).
    """

    def __init__(self, room, fs=44100, crossover_freq=1000.0, temperature=20.0, humidity=50.0):
        """
        :param crossover_freq: Frequency in Hz to split Wave and Geometric methods.
        """
        super().__init__(room, fs, temperature, humidity)
        self.crossover_freq = crossover_freq

        # Initialize FDTD Solver
        # We set max_freq slightly higher than crossover to ensure overlap/good behavior
        self.fdtd = FDTDSolver(room, max_freq=crossover_freq * 1.2)

    def render(self, n_rays=20000, max_hops=50, rir_duration=1.5,
               verbose=True, record_paths=False, ism_order=2):
        """
        Run the spectral hybrid pipeline.
        """
        if verbose:
            print(f"--- Spectral Hybrid Rendering (X-over: {self.crossover_freq} Hz) ---")
            print("Phase 1: Geometric Rendering (High Frequency)...")

        # Get the full-band geometric audio, which is valid for the high-frequency part
        geo_outputs = super().render(n_rays, max_hops, rir_duration,
                                     verbose, record_paths, False, ism_order)

        paths = None
        if record_paths:
            geo_audio_dict, paths, geo_rirs = geo_outputs
        else:
            geo_audio_dict, geo_rirs = geo_outputs

        if verbose:
            print("Phase 2: Wave Simulation (Low Frequency)...")

        valid_sources = [s for s in self.room.sources if s in self.source_audios]

        # Create high-pass and low-pass filters
        sos_lp = butter(4, self.crossover_freq, 'low', fs=self.fs, output='sos')
        sos_hp = butter(4, self.crossover_freq, 'high', fs=self.fs, output='sos')

        # 1. High-pass filter the geometric part to get the HF audio
        hpf_geo_outputs = {
            rx_name: sosfilt(sos_hp, audio, axis=0) if audio is not None else None
            for rx_name, audio in geo_audio_dict.items()
        }

        # 2. Generate and accumulate the low-frequency audio for all sources
        lpf_audio_accumulator = {rx.name: None for rx in self.room.receivers}

        for source in valid_sources:
            if verbose:
                print(f"  FDTD for Source: {source.name}")

            # Run FDTD to get the low-frequency impulse response (IR)
            fdtd_ir, fdtd_fs = self.fdtd.run(duration=min(rir_duration, 0.5),
                                             sources=[source],
                                             receivers=self.room.receivers)

            src_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)

            for rx in self.room.receivers:
                if fdtd_ir.get(rx) is not None:
                    raw_ir = fdtd_ir[rx]

                    # Resample the IR from FDTD's sampling rate to the project's
                    num_samples = int(len(raw_ir) * self.fs / fdtd_fs)
                    resampled_ir = resample(raw_ir, num_samples)

                    # Low-pass filter the resampled IR
                    filtered_ir = sosfilt(sos_lp, resampled_ir)

                    # Convolve the filtered IR with the source audio to get the LF audio
                    processed_lf_audio = fftconvolve(src_audio * gain, filtered_ir, mode='full')

                    # Add this source's LF audio to the accumulator for this receiver
                    current_mix = lpf_audio_accumulator[rx.name]
                    lpf_audio_accumulator[rx.name] = add_to_mix(current_mix, processed_lf_audio)

        # 3. Combine the final HF and LF audio parts
        final_outputs = {}
        for rx in self.room.receivers:
            hpf_part = hpf_geo_outputs.get(rx.name)
            lpf_part = lpf_audio_accumulator.get(rx.name)
            final_outputs[rx.name] = add_to_mix(hpf_part, lpf_part)

        # The returned RIR is from the geometric part only.
        # A true hybrid RIR would involve combining LF and HF RIRs.
        if record_paths:
            return final_outputs, paths, geo_rirs

        return final_outputs, geo_rirs
