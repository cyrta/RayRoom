import numpy as np
from scipy.signal import butter, sosfilt, resample
from scipy.io import wavfile
from scipy.signal import fftconvolve
from .audio import AudioRenderer, generate_rir
from .hybrid import HybridRenderer
from .fdtd import FDTDSolver

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
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        
        # 1. Generate High Frequency RIRs (Geometric: ISM + Ray)
        # We assume the HybridRenderer.render logic can return RIRs? 
        # Actually, HybridRenderer.render returns *mixed audio*.
        # We need the underlying RIRs to filter them.
        # The base classes are designed to convolve internally.
        # Refactoring needed: Separate RIR generation from Convolution.
        # Or, we can just let it render broadband, and we HPF the result?
        # Yes, if the source audio is broadband, rendering broadband geometric and then HPFing 
        # is equivalent to HPFing the RIR.
        
        if verbose:
            print(f"--- Spectral Hybrid Rendering (X-over: {self.crossover_freq} Hz) ---")
            print("Phase 1: Geometric Rendering (High Frequency)...")
        
        # Run the standard Hybrid render
        # This gives us the geometric estimation (valid for HF)
        geo_outputs = super().render(n_rays, max_hops, rir_duration, 
                                     verbose, record_paths, False, ism_order)
        
        # 2. Generate Low Frequency RIRs (Wave: FDTD)
        if verbose:
            print("Phase 2: Wave Simulation (Low Frequency)...")
            
        # Run FDTD with a pulse to get raw LF response
        # We need to identify which sources are active
        valid_sources = [s for s in self.room.sources if s in self.source_audios]
        
        # FDTD run
        # We run one simulation per source? Or all together?
        # IRs need to be separate if we want to convolve with different signals.
        # Standard FDTD can simulate multiple sources, but recording 'who sent what' 
        # requires separate runs if the system is linear.
        
        # Let's run sequentially for each source to get individual IRs
        
        # Filter design
        sos_lp = butter(4, self.crossover_freq, 'low', fs=self.fs, output='sos')
        sos_hp = butter(4, self.crossover_freq, 'high', fs=self.fs, output='sos')
        
        # Extract audio dict from geo_outputs potentially being a tuple
        if isinstance(geo_outputs, tuple):
             geo_audio_dict = geo_outputs[0]
        else:
             geo_audio_dict = geo_outputs

        final_outputs = {rx.name: np.zeros_like(geo_audio_dict.get(rx.name, [])) 
                         if geo_audio_dict.get(rx.name) is not None else None 
                         for rx in self.room.receivers}

        # Helper to ensure length match
        def add_to_mix(mix, signal):
            if mix is None: return signal
            if len(signal) > len(mix):
                mix = np.pad(mix, (0, len(signal)-len(mix)))
            elif len(mix) > len(signal):
                signal = np.pad(signal, (0, len(mix)-len(signal)))
            return mix + signal

        # Geometric Output filtering (High Pass)
        if isinstance(geo_outputs, tuple):
             geo_audio_dict = geo_outputs[0]
        else:
             geo_audio_dict = geo_outputs
             
        for rx_name, audio in geo_audio_dict.items():
            # Check if audio is None before processing
            if audio is not None:
                # High Pass Filter the Geometric Result
                filtered_geo = sosfilt(sos_hp, audio)
                final_outputs[rx_name] = add_to_mix(final_outputs[rx_name], filtered_geo)
            else:
                # If no geometric audio for this receiver (e.g. no rays reached it),
                # we should handle it. final_outputs is already initialized to zeros/None appropriately?
                # Actually, final_outputs initialization logic was:
                # zeros_like(geo_outputs[rx]) if geo_outputs[rx] is not None else None
                # If geo_outputs[rx] is None, final_outputs[rx] is None.
                pass

        # Wave Output (Low Pass)
        for source in valid_sources:
            if verbose:
                print(f"  FDTD for Source: {source.name}")
            
            # Run FDTD
            # Duration slightly longer than RIR? FDTD is slow.
            # Let's limit to 0.5s or 1.0s for MVP. 
            # rir_duration is a good target.
            fdtd_signals, fdtd_fs = self.fdtd.run(duration=min(rir_duration, 0.5), 
                                                  sources=[source], 
                                                  receivers=self.room.receivers)
            
            # Resample and Convolve
            src_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)
            
            for rx in self.room.receivers:
                raw_ir = fdtd_signals[rx]
                
                # Resample FDTD output (usually low fs) to Project fs (44100)
                num_samples = int(len(raw_ir) * self.fs / fdtd_fs)
                resampled_ir = resample(raw_ir, num_samples)
                
                # Low Pass Filter the Wave Result
                # (FDTD naturally has a cutoff, but clean it up)
                filtered_ir = sosfilt(sos_lp, resampled_ir)
                
                # Convolve with source audio
                processed = fftconvolve(src_audio * gain, filtered_ir, mode='full')
                
                # Normalize energy level?
                # FDTD pressure units vs Ray energy units match is TRICKY.
                # RayTracer: Energy ~ 1/r^2.
                # FDTD: Pressure ~ 1/r.
                # fftconvolve uses Pressure IR.
                # But RayTracer IR construction (generate_rir) used sqrt(Energy) ~ Pressure.
                # So units "should" be comparable if source magnitude is calibrated.
                # Calibration is the hardest part of hybrid.
                # MVP: Just sum them.
                
                if final_outputs[rx.name] is None:
                    final_outputs[rx.name] = processed
                else:
                    final_outputs[rx.name] = add_to_mix(final_outputs[rx.name], processed)

        if record_paths:
            # Return paths from Geometric part
            return final_outputs, geo_outputs[1] if isinstance(geo_outputs, tuple) else {}
            
        return final_outputs

