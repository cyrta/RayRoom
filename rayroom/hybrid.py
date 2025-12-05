import numpy as np
from scipy.signal import fftconvolve
from .audio import AudioRenderer, generate_rir
from .ism import ImageSourceEngine

class HybridRenderer(AudioRenderer):
    """
    A hybrid renderer combining Image Source Method (ISM) for early reflections
    and Ray Tracing for late reflections.
    """

    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0):
        super().__init__(room, fs, temperature, humidity)
        self._ism_engine = ImageSourceEngine(room, temperature, humidity)

    def render(self, n_rays=20000, max_hops=50, rir_duration=2.0,
               verbose=True, record_paths=False, interference=False,
               ism_order=2):
        """
        Run the hybrid rendering pipeline.
        
        1. ISM calculates early reflections (Direct + Specular up to ism_order).
        2. RayTracer calculates late reflections and diffuse energy (filtering out what ISM covered).
        3. Combine into RIRs and convolve.
        
        :param ism_order: Maximum reflection order for Image Source Method. Defaults to 2.
        """
        # Initialize outputs
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        all_paths = {} if record_paths else None

        valid_sources = [
            s for s in self.room.sources
            if s in self.source_audios
        ]

        if not valid_sources:
            print("No sources with assigned audio found in the room.")
            return receiver_outputs

        for source in valid_sources:
            if verbose:
                print(f"Hybrid Rendering Source: {source.name} (ISM Order: {ism_order})")

            # 1. Clear Histograms
            for rx in self.room.receivers:
                rx.energy_histogram = []

            # 2. Run ISM (Deterministic Early Reflections)
            # We need to target specific source.
            # ImageSourceEngine iterates all sources in room.
            # Hack: Temporarily set room sources.
            original_sources = self.room.sources
            self.room.sources = [source]
            
            if verbose:
                print("  Running Image Source Method...")
            self._ism_engine.run(max_order=ism_order, verbose=False)
            
            # 3. Run Ray Tracing (Stochastic Late/Diffuse)
            if verbose:
                print("  Running Ray Tracer...")
            
            # Pass min_ism_order to RayTracer
            paths = self._tracer.run(n_rays=n_rays, max_hops=max_hops,
                                     record_paths=record_paths,
                                     min_ism_order=ism_order)
            
            if record_paths and paths:
                all_paths.update(paths)
                
            self.room.sources = original_sources

            # 4. Generate RIR and Convolve (Same as AudioRenderer)
            source_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)

            for rx in self.room.receivers:
                # Generate RIR
                # Note: ISM entries are in histogram just like Ray entries.
                # generate_rir handles them uniformly.
                # However, ISM entries are precise. Ray entries are stochastic.
                # Ideally, ISM entries should be kept as strict impulses (fixed phase usually, or distance based).
                # generate_rir has 'random_phase'.
                # For ISM, phase is deterministic based on distance (and wall impedance phase shift, usually ignored or pi).
                # generate_rir assumes incoherent summing (random phase) or coherent (positive).
                # We should probably use 'interference=True' (random_phase=False) for better low freq, 
                # but generate_rir is simple.
                # Let's stick to standard generate_rir behavior.
                
                rir = generate_rir(rx.energy_histogram, fs=self.fs,
                                   duration=rir_duration,
                                   random_phase=not interference)

                processed = fftconvolve(source_audio * gain, rir, mode='full')

                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    current = receiver_outputs[rx.name]
                    if len(processed) > len(current):
                        padding = np.zeros(len(processed) - len(current))
                        current = np.concatenate((current, padding))
                        receiver_outputs[rx.name] = current
                    elif len(current) > len(processed):
                        padding = np.zeros(len(current) - len(processed))
                        processed = np.concatenate((processed, padding))
                    
                    receiver_outputs[rx.name] += processed

        # Normalize
        for name, audio in receiver_outputs.items():
            if audio is not None and np.max(np.abs(audio)) > 0:
                receiver_outputs[name] = audio / np.max(np.abs(audio))

        if record_paths:
            return receiver_outputs, all_paths
        return receiver_outputs

