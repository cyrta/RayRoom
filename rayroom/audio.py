import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
import os
from .core import RayTracer

class AudioRenderer:
    """
    Handles the audio rendering pipeline for a Room.
    Manages sources, audio data, ray tracing, RIR generation, convolution, and mixing.
    """
    def __init__(self, room, fs=44100):
        self.room = room
        self.fs = fs
        self.source_audios = {} # Map source_obj -> audio_array
        self.source_gains = {} # Map source_obj -> linear gain
        self._tracer = RayTracer(room)
        
    def set_source_audio(self, source, audio_data, gain=1.0):
        """
        Assign audio data to a Source object.
        
        Args:
            source: The Source object.
            audio_data: numpy array or path to wav file.
            gain: Linear gain factor for this source's audio (default 1.0).
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
            print(f"Warning: Sample rate mismatch {fs} vs {self.fs}. Playback speed will change. Resampling not fully implemented.")
            # TODO: Implement resampling
            
        return data
        
    def render(self, n_rays=20000, max_hops=50, rir_duration=2.0, verbose=True):
        """
        Run the full rendering pipeline.
        
        Returns:
             dict: {receiver_name: mixed_audio_array}
        """
        # Initialize outputs for each receiver
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        
        # Iterate over sources that have audio assigned
        # Only render sources that are in the room AND have audio
        valid_sources = [s for s in self.room.sources if s in self.source_audios]
        
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
            self._tracer.run(n_rays=n_rays, max_hops=max_hops) # Prints "Simulating Source..."
            self.room.sources = original_sources
            
            # 3. For each receiver, generate RIR and Convolve
            source_audio = self.source_audios[source]
            # Apply gain
            gain = self.source_gains.get(source, 1.0)
            
            for rx in self.room.receivers:
                # Generate RIR
                rir = generate_rir(rx.energy_histogram, fs=self.fs, duration=rir_duration)
                
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
                        receiver_outputs[rx.name] = current # Update reference
                    elif len(current) > len(processed):
                        padding = np.zeros(len(current) - len(processed))
                        processed = np.concatenate((processed, padding))
                        
                    receiver_outputs[rx.name] += processed
                    
        # Normalize final outputs
        for name, audio in receiver_outputs.items():
            if audio is not None and np.max(np.abs(audio)) > 0:
                receiver_outputs[name] = audio / np.max(np.abs(audio))
                
        return receiver_outputs

def generate_rir(energy_histogram, fs=44100, duration=None):
    """
    Convert energy histogram to a Room Impulse Response (RIR).
    
    Args:
        energy_histogram: List of (time, energy) tuples.
        fs: Sampling rate.
        duration: Length of IR in seconds. If None, fits to max time.
        
    Returns:
        np.array: The audio impulse response.
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
    
    # RIR amplitude ~ sqrt(Energy) * random_sign
    signs = np.random.choice([-1, 1], size=len(indices))
    amplitudes = signs * np.sqrt(energies)
    
    np.add.at(rir, indices, amplitudes)
        
    return rir

def convolve_and_mix(sources_data, fs=44100):
    """
    Legacy helper: Convolve source audios with their RIRs and mix.
    Kept for backward compatibility or manual usage.
    """
    max_len = 0
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
