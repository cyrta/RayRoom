import sys
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from rayroom import Room, Source, Receiver, SpectralRenderer, get_material, Person

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    FS = 44100

    # 1. Define Small Room
    # Keeping it very small for FDTD speed in this example
    # 4m x 2m x 2.5m
    print("Creating room (4m x 2m x 2.5m)...")
    room = Room.create_shoebox([4, 2, 2.5], materials={
        "floor": get_material("wood"),
        "ceiling": get_material("plaster"),
        "walls": get_material("brick")
    })

    # 2. Add Receiver
    mic = Receiver("Microphone", [2, 1, 1.5], radius=0.15)
    room.add_receiver(mic)

    # 3. Add Furniture (Blocker)
    # Note: FDTD Voxelizer in MVP ignores furniture for simplicity, 
    # but RayTracer will see it.
    person = Person("Person", [1.2, 1, 0], height=1.7, width=0.5, depth=0.3, material_name="human")
    room.add_furniture(person)

    table = Person("Table", [3, 1, 0], height=0.8, width=0.8, depth=0.8, material_name="wood")
    room.add_furniture(table)

    # 4. Define Sources
    src1 = Source("Speaker 1", [0.5, 1.5, 1.5], power=1.0, orientation=[1, 0, 0], directivity="cardioid")
    src2 = Source("Speaker 2", [3.5, 1.5, 1.5], power=1.0, orientation=[-1, 0, 0], directivity="cardioid")
    
    room.add_source(src1)
    room.add_source(src2)

    # 5. Setup Spectral Renderer
    # Crossover at 500 Hz. 
    # FDTD will handle < 500Hz (Diffraction important here).
    # Geometric will handle > 500Hz.
    print("Initializing Spectral Renderer (Crossover: 500 Hz)...")
    renderer = SpectralRenderer(room, fs=FS, crossover_freq=500.0)

    # Assign Audio
    print("Assigning audio...")
    base_path = "examples/audios-indextts" # Uncomment if needed
    # base_path = "examples/audios"
    
    audio_file_1 = os.path.join(base_path, "speaker_1.wav")
    audio_file_2 = os.path.join(base_path, "speaker_2.wav")
    
    if not os.path.exists(audio_file_1):
        print("Warning: Audio file not found. Creating dummy sine sweep.")
        # Create dummy audio
        t = np.linspace(0, 1, FS)
        audio = np.sin(2 * np.pi * 200 * t * t) # Sweep
        renderer.set_source_audio(src1, audio, gain=1.0)
        renderer.set_source_audio(src2, audio, gain=1.0)
    else:
        renderer.set_source_audio(src1, audio_file_1, gain=1.0)
        if os.path.exists(audio_file_2):
            renderer.set_source_audio(src2, audio_file_2, gain=1.0)
        else:
             # Use same for src2 if src2 wav missing
            renderer.set_source_audio(src2, audio_file_1, gain=1.0)

    # 6. Render
    print("Starting Spectral Rendering pipeline...")
    print("Phase 1: HF (Geometric) + Phase 2: LF (FDTD)")
    print("Note: FDTD step may take time...")
    
    outputs, _ = renderer.render(
        n_rays=10000,
        max_hops=30,
        rir_duration=0.5, # Short duration for demo speed
        record_paths=True,
        ism_order=1
    )

    # 7. Save Result
    mixed_audio = outputs["Microphone"]

    if mixed_audio is not None:
        output_file = "spectral_simulation.wav"
        # Normalize
        mixed_audio /= np.max(np.abs(mixed_audio))
        wavfile.write(output_file, FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Simulation complete. Saved to {output_file}")

        # Plot Spectrogram
        plt.figure(figsize=(10, 4))
        plt.specgram(mixed_audio, Fs=FS, NFFT=1024, noverlap=512)
        plt.title("Output Spectrogram (Spectral Method)")
        plt.colorbar(format='%+2.0f dB')
        plt.savefig("spectral_spectrogram.png")
        print("Saved spectral_spectrogram.png")

if __name__ == "__main__":
    main()
