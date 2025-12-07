import sys
import os
import argparse
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from rayroom import Room, Source, Receiver, AmbisonicReceiver, SpectralRenderer, get_material, Person

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs'):
    FS = 44100

    # 1. Define Small Room
    # Keeping it very small for FDTD speed in this example
    # 4m x 2m x 2.5m
    print("Creating room (4m x 2m x 2.5m)...")
    room = Room.create_shoebox([4, 2, 2.5], materials={
        "floor": get_material("heavy_curtain"),
        "ceiling": get_material("heavy_curtain"),
        "walls": get_material("wood")
    })

    # 2. Add Receiver
    mic_pos = [2, 1, 1.5]
    if mic_type == 'ambisonic':
        print("Using Ambisonic Receiver.")
        print(
            "Note: FDTD (low-frequency) part of Spectral rendering is MONO "
            "and will only be applied to the W channel."
        )
        mic = AmbisonicReceiver("AmbiMic", mic_pos, radius=0.02)
    else:
        print("Using Mono Receiver.")
        mic = Receiver("MonoMic", mic_pos, radius=0.15)
    room.add_receiver(mic)

    # 3. Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 4. Add Furniture (Blocker)
    # Note: FDTD Voxelizer in MVP ignores furniture for simplicity,
    # but RayTracer will see it.
    person = Person("Person", [1.2, 1, 0], height=1.7, width=0.5, depth=0.3, material_name="human")
    room.add_furniture(person)

    table = Person("Table", [3, 1, 0], height=0.8, width=0.8, depth=0.8, material_name="wood")
    room.add_furniture(table)

    # 5. Define Sources
    src1 = Source("Speaker 1", [0.5, 1.5, 1.5], power=1.0, orientation=[1, 0, 0], directivity="cardioid")
    src2 = Source("Speaker 2", [3.5, 1.5, 1.5], power=1.0, orientation=[-1, 0, 0], directivity="cardioid")
    src_bg = Source("Background Noise", [2, 0.5, 2.4], power=0.5)

    room.add_source(src1)
    room.add_source(src2)
    room.add_source(src_bg)

    # Save layout visualization
    print(f"Saving room layout visualization to {output_dir}...")
    room.plot(os.path.join(output_dir, "spectral_layout.png"), show=False)
    room.plot(os.path.join(output_dir, "spectral_layout_2d.png"), show=False, view='2d')

    # 6. Setup Spectral Renderer
    # Crossover at 500 Hz.
    # FDTD will handle < 500Hz (Diffraction important here).
    # Geometric will handle > 500Hz.
    print("Initializing Spectral Renderer (Crossover: 500 Hz)...")
    renderer = SpectralRenderer(room, fs=FS, crossover_freq=500.0)

    # 7. Assign Audio
    print("Assigning audio...")
    base_path = "audios-trump-indextts15"  # Uncomment if needed
    # base_path = "audios-indextts" # Uncomment if needed
    # base_path = "audios"

    audio_file_1 = os.path.join(base_path, "speaker_1.wav")
    audio_file_2 = os.path.join(base_path, "speaker_2.wav")

    if not os.path.exists(audio_file_1):
        print("Warning: Audio file not found. Creating dummy sine sweep.")
        # Create dummy audio
        t = np.linspace(0, 1, FS)
        audio = np.sin(2 * np.pi * 200 * t * t)  # Sweep
        renderer.set_source_audio(src1, audio, gain=1.0)
        renderer.set_source_audio(src2, audio, gain=1.0)
    else:
        renderer.set_source_audio(src1, audio_file_1, gain=1.0)
        if os.path.exists(audio_file_2):
            renderer.set_source_audio(src2, audio_file_2, gain=1.0)
        else:
            # Use same for src2 if src2 wav missing
            renderer.set_source_audio(src2, audio_file_1, gain=1.0)

    audio_file_bg = os.path.join(base_path, "foreground.wav")
    if os.path.exists(audio_file_bg):
        renderer.set_source_audio(src_bg, audio_file_bg, gain=0.1)

    # 8. Render
    print("Starting Spectral Rendering pipeline...")
    print("Phase 1: HF (Geometric) + Phase 2: LF (FDTD)")
    print("Note: FDTD step may take time...")

    outputs, _ = renderer.render(
        n_rays=10000,
        max_hops=10,
        rir_duration=0.2,  # Short duration for demo speed
        record_paths=True,
        ism_order=1
    )

    # 9. Save Result
    mixed_audio = outputs[mic.name]

    if mixed_audio is not None:
        if mic_type == 'ambisonic':
            output_file = "spectral_simulation_ambi.wav"
            mixed_audio /= np.max(np.abs(mixed_audio))
            wavfile.write(os.path.join(output_dir, output_file), FS, mixed_audio.astype(np.float32))
        else:
            output_file = "spectral_simulation.wav"
            # Normalize
            mixed_audio /= np.max(np.abs(mixed_audio))
            wavfile.write(os.path.join(output_dir, output_file), FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Simulation complete. Saved to {os.path.join(output_dir, output_file)}")

        # Plot Spectrogram
        plot_audio = mixed_audio[:, 0] if mic_type == 'ambisonic' else mixed_audio
        plt.figure(figsize=(10, 4))
        plt.specgram(plot_audio, Fs=FS, NFFT=1024, noverlap=512)
        plt.title(f"Output Spectrogram (Spectral Method, {mic_type})")
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(os.path.join(output_dir, f"spectral_spectrogram_{mic_type}.png"))
        print(f"Saved spectral_spectrogram_{mic_type}.png to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a spectral simulation with different microphone types.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
                        help="Type of microphone to use ('mono' or 'ambisonic').")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Output directory for saving files.")
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir)
