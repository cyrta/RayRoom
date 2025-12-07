import os
import sys
import argparse
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from rayroom import Room, Source, Receiver, AmbisonicReceiver, RaytracingRenderer, get_material, Person

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs'):
    FS = 44100

    # 1. Define Room for Raytracing (8 square meters -> e.g., 4m x 2m or 2.83m x 2.83m)
    # Using 4m x 2m x 2.5m height
    print("Creating room for raytracing demo (4m x 2m x 2.5m)...")
    room = Room.create_shoebox([4, 2, 2.5], materials={
        "floor": get_material("heavy_curtain"),
        "ceiling": get_material("heavy_curtain"),
        "walls": get_material("wood")
    })

    # 2. Add Receiver (Microphone) - centered
    mic_pos = [2, 1, 1.5]
    if mic_type == 'ambisonic':
        print("Using Ambisonic Receiver.")
        mic = AmbisonicReceiver("AmbiMic", mic_pos, radius=0.02)
    else:
        print("Using Mono Receiver.")
        mic = Receiver("MonoMic", mic_pos, radius=0.15)
    room.add_receiver(mic)

    # 3. Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 4. Add Furniture
    # Add a Person (blocker) between source 1 and mic
    # Source 1 will be at (0.5, 1)
    # Mic at (2, 1)
    # Person at (1.2, 1)
    person = Person("Person", [1.2, 1, 0], height=1.7, width=0.5, depth=0.3, material_name="human")
    room.add_furniture(person)

    # Add a Table
    table = Person("Table", [3, 1, 0], height=0.8, width=0.8, depth=0.8, material_name="wood")
    room.add_furniture(table)

    # 5. Define Sources
    # Speaker 1 at one end
    src1 = Source("Speaker 1", [0.5, 1.5, 1.5], power=1.0, orientation=[1, 0, 0], directivity="cardioid")
    # Speaker 2 at the other end
    src2 = Source("Speaker 2", [3.5, 1.5, 1.5], power=1.0, orientation=[-1, 0, 0], directivity="cardioid")
    # Background noise near ceiling
    src_bg = Source("Background Noise", [2, 0.5, 2.4], power=0.5)

    room.add_source(src1)
    room.add_source(src2)
    room.add_source(src_bg)

    # 5. Save layout visualization
    print(f"Saving room layout visualization to {output_dir}...")
    room.plot(os.path.join(output_dir, "raytracing_layout.png"), show=False)
    room.plot(os.path.join(output_dir, "raytracing_layout_2d.png"), show=False, view='2d')

    # 6. Setup Renderer
    renderer = RaytracingRenderer(room, fs=FS, temperature=20.0, humidity=50.0)

    # 7. Assign Audio Files
    print("Assigning audio files...")
    # build the path to the audio files folder relative to this script file
    base_path = os.path.join(os.path.dirname(__file__), "audios-trump-indextts15")
    # base_path = "audios-indextts"
    # base_path = "audios"

    # Check if audio files exist, otherwise use placeholders or warnings
    if not os.path.exists(os.path.join(base_path, "speaker_1.wav")):
        print(
            "Warning: Example audio files not found. "
            "Please ensure 'examples/audios/' has speaker_1.wav, speaker_2.wav, foreground.wav"
        )
        return

    renderer.set_source_audio(src1, os.path.join(base_path, "speaker_1.wav"), gain=1.0)
    renderer.set_source_audio(src2, os.path.join(base_path, "speaker_2.wav"), gain=1.0)
    renderer.set_source_audio(src_bg, os.path.join(base_path, "foreground.wav"), gain=0.1)

    # 8. Render
    print("Starting rendering pipeline...")
    # Smaller room might need fewer rays or hops, but keeping high for quality
    outputs, paths_data = renderer.render(
        n_rays=30000,
        max_hops=40,
        rir_duration=1.5,
        record_paths=True,
        interference=False
    )  # With interference=False, the audio is much cleaner.

    # 9. Save Result
    print(
        f"DEBUG: Receiver '{mic.name}' recorded "
        f"{len(mic.amplitude_histogram if mic_type == 'mono' else mic.w_histogram)} "
        "hits during simulation."
    )
    mixed_audio = outputs[mic.name]

    if mixed_audio is not None:
        # Debugging output to check the audio signal before saving
        max_abs_val = np.max(np.abs(mixed_audio))
        print(f"Max absolute value in mixed_audio before saving: {max_abs_val}")
        if max_abs_val < 1e-9:  # Using a small threshold for floating point
            print("WARNING: The rendered audio is silent or extremely quiet.")

        if mic_type == 'ambisonic':
            output_file = "raytracing_simulation_ambi.wav"
            # Ambisonic is 4-channel, float format.
            # Normalization is handled by the RaytracingRenderer.
            wavfile.write(os.path.join(output_dir, output_file), FS, mixed_audio.astype(np.float32))

        else:
            output_file = "raytracing_simulation.wav"
            # Normalization is handled by the RaytracingRenderer.
            wavfile.write(os.path.join(output_dir, output_file), FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Simulation complete. Saved to {os.path.join(output_dir, output_file)}")

        # Plot Spectrogram (of the W channel for ambisonic)
        plot_audio = mixed_audio[:, 0] if mic_type == 'ambisonic' and mixed_audio.ndim > 1 else mixed_audio
        plt.figure(figsize=(10, 4))
        plt.specgram(plot_audio, Fs=FS, NFFT=1024, noverlap=512)
        plt.title(f"Output Spectrogram (Raytracing, {mic_type})")
        plt.colorbar(format='%+2.0f dB')
        plt.savefig(os.path.join(output_dir, f"raytracing_spectrogram_{mic_type}.png"))
        print(f"Saved raytracing_spectrogram_{mic_type}.png to {output_dir}")
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a raytracing room simulation with different microphone types.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
                        help="Type of microphone to use ('mono' or 'ambisonic').")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Output directory for saving files.")
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir)
