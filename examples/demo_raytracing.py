import os
import sys
import argparse
import numpy as np
from scipy.io import wavfile
import json

from rayroom import (
    Room,
    Source,
    Receiver,
    AmbisonicReceiver,
    RaytracingRenderer,
    get_material,
    Person,
)
from rayroom.analytics.acoustics import (
    calculate_clarity,
    calculate_drr,
)
from rayroom.room.visualize import (
    plot_reverberation_time,
    plot_decay_curve,
    plot_spectrogram,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


FS = 44100


def main(mic_type='mono', output_dir='outputs'):
    # 1. Define Room for Raytracing (8 square meters -> e.g., 4m x 2m or 2.83m x 2.83m)
    # Using 4m x 2m x 2.5m height
    print("Creating room for raytracing demo (4m x 2m x 2.5m)...")
    room = Room.create_shoebox([4, 2, 2.5], materials={
        "floor": get_material("carpet"),
        "ceiling": get_material("plaster"),
        "walls": get_material("concrete")
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

    # 6. Setup Ray-tracing Renderer
    print("Initializing Ray-tracing Renderer...")
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
    print("Starting Ray-tracing Rendering pipeline...")
    outputs, rirs = renderer.render(
        n_rays=20000,
        max_hops=50,
        rir_duration=1.5
    )

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

        # --- Generate new acoustic plots ---
        rir = rirs[mic.name]
        # For ambisonic, use the first channel (W) for analysis
        if mic_type == 'ambisonic' and rir.ndim > 1:
            rir = rir[:, 0]

        # 1. RT60 vs. Frequency
        rt_path = os.path.join(output_dir, f"reverberation_time_{mic_type}.png")
        plot_reverberation_time(rir, FS, filename=rt_path, show=False)

        # 2. Decay curve for one octave band (e.g., 1000 Hz)
        decay_path = os.path.join(output_dir, f"decay_curve_1000hz_{mic_type}.png")
        plot_decay_curve(rir, FS, band=1000, schroeder=False, filename=decay_path, show=False)

        # 3. Schroeder curve (broadband)
        schroeder_path = os.path.join(output_dir, f"schroeder_curve_{mic_type}.png")
        plot_decay_curve(rir, FS, schroeder=True, filename=schroeder_path, show=False)
        # --- End of new plots ---

        # --- Calculate and print new metrics ---
        c50 = calculate_clarity(rir, FS, 50)
        c80 = calculate_clarity(rir, FS, 80)
        drr = calculate_drr(rir, FS)

        print("\nAcoustic Metrics:")
        print(f"  - C50 (Speech Clarity): {c50:.2f} dB")
        print(f"  - C80 (Music Clarity):  {c80:.2f} dB")
        print(f"  - DRR (Direct-to-Reverberant Ratio): {drr:.2f} dB")

        # --- Save metrics to JSON ---
        metrics = {
            "c50_db": c50,
            "c80_db": c80,
            "drr_db": drr,
        }
        metrics_path = os.path.join(output_dir, "acoustic_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Acoustic metrics saved to {metrics_path}")
        # --- End of new metrics ---

        # Plot Spectrogram
        plot_audio = mixed_audio[:, 0] if mic_type == 'ambisonic' else mixed_audio
        plot_spectrogram(
            plot_audio,
            FS,
            title=f"Output Spectrogram (Raytracing, {mic_type})",
            filename=os.path.join(
                output_dir, f"raytracing_spectrogram_{mic_type}.png"
            ),
            show=False,
        )

        # Analytics
        rir = rirs[mic.name]
        plot_reverberation_time(
            rir, FS, filename=os.path.join(output_dir, "reverberation_time.png"), show=False
        )

        # 2. Decay curve for one octave band (e.g., 1000 Hz)
        decay_path = os.path.join(output_dir, f"decay_curve_1000hz_{mic_type}.png")
        plot_decay_curve(rir, FS, band=1000, schroeder=False, filename=decay_path, show=False)

        # 3. Schroeder curve (broadband)
        schroeder_path = os.path.join(output_dir, f"schroeder_curve_{mic_type}.png")
        plot_decay_curve(rir, FS, schroeder=True, filename=schroeder_path, show=False)
        # --- End of new plots ---
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a raytracing room simulation with different microphone types.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
                        help="Type of microphone to use ('mono' or 'ambisonic').")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Output directory for saving files.")
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir)
