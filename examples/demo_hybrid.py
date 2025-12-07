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
    HybridRenderer,
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


def main(mic_type='mono', output_dir='outputs/hybrid'):
    """
    Main function to run the hybrid simulation.
    """
    # 1. Define Small Room (Same as small room example)
    print("Creating small room (4m x 2m x 2.5m)...")
    room = Room.create_shoebox([4, 2, 2.5], materials={
        "floor": get_material("carpet"),
        "ceiling": get_material("plaster"),
        "walls": get_material("concrete")
    })

    # 2. Add Receiver
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

    # 5. Save layout visualization
    print(f"Saving room layout visualization to {output_dir}...")
    room.plot(os.path.join(output_dir, "hybrid_layout.png"), show=False)
    room.plot(os.path.join(output_dir, "hybrid_layout_2d.png"), show=False, view='2d')

    # 6. Setup Hybrid Renderer
    print("Initializing Hybrid Renderer...")
    renderer = HybridRenderer(room, fs=FS, temperature=20.0, humidity=50.0)

    # Assign Audio Files
    print("Assigning audio files...")
    base_path = "audios-trump-indextts15"
    # base_path = "audios-indextts"
    # base_path = "audios"

    if not os.path.exists(os.path.join(base_path, "speaker_1.wav")):
        print("Warning: Example audio files not found.")
        return

    renderer.set_source_audio(src1, os.path.join(base_path, "speaker_1.wav"), gain=1.0)
    renderer.set_source_audio(src2, os.path.join(base_path, "speaker_2.wav"), gain=1.0)
    renderer.set_source_audio(src_bg, os.path.join(base_path, "foreground.wav"), gain=0.1)

    # 7. Render using Hybrid Method
    # ism_order=2 means reflections of order 0, 1, 2 are handled by ISM.
    # RayTracer will skip specular reflections <= 2.
    print("Starting Hybrid Rendering pipeline (ISM Order 2 + Ray Tracing)...")

    outputs, _, rirs = renderer.render(
        n_rays=20000,       # Reduced ray count since early reflections are exact
        max_hops=40,
        rir_duration=1.5,
        record_paths=True,
        interference=False,
        ism_order=2         # Enable Hybrid Mode
    )

    # 8. Save Result
    mixed_audio = outputs[mic.name]

    if mixed_audio is not None:
        if mic_type == 'ambisonic':
            output_file = "hybrid_simulation_ambi.wav"
            mixed_audio /= np.max(np.abs(mixed_audio))
            wavfile.write(os.path.join(output_dir, output_file), FS, mixed_audio.astype(np.float32))
        else:
            output_file = "hybrid_simulation.wav"
            mixed_audio /= np.max(np.abs(mixed_audio))
            wavfile.write(os.path.join(output_dir, output_file), FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Hybrid simulation complete. Saved to {os.path.join(output_dir, output_file)}")

        # --- Generate new acoustic plots ---
        rir = rirs[mic.name]

        # 1. RT60 vs. Frequency
        rt_path = os.path.join(output_dir, "reverberation_time.png")
        plot_reverberation_time(rir, FS, filename=rt_path, show=False)

        # 2. Decay curve for one octave band (e.g., 1000 Hz)
        decay_path = os.path.join(output_dir, "decay_curve_1000hz.png")
        plot_decay_curve(rir, FS, band=1000, schroeder=False, filename=decay_path, show=False)

        # 3. Schroeder curve (broadband)
        schroeder_path = os.path.join(output_dir, "schroeder_curve.png")
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
            title=f"Output Spectrogram (Hybrid, {mic_type})",
            filename=os.path.join(output_dir, f"hybrid_spectrogram_{mic_type}.png"),
            show=False,
        )
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a hybrid simulation with different microphone types.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
                        help="Type of microphone to use ('mono' or 'ambisonic').")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Output directory for saving files.")
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir)
