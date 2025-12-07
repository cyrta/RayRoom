import os
import sys
import argparse
import numpy as np
from scipy.io import wavfile
import json

from rayroom import Room, Source, Receiver, AmbisonicReceiver, RadiosityRenderer, get_material, Person
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


def main(mic_type='mono', output_dir='outputs/radiosity'):
    FS = 44100

    # 1. Define Room
    print("Creating room (4m x 2m x 2.5m)...")
    room = Room.create_shoebox([4, 2, 2.5], materials={
        "floor": get_material("carpet"),
        "ceiling": get_material("plaster"),
        "walls": get_material("concrete")
    })

    # 2. Add Receiver
    mic_pos = [2, 1, 1.5]
    if mic_type == 'ambisonic':
        mic = AmbisonicReceiver("AmbiMic", mic_pos, radius=0.02)
    else:
        mic = Receiver("MonoMic", mic_pos, radius=0.15)
    room.add_receiver(mic)

    # 3. Create output directory
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

    # Save layout visualization
    print(f"Saving room layout visualization to {output_dir}...")
    room.plot(os.path.join(output_dir, "radiosity_layout.png"), show=False)
    room.plot(os.path.join(output_dir, "radiosity_layout_2d.png"), show=False, view='2d')

    # 6. Setup Radiosity Renderer
    print("Initializing Radiosity Renderer...")
    renderer = RadiosityRenderer(room, fs=FS)

    # 7. Assign Audio Files
    print("Assigning audio files...")
    audio_path = os.path.join(os.path.dirname(__file__), "audios", "speaker_1.wav")
    if not os.path.exists(audio_path):
        print(f"Warning: Example audio file not found at {audio_path}")
        return
    renderer.set_source_audio(src1, audio_path, gain=1.0)
    audio_path_2 = os.path.join(os.path.dirname(__file__), "audios", "speaker_2.wav")
    if not os.path.exists(audio_path_2):
        print(f"Warning: Example audio file not found at {audio_path_2}")
        return
    renderer.set_source_audio(src2, audio_path_2, gain=1.0)

    audio_path_bg = os.path.join(os.path.dirname(__file__), "audios", "foreground.wav")
    if not os.path.exists(audio_path_bg):
        print(f"Warning: Example audio file not found at {audio_path_bg}")
    else:
        renderer.set_source_audio(src_bg, audio_path_bg, gain=0.1)

    # 8. Render
    print("Starting Radiosity Rendering pipeline (ISM Order 2 + Radiosity)...")
    outputs, rirs = renderer.render(
        ism_order=2,
        rir_duration=1.5
    )

    # 9. Save Result
    mixed_audio = outputs[mic.name]

    if mixed_audio is not None:
        output_file = f"radiosity_simulation_{mic_type}.wav"

        # Normalize
        mixed_audio /= np.max(np.abs(mixed_audio))

        if mic_type == 'ambisonic':
            wavfile.write(os.path.join(output_dir, output_file), FS, mixed_audio.astype(np.float32))
        else:
            wavfile.write(os.path.join(output_dir, output_file), FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Radiosity simulation complete. Saved to {os.path.join(output_dir, output_file)}")

        # Plot Spectrogram
        plot_audio = mixed_audio[:, 0] if mic_type == 'ambisonic' else mixed_audio
        plot_spectrogram(
            plot_audio,
            FS,
            title=f"Output Spectrogram (Radiosity, {mic_type})",
            filename=os.path.join(output_dir, f"radiosity_spectrogram_{mic_type}.png"),
            show=False,
        )

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

    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a radiosity simulation.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'], help="Microphone type.")
    parser.add_argument('--output_dir', type=str, default='outputs/radiosity', help="Output directory.")
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir)
