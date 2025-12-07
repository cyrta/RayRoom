import os
import sys
import numpy as np
from scipy.io import wavfile

from rayroom import Room, Source, Receiver, HybridRenderer, get_material
from rayroom.room.visualize import plot_reverberation_time, plot_decay_curve

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    # Create output directory if it doesn't exist
    output_dir = os.path.join("outputs", "polygon+hybrid")
    os.makedirs(output_dir, exist_ok=True)
    FS = 44100

    # Define a U-shaped room using corners
    corners = [
        (0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)
    ]

    materials = {
        "floor": get_material("carpet"),
        "ceiling": get_material("plaster"),
        "walls": get_material("concrete"),
    }

    room = Room.create_from_corners(corners, height=3.0, materials=materials)

    # Add Source in one leg of U
    source1 = Source("Speaker 1", [2.0, 0.5, 1.5])  # Moved to the center bottom
    source2 = Source("Speaker 2", [3.5, 1.0, 1.5])
    room.add_source(source1)
    room.add_source(source2)

    # Add Receiver in the other leg of U
    receiver = Receiver("Mic", [3.5, 2.5, 1.5], radius=0.2)
    room.add_receiver(receiver)

    # Save 2D layout plot
    layout_path = os.path.join(output_dir, "u_shape_layout_2d.png")
    print(f"Saving 2D layout plot to {layout_path}...")
    room.plot(filename=layout_path, view='2d', show=False)
    print("Saved 2D layout plot.")

    print("Running simulation for U-shaped room...")
    renderer = HybridRenderer(room, fs=FS)
    base_path = "audios"
    renderer.set_source_audio(source1, os.path.join(base_path, "speaker_1.wav"), gain=1.0)
    renderer.set_source_audio(source2, os.path.join(base_path, "speaker_2.wav"), gain=1.0)

    outputs, rirs = renderer.render(
        n_rays=20000,
        max_hops=40,
        rir_duration=1.5,
        ism_order=2
    )

    mixed_audio = outputs[receiver.name]
    if mixed_audio is not None:
        output_file = "hybrid_u_shape_simulation.wav"
        mixed_audio /= np.max(np.abs(mixed_audio))
        wavfile.write(os.path.join(output_dir, output_file), FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Hybrid simulation complete. Saved to {os.path.join(output_dir, output_file)}")

        # --- Generate new acoustic plots ---
        rir = rirs[receiver.name]

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


if __name__ == "__main__":
    main()
