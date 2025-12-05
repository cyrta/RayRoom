import sys
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from rayroom import Room, Source, Receiver, AudioRenderer, get_material, Person

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    FS = 44100

    # 1. Define Small Room (8 square meters -> e.g., 4m x 2m or 2.83m x 2.83m)
    # Using 4m x 2m x 2.5m height
    print("Creating small room (4m x 2m x 2.5m)...")
    room = Room.create_shoebox([4, 2, 2.5], materials={
        "floor": get_material("wood"),
        "ceiling": get_material("plaster"),
        "walls": get_material("brick")
    })

    # 2. Add Receiver (Microphone) - centered
    mic = Receiver("Microphone", [2, 1, 1.5], radius=0.15)
    room.add_receiver(mic)

    # 3. Add Furniture
    # Add a Person (blocker) between source 1 and mic
    # Source 1 will be at (0.5, 1)
    # Mic at (2, 1)
    # Person at (1.2, 1)
    person = Person("Person", [1.2, 1, 0], height=1.7, width=0.5, depth=0.3, material_name="human")
    room.add_furniture(person)

    # Add a Table
    table = Person("Table", [3, 1, 0], height=0.8, width=0.8, depth=0.8, material_name="wood")
    room.add_furniture(table)

    # 4. Define Sources
    # Speaker 1 at one end
    src1 = Source("Speaker 1", [0.5, 1, 1.5], power=1.0, orientation=[1, 0, 0], directivity="cardioid")
    # Speaker 2 at the other end
    src2 = Source("Speaker 2", [3.5, 1, 1.5], power=1.0, orientation=[-1, 0, 0], directivity="cardioid")
    # Background noise near ceiling
    src_bg = Source("Background Noise", [2, 0.5, 2.4], power=0.5)

    room.add_source(src1)
    room.add_source(src2)
    room.add_source(src_bg)

    # Save layout visualization
    print("Saving room layout visualization to small_room_layout.png...")
    room.plot("small_room_layout.png", show=False)
    room.plot("small_room_layout_2d.png", show=False, view='2d')

    # 5. Setup Renderer
    renderer = AudioRenderer(room, fs=FS, temperature=20.0, humidity=50.0)

    # Assign Audio Files
    print("Assigning audio files...")
    base_path = "examples/audios"

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

    # 6. Render
    print("Starting rendering pipeline...")
    # Smaller room might need fewer rays or hops, but keeping high for quality
    outputs, paths_data = renderer.render(
        n_rays=30000,
        max_hops=40,
        rir_duration=1.5,
        record_paths=True,
        interference=False
    )  # With interference=False, the audio is much cleaner.

    # 7. Save Result
    mixed_audio = outputs["Microphone"]

    if mixed_audio is not None:
        output_file = "small_room_simulation.wav"
        wavfile.write(output_file, FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Simulation complete. Saved to {output_file}")

        # Plot Spectrogram
        plt.figure(figsize=(10, 4))
        plt.specgram(mixed_audio, Fs=FS, NFFT=1024, noverlap=512)
        plt.title("Output Spectrogram (Small Room)")
        plt.colorbar(format='%+2.0f dB')
        plt.savefig("small_room_spectrogram.png")
        print("Saved small_room_spectrogram.png")
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    main()
