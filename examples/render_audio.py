import sys
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from rayroom import Room, Source, Receiver, AudioRenderer, get_material, Person

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    FS = 44100

    # 1. Define Room
    room = Room.create_shoebox([6, 5, 3], materials={
        "floor": get_material("wood"),
        "ceiling": get_material("plaster"),
        "walls": get_material("brick")
    })

    # 2. Add Receiver (Microphone)
    mic = Receiver("Microphone", [3, 2.5, 1.5], radius=0.15)
    room.add_receiver(mic)

    # 3. Add Furniture
    # Add a Person (blocker)
    person = Person("Person", [2, 2, 0], height=1.7, width=0.5, depth=0.3, material_name="human")
    room.add_furniture(person)

    # Add a Table (using Person class as a box proxy with wood material)
    table = Person("Table", [4, 2.5, 0], height=0.8, width=1.2, depth=0.8, material_name="wood")
    room.add_furniture(table)

    # 4. Define Sources
    # Speaker 1 points towards the center/receiver (Cardioid)
    src1 = Source("Speaker 1", [1, 1, 1.5], power=1.0, orientation=[1, 0.5, 0], directivity="cardioid")
    # Speaker 2 points towards center (Cardioid)
    src2 = Source("Speaker 2", [5, 4, 1.5], power=1.0, orientation=[-1, -0.5, 0], directivity="cardioid")
    # AC is omnidirectional (default)
    src_ac = Source("AC", [3, 0.5, 2.8], power=0.5)

    room.add_source(src1)
    room.add_source(src2)
    room.add_source(src_ac)

    # Save layout visualization (NOW that sources are added)
    print("Saving room layout visualization to audio_room_layout.png...")
    room.plot("audio_room_layout.png", show=False)

    print("Saving room 2D layout to audio_room_layout_2d.png...")
    room.plot("audio_room_layout_2d.png", show=False, view='2d')

    # 5. Setup Renderer
    # Standard conditions: 20C, 50% Humidity
    renderer = AudioRenderer(room, fs=FS, temperature=20.0, humidity=50.0)

    # Assign Audio Files
    print("Assigning audio files...")
    base_path = "examples/audios"
    renderer.set_source_audio(src1, os.path.join(base_path, "speaker_1.wav"), gain=1.0)
    renderer.set_source_audio(src2, os.path.join(base_path, "speaker_2.wav"), gain=1.0)
    # Reduce AC volume to be background noise (gain=0.1)
    renderer.set_source_audio(src_ac, os.path.join(base_path, "foreground.wav"), gain=0.1)

    # 6. Render
    print("Starting rendering pipeline...")
    # This will run ray tracing for each source and mix the results
    outputs, paths_data = renderer.render(n_rays=30000, max_hops=40, rir_duration=2.0, record_paths=True)

    # 7. Save Result
    mixed_audio = outputs["Microphone"]

    if mixed_audio is not None:
        output_file = "room_simulation_mix_renderer.wav"
        wavfile.write(output_file, FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Simulation complete. Saved to {output_file}")

        # Plot Spectrogram
        plt.figure(figsize=(10, 4))
        plt.specgram(mixed_audio, Fs=FS, NFFT=1024, noverlap=512)
        plt.title("Output Spectrogram")
        plt.colorbar(format='%+2.0f dB')
        plt.savefig("output_spectrogram_renderer.png")
        print("Saved output_spectrogram_renderer.png")
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    main()
