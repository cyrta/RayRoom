import os
import sys
import argparse

from rayroom import RadiosityRenderer
from rayroom.effects import presets
from demo_utils import (
    create_demo_room,
    generate_layouts,
    process_effects_and_save,
    DEFAULT_SAMPLING_RATE,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs/radiosity', effects=None):
    # 1. Define Room
    room, sources, mic = create_demo_room(mic_type)
    src1 = sources["src1"]
    src2 = sources["src2"]
    src_bg = sources["src_bg"]

    # 3. Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save layout visualization
    generate_layouts(room, output_dir, "radiosity")

    # 6. Setup Radiosity Renderer
    print("Initializing Radiosity Renderer...")
    renderer = RadiosityRenderer(room, fs=DEFAULT_SAMPLING_RATE)

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
    rir = rirs[mic.name]

    if mixed_audio is not None:
        process_effects_and_save(mixed_audio, rir, mic.name, mic_type, DEFAULT_SAMPLING_RATE, output_dir, "radiosity", effects)
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a radiosity simulation.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'], help="Microphone type.")
    parser.add_argument('--output_dir', type=str, default='outputs/radiosity', help="Output directory.")
    parser.add_argument(
        '--effects',
        type=str,
        nargs='*',
        default=None,
        choices=list(presets.EFFECTS.keys())+["original"],
        help="Apply a post-processing effect to the output audio."
    )
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir, effects=args.effects)
