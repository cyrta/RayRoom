import os
import sys
import argparse

from rayroom import (
    RaytracingRenderer,
)
from rayroom.effects import presets
from demo_utils import (
    create_demo_room,
    generate_layouts,
    process_effects_and_save,
    DEFAULT_SAMPLING_RATE,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs', effects=None):
    # 1. Define Room for Raytracing (8 square meters -> e.g., 4m x 2m or 2.83m x 2.83m)
    # Using 4m x 2m x 2.5m height
    room, sources, mic = create_demo_room(mic_type)
    src1 = sources["src1"]
    src2 = sources["src2"]
    src_bg = sources["src_bg"]

    # 3. Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 5. Save layout visualization
    generate_layouts(room, output_dir, "raytracing")

    # 6. Setup Ray-tracing Renderer
    print("Initializing Ray-tracing Renderer...")
    renderer = RaytracingRenderer(room, fs=DEFAULT_SAMPLING_RATE, temperature=20.0, humidity=50.0)

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
    mixed_audio = outputs[mic.name]
    rir = rirs[mic.name]

    if mixed_audio is None:
        print("Error: No audio output generated.")
        return

    process_effects_and_save(
        mixed_audio, rir, mic.name, mic_type, DEFAULT_SAMPLING_RATE, output_dir, "raytracing", effects
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a raytracing room simulation with different microphone types.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
                        help="Type of microphone to use ('mono' or 'ambisonic').")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Output directory for saving files.")
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
