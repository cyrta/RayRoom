import os
import sys
import argparse

from rayroom import (
    HybridRenderer,
)
from rayroom.effects import presets
from demo_utils import (
    create_demo_room,
    generate_layouts,
    process_effects_and_save,
    DEFAULT_SAMPLING_RATE,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs/hybrid', effects=None):
    """
    Main function to run the hybrid simulation.
    """
    # 1. Define Small Room (Same as small room example)
    room, sources, mic = create_demo_room(mic_type)
    src1 = sources["src1"]
    src2 = sources["src2"]
    src_bg = sources["src_bg"]

    # 3. Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 5. Save layout visualization
    generate_layouts(room, output_dir, "hybrid")

    # 6. Setup Hybrid Renderer
    print("Initializing Hybrid Renderer...")
    renderer = HybridRenderer(room, fs=DEFAULT_SAMPLING_RATE, temperature=20.0, humidity=50.0)

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
        ism_order=2,         # Enable Hybrid Mode
        show_path_plot=False
    )

    # 8. Save Result
    mixed_audio = outputs[mic.name]
    rir = rirs[mic.name]

    if mixed_audio is not None:
        process_effects_and_save(
            mixed_audio, rir, mic.name, mic_type, DEFAULT_SAMPLING_RATE, output_dir, "hybrid", effects
        )
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a hybrid simulation with different microphone types.")
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
