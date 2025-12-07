import os
import sys
import argparse
import numpy as np

from rayroom import (
    SpectralRenderer,
)
from rayroom.effects import presets
from demo_utils import (
    create_demo_room,
    generate_layouts,
    process_effects_and_save,
    DEFAULT_SAMPLING_RATE,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs/spectral', effects=None):
    """
    Main function to run the spectral simulation.
    """
    # 1. Define Small Room
    room, sources, mic = create_demo_room(mic_type)
    src1 = sources["src1"]
    src2 = sources["src2"]
    src_bg = sources["src_bg"]

    # 3. Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save layout visualization
    generate_layouts(room, output_dir, "spectral")

    # 6. Setup Spectral Renderer
    # Crossover at 500 Hz.
    # FDTD will handle < 500Hz (Diffraction important here).
    # Geometric will handle > 500Hz.
    print("Initializing Spectral Renderer (Crossover: 500 Hz)...")
    renderer = SpectralRenderer(room, fs=DEFAULT_SAMPLING_RATE, crossover_freq=500.0)

    # 7. Assign Audio
    print("Assigning audio...")
    base_path = "audios-trump-indextts15"  # Uncomment if needed
    # base_path = "audios-indextts" # Uncomment if needed
    # base_path = "audios"

    audio_file_1 = os.path.join(base_path, "speaker_1.wav")
    audio_file_2 = os.path.join(base_path, "speaker_2.wav")

    if not os.path.exists(audio_file_1):
        print("Warning: Audio file not found. Creating dummy sine sweep.")
        # Create dummy audio
        t = np.linspace(0, 1, DEFAULT_SAMPLING_RATE)
        audio = np.sin(2 * np.pi * 200 * t * t)  # Sweep
        renderer.set_source_audio(src1, audio, gain=1.0)
        renderer.set_source_audio(src2, audio, gain=1.0)
    else:
        renderer.set_source_audio(src1, audio_file_1, gain=1.0)
        if os.path.exists(audio_file_2):
            renderer.set_source_audio(src2, audio_file_2, gain=1.0)
        else:
            # Use same for src2 if src2 wav missing
            renderer.set_source_audio(src2, audio_file_1, gain=1.0)

    audio_file_bg = os.path.join(base_path, "foreground.wav")
    if os.path.exists(audio_file_bg):
        renderer.set_source_audio(src_bg, audio_file_bg, gain=0.1)

    # 8. Render
    print("Starting Spectral Rendering pipeline...")
    print("Phase 1: HF (Geometric) + Phase 2: LF (FDTD)")
    print("Note: FDTD step may take time...")

    outputs, rirs = renderer.render(
        n_rays=10000,
        max_hops=10,
        rir_duration=1.0,  # Short duration for demo speed
        record_paths=False,
        ism_order=1,
        show_path_plot=False
    )

    # 9. Save Result
    mixed_audio = outputs[mic.name]
    rir = rirs[mic.name]

    if mixed_audio is not None:
        process_effects_and_save(mixed_audio, rir, mic.name, mic_type, DEFAULT_SAMPLING_RATE, output_dir, "spectral", effects)
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a spectral simulation with different microphone types.")
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
