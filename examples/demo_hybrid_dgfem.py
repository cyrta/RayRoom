import argparse
import os
import sys

import numpy as np
import soundfile as sf
from rayroom.engines.hybrid_dgfem.hybrid_dgfem import HybridDGFEMMethod, HybridDGFEMConfig
from rayroom.engines.dgfem.dgfem import DGFEMConfig
from demo_utils import (
    create_demo_room,
    generate_layouts,
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main(mic_type='mono', output_dir='outputs'):
    """
    A demonstration of the Hybrid DG-FEM engine.
    """
    print("="*70)
    print("Hybrid DG-FEM Acoustic Solver Demo")
    print("="*70)

    # 1. Create a room from demo_utils
    room, _, mic = create_demo_room(mic_type=mic_type)
    room.fs = 16000  # Lower fs for DGFEM performance

    # 2. Create output directory
    output_path = os.path.join(output_dir, 'hybrid_dgfem', mic_type)
    os.makedirs(output_path, exist_ok=True)

    # 3. Save layout visualization
    generate_layouts(room, output_path, "hybrid_dgfem")

    # 4. Configure the Hybrid DG-FEM solver
    # This involves setting the crossover frequency and configuring the
    # individual solvers (DG-FEM, ISM, Ray Tracing).
    print("Configuring Hybrid DG-FEM solver...")
    hybrid_config = HybridDGFEMConfig(
        crossover_freq=350,  # Hz
        transition_time=0.08,  # s
        dgfem_config=DGFEMConfig(
            polynomial_order=2,
            mesh_resolution=0.5,  # Lower resolution for faster demo
            cfl_number=0.4
        ),
        ism_config={'max_order': 4},
        raytracer_config={'num_rays': 5000}  # Lower ray count for faster demo
    )

    # 5. Initialize the solver
    solver = HybridDGFEMMethod(room, hybrid_config)

    # 6. Compute the Room Impulse Response (RIR)
    print(f"Computing RIR for source: '{room.sources[0].name}' to receiver: '{mic.name}'...")
    rir = solver.compute_rir(duration=0.5)

    print(f"\nFinal Hybrid RIR computed with {len(rir)} samples.")
    print(f"Energy: {np.sum(rir**2):.6f}")

    # 7. Save RIR
    rir_filename = os.path.join(output_path, "rir_hybrid_dgfem.wav")
    sf.write(rir_filename, rir, room.fs)
    print(f"RIR saved to {rir_filename}")
    print("\nTo generate audio, convolve this RIR with a dry audio signal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a Hybrid DG-FEM simulation.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
                        help="Type of microphone to use ('mono' or 'ambisonic').")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Output directory for saving files.")
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir)
