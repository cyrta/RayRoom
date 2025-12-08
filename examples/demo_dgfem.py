import argparse
import os
import sys

import numpy as np
import soundfile as sf
from rayroom.engines.dgfem.dgfem import DGFEMSolver, DGFEMConfig
from demo_utils import (
    create_demo_room,
    generate_layouts,
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main(mic_type='mono', output_dir='outputs'):
    """
    A demonstration of the Discontinuous Galerkin Finite Element Method (DG-FEM) engine.
    """
    print("="*70)
    print("DG-FEM Acoustic Solver Demo")
    print("="*70)

    # 1. Create a room from demo_utils
    # We use a lower sampling rate for DGFEM for performance reasons in this demo.
    room, _, mic = create_demo_room(mic_type=mic_type)
    room.fs = 16000

    # 2. Create output directory if it doesn't exist
    output_path = os.path.join(output_dir, 'dgfem', mic_type)
    os.makedirs(output_path, exist_ok=True)

    # 3. Save layout visualization
    generate_layouts(room, output_path, "dgfem")

    # 4. Configure the DG-FEM solver
    # The polynomial order and mesh resolution are key parameters.
    # Higher order and finer resolution increase accuracy but also computational cost.
    print("Configuring DG-FEM solver...")
    config = DGFEMConfig(
        polynomial_order=2,
        use_gpu=False,  # Set to True if CuPy is available and a GPU is present
        cfl_number=0.3,
        mesh_resolution=0.4  # meters
    )

    # 5. Initialize the solver
    # This will create the tetrahedral mesh from the room geometry.
    solver = DGFEMSolver(room, config)

    # 6. Compute the Room Impulse Response (RIR)
    # The DGFEM solver uses the first source and receiver defined in the room.
    print(f"Computing RIR for source: '{room.sources[0].name}' to receiver: '{mic.name}'...")
    rir = solver.compute_rir(duration=0.1)

    print(f"\nRIR computed with {len(rir)} samples.")
    print(f"Energy: {np.sum(rir**2):.6f}")

    # 7. Save RIR
    rir_filename = os.path.join(output_path, "rir_dgfem.wav")
    sf.write(rir_filename, rir, room.fs)
    print(f"RIR saved to {rir_filename}")
    print("\nTo generate audio, convolve this RIR with a dry audio signal.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a DG-FEM simulation.")
    parser.add_argument('--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
                        help="Type of microphone to use ('mono' or 'ambisonic').")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Output directory for saving files.")
    args = parser.parse_args()

    main(mic_type=args.mic, output_dir=args.output_dir)
