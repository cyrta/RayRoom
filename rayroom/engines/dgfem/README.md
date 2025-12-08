# Discontinuous Galerkin Finite Element Method (DG-FEM) Engine

This engine implements a time-domain acoustic wave propagation solver based on the Discontinuous Galerkin Finite Element Method (DG-FEM). It is a wave-based method that numerically solves the acoustic wave equation, making it highly accurate for low-to-mid frequencies where phenomena like diffraction and room modes are prominent.

Unlike geometric methods, DG-FEM discretizes the entire room volume into a mesh of small elements (tetrahedra) and solves for the sound pressure at nodes within these elements. This approach captures the full wave physics of sound propagation.

### Block Diagram

```mermaid
graph LR
    A[Input: Room Geometry, Source/Receiver] --> B{Generate Tetrahedral Mesh};
    B --> C{Define Nodal Basis Functions};
    C --> D{Time-Stepping Loop};
    D -- For each time step --> E{Compute Volume Integrals (Derivatives)};
    E --> F{Compute Surface Integrals (Flux)};
    F --> G{Update Pressure & Velocity (RK4)};
    G -- Record pressure at receiver --> H[Construct RIR];
    D --> H;
    H --> I[Final RIR];
```

## Core Equations & Principles

The DG-FEM solver directly discretizes and solves the first-order, time-domain acoustic wave equations, which describe the relationship between acoustic pressure $p$ and particle velocity $\mathbf{v}$:

$$
\frac{\partial p}{\partial t} + \rho_0 c^2 \nabla \cdot \mathbf{v} = 0
$$

$$
\frac{\partial \mathbf{v}}{\partial t} + \frac{1}{\rho_0} \nabla p = 0
$$

where:
-   $p$ is the acoustic pressure.
-   $\mathbf{v}$ is the particle velocity vector.
-   $\rho_0$ is the density of the medium (air).
-   $c$ is the speed of sound.
-   $\rho_0 c^2$ is the bulk modulus of the medium.

The Discontinuous Galerkin method allows the solution to be discontinuous across element boundaries, which provides flexibility for meshing complex geometries and adapting the polynomial order of the basis functions ($p$-adaptivity).

## Implementation Details

-   **Meshing:** The room volume is discretized into an unstructured mesh of tetrahedra using the Delaunay triangulation algorithm. The mesh density is controlled by a `mesh_resolution` parameter.
-   **Basis Functions:** High-order Lagrange polynomials are used as basis functions within each element on a set of optimized Warp & Blend nodes. The `polynomial_order` parameter controls the accuracy of the solution.
-   **Time Integration:** An explicit fourth-order Runge-Kutta scheme is used for stable and accurate time-stepping. The time step size is determined by the CFL condition to ensure numerical stability.
-   **Numerical Flux:** An upwind flux (Lax-Friedrichs) is used to compute the interaction between adjacent, discontinuous elements, ensuring correct wave propagation across the mesh.
-   **GPU Acceleration:** The engine can leverage CuPy for GPU acceleration, which significantly speeds up the computationally intensive parts of the simulation.

## Usage

The `DGFEMSolver` class is the main interface for this engine. It takes a `Room` object and a `DGFEMConfig` object for configuration. The `compute_rir` method runs the simulation and returns the Room Impulse Response.

```python
import numpy as np
from rayroom.room.base import Room, SoundSource, Microphone
from rayroom.engines.dgfem.dgfem import DGFEMSolver, DGFEMConfig

# Create a room
room_dim = [4, 5, 3]
room = Room(room_dim)

# Add a source and a microphone
room.add_source(SoundSource(position=[1, 1, 1.5]))
room.add_mic(Microphone(position=[3, 4, 1.5]))

# Configure the DG-FEM solver
config = DGFEMConfig(
    polynomial_order=2,
    mesh_resolution=0.5
)

# Initialize and run the solver
solver = DGFEMSolver(room, config)
rir = solver.compute_rir(duration=0.5)

# The result is the Room Impulse Response
print(f"RIR computed with {len(rir)} samples.")
```

## When to Use

This engine is particularly well-suited for:

- **Low to Mid Frequencies**: Accurate simulation of wave-based effects that are prominent at lower frequencies (e.g., below 1 kHz).
- **Small to Medium Rooms**: The computational cost of FEM-based methods increases with room size and frequency, making it ideal for smaller environments where wave effects are critical.
- **High-Accuracy Benchmarking**: Can be used as a reference for benchmarking other, more approximate methods like ray tracing.

## Seminal Papers

1.  **Käser, M., & Dumbser, M. (2006).** *An arbitrary high-order discontinuous Galerkin method for elastic waves on unstructured meshes—I. The two-dimensional isotropic case with exterior forces*. Geophysical Journal International, 166(2), 855-877.

```bibtex
@article{kaser2006arbitrary,
  title={An arbitrary high-order discontinuous {G}alerkin method for elastic waves on unstructured meshes—{I}. {T}he two-dimensional isotropic case with exterior forces},
  author={K{\"a}ser, Martin and Dumbser, Michael},
  journal={Geophysical Journal International},
  volume={166},
  number={2},
  pages={855--877},
  year={2006},
  publisher={Oxford University Press}
}
```

2.  **Hesthaven, J. S., & Warburton, T. (2008).** *Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications*. Springer Science & Business Media.

```bibtex
@book{hesthaven2008nodal,
  title={Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications},
  author={Hesthaven, Jan S and Warburton, Tim},
  year={2008},
  publisher={Springer Science \& Business Media}
}
```
