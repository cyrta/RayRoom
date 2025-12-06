<p align="center">
  <img src="https://raw.githubusercontent.com/RayRoom/RayRoom/refs/heads/main/docs/_static/RayRoom.jpg" alt="RayRoom Logo" width="300">
</p>

A Python-based ray tracing acoustics simulator supporting complex room geometries, materials, and furniture.

[![PyPI version](https://badge.fury.io/py/rayroom.svg)](https://badge.fury.io/py/rayroom)
[![Documentation Status](https://readthedocs.org/projects/rayroom/badge/?version=latest)](https://rayroom.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Ray Tracing Engine**: Stochastic ray tracing for late reflections.
- **Image Source Method (ISM)**: Deterministic early specular reflections.
- **Acoustic Radiosity**: Modeling of diffuse field energy exchange between surfaces.
- **FDTD Wave Solver**: Accurate low-frequency wave simulation using Finite Difference Time Domain.
- **Hybrid Rendering**: Combines methods for optimal accuracy and performance (ISM + Ray Tracing).
- **Spectral Rendering**: Frequency-dependent strategy using Wave equation for low frequencies and Geometric methods for high frequencies.
- **Room Creation**: Create shoebox rooms or complex polygons from corner lists.
- **Materials**: Frequency-dependent absorption, transmission (transparency), and scattering coefficients.
- **Objects**: Support for furniture, people (blockers), sources, and receivers (microphones).

## Physics & Rendering

RayRoom implements multiple rendering strategies to model sound propagation accurately across different frequency bands and acoustic phenomena.

### 1. Ray Tracing (Stochastic)
Models sound as particles (rays) that bounce around the room.
- **Geometric Divergence**: Naturally handled by the divergence of rays.
- **Air Absorption**: ISO 9613-1 standard model based on temperature, humidity, and pressure.
- **Scattering**: Walls scatter rays based on their scattering coefficient.

### 2. Image Source Method (ISM)
Models specular reflections by mirroring sources across boundaries.
- **Purpose**: Captures precise early reflections (echoes) that are crucial for spatial perception.
- **Deterministic**: Unlike ray tracing, it finds exact reflection paths up to a specified order.

### 3. Acoustic Radiosity
A patch-based energy exchange method.
- **Purpose**: Models the late diffuse reverberation field.
- **Mechanism**: Divides walls into patches and solves the energy transfer matrix (view factors) to simulate diffuse inter-reflections.

### 4. Finite Difference Time Domain (FDTD)
Solves the acoustic wave equation on a 3D grid.
- **Purpose**: Accurate simulation of low-frequency phenomena like standing waves, diffraction, and interference which geometric methods (Ray/ISM) miss.
- **Mechanism**: Voxelizes the room and updates pressure fields over time steps.

### 5. Hybrid Engines
- **Hybrid Renderer**: Combines ISM (Early) and Ray Tracing (Late) for a complete impulse response.
- **Spectral Renderer**: Splits the audio spectrum. Uses **FDTD** for low frequencies (Wave physics) and **Hybrid Geometric** (ISM+Ray) for high frequencies.

## Installation

You can install RayRoom directly from PyPI:

```bash
pip install rayroom
```

Or install from source:

```bash
git clone https://github.com/rayroom/rayroom.git
cd rayroom
pip install -e .
```

## Comparison & Demonstrations

To hear the differences between **RayRoom**'s rendering engines, including examples featuring Donald Trump and Bill Gates, please visit our [Comparisons page](examples/README.md).

## Usage

### Simple Shoebox Room with Audio Rendering

```python
from rayroom import Room, Source, Receiver, AudioRenderer
import scipy.io.wavfile as wavfile
import numpy as np

# Create Room
room = Room.create_shoebox([5, 4, 3])

# Add Source and Receiver
source = Source("Speaker", [1, 1, 1.5])
room.add_source(source)
room.add_receiver(Receiver("Mic", [4, 3, 1.5]))

# Setup Audio Renderer
renderer = AudioRenderer(room, fs=44100)

# Assign Audio to Source (requires an input wav file)
renderer.set_source_audio(source, "input.wav")

# Run Simulation
outputs = renderer.render(n_rays=10000)

# Save Result
mixed_audio = outputs["Mic"]
if mixed_audio is not None:
    wavfile.write("output.wav", 44100, (mixed_audio * 32767).astype(np.int16))
```

### Hybrid Rendering (ISM + Ray Tracing)

Use the `HybridRenderer` to combine deterministic early reflections with stochastic late reverberation.

```python
from rayroom.hybrid import HybridRenderer

# ... setup room ...

renderer = HybridRenderer(room, fs=44100)
renderer.set_source_audio(source, "input.wav")

# ism_order=2 calculates exact reflections up to 2 bounces
outputs = renderer.render(n_rays=20000, ism_order=2)
```

### Spectral Rendering (Wave + Geometric)

Use the `SpectralRenderer` for high-fidelity simulation that accounts for wave physics at low frequencies.

```python
from rayroom.spectral import SpectralRenderer

# ... setup room ...

# crossover_freq determines where to switch from Wave to Geometric physics
renderer = SpectralRenderer(room, fs=44100, crossover_freq=1000)
renderer.set_source_audio(source, "input.wav")

outputs = renderer.render(rir_duration=1.0)
```

### Radiosity Rendering (ISM + Diffuse Energy)

Use `RadiosityRenderer` for smooth diffuse tails without ray sampling noise.

```python
from rayroom.render_radiosity import RadiosityRenderer

# ... setup room ...

renderer = RadiosityRenderer(room, fs=44100, patch_size=0.5)
renderer.set_source_audio(source, "input.wav")

outputs = renderer.render(ism_order=2)
```

### Complex Geometry

See `examples/polygon_room.py` for creating rooms from 2D floor plans.

## Structure

- `rayroom/core.py`: Main Ray Tracing engine.
- `rayroom/room.py`: Room and wall definitions.
- `rayroom/objects.py`: Source, Receiver, Furniture classes.
- `rayroom/materials.py`: Material properties.
- `rayroom/geometry.py`: Vector math and intersection tests.
- `rayroom/audio.py`: Audio rendering and processing.
- `rayroom/physics.py`: Acoustic physics models.
- `rayroom/visualize.py`: Visualization tools.
- `rayroom/ism.py`: Image Source Method engine.
- `rayroom/radiosity.py`: Acoustic Radiosity solver.
- `rayroom/render_radiosity.py`: Radiosity Renderer.
- `rayroom/fdtd.py`: FDTD Wave solver.
- `rayroom/hybrid.py`: Hybrid Geometric Renderer.
- `rayroom/spectral.py`: Spectral Hybrid Renderer.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

```plain
MIT License

Copyright (c) 2025 Yanis Labrak

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
