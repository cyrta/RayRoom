<p align="center">
  <img src="docs/_static/RayRoom.jpg" alt="RayRoom Logo" width="300">
</p>

A Python-based ray tracing acoustics simulator supporting complex room geometries, materials, and furniture.

[![PyPI version](https://badge.fury.io/py/rayroom.svg)](https://badge.fury.io/py/rayroom)
[![Documentation Status](https://readthedocs.org/projects/rayroom/badge/?version=latest)](https://rayroom.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Ray Tracing Engine**: Stochastic ray tracing for impulse response estimation.
- **Room Creation**: Create shoebox rooms or complex polygons from corner lists.
- **Materials**: Frequency-dependent absorption, transmission (transparency), and scattering coefficients.
- **Objects**: Support for furniture, people (blockers), sources, and receivers (microphones).
- **Transparency**: Walls can be partially transparent (transmission).

## Physics & Rendering

### Ray Tracing with Air Absorption
The simulation engine employs stochastic ray tracing to model sound propagation. Energy decay is modeled through:
1.  **Geometric Divergence**: Naturally handled by the divergence of rays from the source.
2.  **Air Absorption**: An explicit attenuation factor is applied to each ray segment based on distance. The implementation uses the full **ISO 9613-1** standard model to calculate the absorption coefficient based on frequency, **temperature**, **humidity**, and **pressure**.
    *   Formula: $E_{new} = E_{old} \cdot 10^{-\alpha \cdot d / 10}$
    *   Where $\alpha$ is the absorption coefficient derived from environmental conditions using ISO 9613-1.

### Deterministic Phase & Interference
The renderer converts the collected energy histogram into a Room Impulse Response (RIR) using one of two phase strategies:

1.  **Stochastic Phase (Default)**
    -   **Method**: Randomly assigns positive or negative polarity ($+1/-1$) to each acoustic impulse.
    -   **Purpose**: Models incoherent energy summation. This is accurate for high frequencies and complex reverberation where phase relationships are randomized.

2.  **Deterministic Phase (`interference=True`)**
    -   **Method**: Assigns a fixed positive polarity ($+1$) to all impulses.
    -   **Purpose**: Preserves precise path-length differences. When convolved with the audio signal, this allows for **coherent interference** (phase cancellation/reinforcement) and standing wave phenomena to emerge naturally from the geometry.


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
# Generates Impulse Response and convolves with input audio
outputs = renderer.render(n_rays=10000)

# Save Result
mixed_audio = outputs["Mic"]
if mixed_audio is not None:
    wavfile.write("output.wav", 44100, (mixed_audio * 32767).astype(np.int16))
```

### Complex Geometry

See `examples/polygon_room.py` for creating rooms from 2D floor plans.

## Structure

- `rayroom/core.py`: Main simulation engine.
- `rayroom/room.py`: Room and wall definitions.
- `rayroom/objects.py`: Source, Receiver, Furniture classes.
- `rayroom/materials.py`: Material properties.
- `rayroom/geometry.py`: Vector math and intersection tests.
- `rayroom/audio.py`: Audio rendering and processing.
- `rayroom/physics.py`: Acoustic physics models.
- `rayroom/visualize.py`: Visualization tools.

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
